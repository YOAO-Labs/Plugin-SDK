from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .managed_pool import ResourceLease, _PoolEntryRetired

ResourceT = TypeVar("ResourceT")
StateT = TypeVar("StateT")
ResultT = TypeVar("ResultT")


async def _await_completion(future: asyncio.Future[ResultT]) -> ResultT:
    """Keep the caller alive until its owned work settles, even on cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
        except BaseException:
            if not future.done():
                raise
    if cancellation is not None:
        try:
            future.result()
        except BaseException as exc:
            if exc is not cancellation:
                raise cancellation from exc
        raise cancellation
    return future.result()


class TrackerSessionLost(KeyError):
    pass


@dataclass
class _Command(Generic[ResourceT, StateT]):
    run: Callable[[ResourceT, StateT], Awaitable[object]]
    future: asyncio.Future[object]


@dataclass
class _Session(Generic[ResourceT, StateT]):
    binding: tuple[str, ...]
    queue: asyncio.Queue[_Command[ResourceT, StateT] | None]
    task: asyncio.Task[None]
    last_used_at: float
    in_flight: int = 0
    closing: bool = False


class _OwnedState(Generic[ResourceT, StateT]):
    """The pool artifact roots the state until its one cleanup has succeeded."""

    def __init__(self, lease: ResourceLease[Any, Any], resource: ResourceT,
                 state: StateT, close: Callable[[ResourceT, StateT], Awaitable[None]],
                 invalidate: Callable[[], None]) -> None:
        self.lease = lease
        self.resource = resource
        self.state = state
        self.close = close
        self._invalidate = invalidate
        self.retired = False
        self._cleanup_task: asyncio.Task[None] | None = None

    def invalidate(self) -> None:
        self.retired = True
        self._invalidate()

    async def cleanup(self) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup())
        await _await_completion(self._cleanup_task)

    async def _cleanup(self) -> None:
        try:
            await self.close(self.resource, self.state)
            if getattr(self.lease.resource, "cleanup_uncertain", False) is True:
                raise RuntimeError("tracker session cleanup is uncertain")
        except BaseException:
            self.lease._mark_cleanup_uncertain()
            raise
        self.lease._detach_owned_state(self)
        # A callback wrapper may itself retain the model and accelerator state.
        self.lease = None  # type: ignore[assignment]
        self.resource = None  # type: ignore[assignment]
        self.state = None  # type: ignore[assignment]
        self.close = None  # type: ignore[assignment]
        self._invalidate = lambda: None


class TrackerSessionManager(Generic[ResourceT, StateT]):
    """Keep opaque state on its exact pool entry, borrowing only for dispatches."""

    def __init__(self, *, ttl_seconds: float = 300.0) -> None:
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, _Session[ResourceT, StateT]] = {}
        self._cleanup_failure: BaseException | None = None

    def _owner_finished(self, token: str, ready: asyncio.Future[None],
                        task: asyncio.Task[None]) -> None:
        failure = asyncio.CancelledError() if task.cancelled() else task.exception()
        if not ready.done():
            ready.set_exception(failure or TrackerSessionLost(token))
        self._sessions.pop(token, None)

    async def start(
        self,
        *,
        binding: tuple[str, ...],
        borrow: Callable[[], AbstractAsyncContextManager[tuple[ResourceLease[Any, Any], ResourceT]]],
        open_session: Callable[[ResourceT], Awaitable[StateT]],
        close_session: Callable[[ResourceT, StateT], Awaitable[None]],
    ) -> str:
        token = secrets.token_urlsafe(32)
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        queue: asyncio.Queue[_Command[ResourceT, StateT] | None] = asyncio.Queue()
        task = asyncio.create_task(self._serve(token, borrow, open_session, close_session, queue, ready))
        task.add_done_callback(lambda owner: self._owner_finished(token, ready, owner))
        session = _Session(binding, queue, task, time.monotonic(), in_flight=1)
        self._sessions[token] = session
        try:
            await _await_completion(ready)
        except BaseException:
            self._begin_close(session)
            await _await_completion(asyncio.gather(task, return_exceptions=True))
            raise
        finally:
            session.in_flight -= 1
            session.last_used_at = time.monotonic()
        return token

    async def call(self, token: str, binding: tuple[str, ...],
                   run: Callable[[ResourceT, StateT], Awaitable[ResultT]]) -> ResultT:
        session = self._get(token, binding)
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        session.last_used_at = time.monotonic()
        session.in_flight += 1
        try:
            session.queue.put_nowait(_Command(run=run, future=future))
            return await _await_completion(future)  # type: ignore[return-value]
        finally:
            session.in_flight -= 1
            session.last_used_at = time.monotonic()

    async def close(self, token: str, binding: tuple[str, ...]) -> None:
        session = self._get(token, binding)
        self._begin_close(session)
        await _await_completion(session.task)

    @staticmethod
    def _begin_close(session: _Session[ResourceT, StateT]) -> None:
        if not session.closing and not session.task.done():
            session.closing = True
            session.queue.put_nowait(None)

    async def _wait_closed(self, sessions: list[_Session[ResourceT, StateT]]) -> None:
        results = await _await_completion(
            asyncio.gather(*(session.task for session in sessions), return_exceptions=True)
        )
        if self._cleanup_failure is not None:
            raise self._cleanup_failure
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def expire(self) -> int:
        cutoff = time.monotonic() - self._ttl_seconds
        expired = [session for session in self._sessions.values()
                   if not session.closing and session.in_flight == 0
                   and session.last_used_at <= cutoff]
        for session in expired:
            self._begin_close(session)
        await self._wait_closed(expired)
        return len(expired)

    async def shutdown(self) -> None:
        """Close current owners after the caller has fenced new admission."""
        sessions = list(self._sessions.values())
        for session in sessions:
            self._begin_close(session)
        await self._wait_closed(sessions)

    def _get(self, token: str, binding: tuple[str, ...]) -> _Session[ResourceT, StateT]:
        session = self._sessions.get(token)
        if session is None or session.binding != binding or session.closing or session.task.done():
            raise TrackerSessionLost(token)
        return session

    def _invalidate(self, token: str) -> None:
        session = self._sessions.get(token)
        if session is not None:
            self._begin_close(session)

    async def _serve(
        self,
        token: str,
        borrow: Callable[[], AbstractAsyncContextManager[tuple[ResourceLease[Any, Any], ResourceT]]],
        open_session: Callable[[ResourceT], Awaitable[StateT]],
        close_session: Callable[[ResourceT, StateT], Awaitable[None]],
        queue: asyncio.Queue[_Command[ResourceT, StateT] | None],
        ready: asyncio.Future[None],
    ) -> None:
        owned: _OwnedState[ResourceT, StateT] | None = None
        try:
            async with borrow() as (lease, resource):
                state = await open_session(resource)
                owned = _OwnedState(lease, resource, state, close_session,
                                    lambda: self._invalidate(token))
                lease._attach_owned_state(owned)
            # Do not retain callback wrappers or model roots in this coroutine.
            del lease, resource, state, borrow, open_session, close_session
            if owned.retired:
                if not ready.done():
                    ready.set_exception(TrackerSessionLost(token))
                return
            if not ready.done():
                ready.set_result(None)
            while True:
                command = await queue.get()
                if command is None:
                    break
                if owned.retired:
                    command.future.set_exception(TrackerSessionLost(token))
                    break
                try:
                    async with owned.lease._borrow_entry():
                        result = await command.run(owned.resource, owned.state)
                except _PoolEntryRetired:
                    if not command.future.done():
                        command.future.set_exception(TrackerSessionLost(token))
                    break
                except BaseException as exc:
                    if not command.future.done():
                        command.future.set_exception(exc)
                    break
                if not command.future.done():
                    command.future.set_result(result)
                del command, result
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                raise
        finally:
            try:
                if owned is not None:
                    if owned.retired:
                        await owned.cleanup()
                    else:
                        try:
                            async with owned.lease._borrow_entry():
                                await owned.cleanup()
                        except _PoolEntryRetired:
                            # Retirement waits on this same cleanup before disposal.
                            await owned.cleanup()
            except BaseException as exc:
                if self._cleanup_failure is None:
                    self._cleanup_failure = exc
                raise
            finally:
                while not queue.empty():
                    command = queue.get_nowait()
                    if command is not None and not command.future.done():
                        command.future.set_exception(TrackerSessionLost(token))
