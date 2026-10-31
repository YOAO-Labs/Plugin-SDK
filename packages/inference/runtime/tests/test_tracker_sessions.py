import asyncio
from contextlib import asynccontextmanager

import pytest

from aap_backend_runtime import TrackerSessionLost, TrackerSessionManager


def test_tracker_session_holds_resource_and_checks_binding() -> None:
    async def run() -> None:
        events: list[str] = []

        @asynccontextmanager
        async def borrow():
            events.append("borrow")
            try:
                yield "resource"
            finally:
                events.append("release")

        manager = TrackerSessionManager[str, dict]()
        token = await manager.start(
            binding=("job-1", "forward"),
            borrow=borrow,
            open_session=lambda resource: _value({"resource": resource, "count": 0}),
            close_session=lambda _resource, state: _event(
                events, f"close:{state['count']}"
            ),
        )
        assert (
            await manager.call(
                token,
                ("job-1", "forward"),
                _increment,
            )
            == 1
        )
        with pytest.raises(TrackerSessionLost):
            await manager.call(token, ("job-2", "forward"), _increment)
        await manager.close(token, ("job-1", "forward"))
        assert events == ["borrow", "close:1", "release"]

    asyncio.run(run())


def test_tracker_session_expires_and_shutdown_releases_resources() -> None:
    async def run() -> None:
        events: list[str] = []

        @asynccontextmanager
        async def borrow():
            try:
                yield "resource"
            finally:
                events.append("release")

        manager = TrackerSessionManager[str, dict](ttl_seconds=0)
        token = await manager.start(
            binding=("job-1", "forward"),
            borrow=borrow,
            open_session=lambda _resource: _value({}),
            close_session=lambda _resource, _state: _event(events, "close"),
        )
        assert await manager.expire() == 1
        assert events == ["close", "release"]
        with pytest.raises(TrackerSessionLost):
            await manager.close(token, ("job-1", "forward"))

        await manager.start(
            binding=("job-2", "forward"),
            borrow=borrow,
            open_session=lambda _resource: _value({}),
            close_session=lambda _resource, _state: _event(events, "close"),
        )
        await manager.shutdown()
        assert events == ["close", "release", "close", "release"]

    asyncio.run(run())


def test_tracker_session_does_not_expire_during_command() -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        @asynccontextmanager
        async def borrow():
            yield "resource"

        async def slow_command(_resource, _state):
            started.set()
            await release.wait()
            return "done"

        manager = TrackerSessionManager[str, dict](ttl_seconds=0)
        token = await manager.start(
            binding=("job-1", "forward"),
            borrow=borrow,
            open_session=lambda _resource: _value({}),
            close_session=lambda _resource, _state: _value(None),
        )
        pending = asyncio.create_task(
            manager.call(token, ("job-1", "forward"), slow_command)
        )
        await started.wait()

        assert await manager.expire() == 0
        release.set()
        assert await pending == "done"
        assert await manager.expire() == 1

    asyncio.run(run())


def test_loop_shutdown_settles_start_before_owner_first_runs() -> None:
    async def run() -> None:
        events = []
        manager = TrackerSessionManager()
        parent = asyncio.current_task()
        starting = asyncio.create_task(manager.start(
            binding=("job",), borrow=lambda: _borrow(events, "resource"),
            open_session=lambda _resource: _value({}),
            close_session=lambda _resource, _state: _event(events, "closed"),
        ))

        def cancel_children():
            for task in asyncio.all_tasks():
                if task is not parent:
                    task.cancel()

        asyncio.get_running_loop().call_soon(cancel_children)
        with pytest.raises(asyncio.CancelledError):
            await starting
        await manager.shutdown()
        assert events == []

    asyncio.run(run())


def test_cancelled_start_waits_for_opened_owner_cleanup() -> None:
    async def run() -> None:
        events = []
        opening, open_gate = asyncio.Event(), asyncio.Event()
        closing, close_gate = asyncio.Event(), asyncio.Event()
        manager = TrackerSessionManager(ttl_seconds=0)

        async def open_session(_resource):
            opening.set()
            await open_gate.wait()
            events.append("opened")
            return {}

        async def close_session(_resource, _state):
            closing.set()
            await close_gate.wait()
            events.append("closed")

        start = asyncio.create_task(manager.start(
            binding=("job",), borrow=lambda: _borrow(events, "resource"),
            open_session=open_session, close_session=close_session,
        ))
        await opening.wait()
        start.cancel()
        await asyncio.sleep(0)
        assert not start.done()
        assert await manager.expire() == 0
        open_gate.set()
        await closing.wait()
        start.cancel()
        await asyncio.sleep(0)
        assert not start.done()
        assert events == ["borrow:resource", "opened"]
        close_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await start
        assert events == ["borrow:resource", "opened", "closed", "release:resource"]
        await manager.shutdown()

    asyncio.run(asyncio.wait_for(run(), 10))


@pytest.mark.parametrize("phase", ["queued", "running"])
def test_cancelled_call_keeps_its_execution_owned(phase) -> None:
    async def run() -> None:
        events = []
        first_started, first_gate = asyncio.Event(), asyncio.Event()
        second_started, second_gate = asyncio.Event(), asyncio.Event()
        caller_started = asyncio.Event()
        manager = TrackerSessionManager(ttl_seconds=0)
        token = await manager.start(
            binding=("job",), borrow=lambda: _borrow(events, "resource"),
            open_session=lambda _resource: _value({}),
            close_session=lambda _resource, _state: _event(events, "closed"),
        )

        async def first_command(_resource, _state):
            first_started.set()
            await first_gate.wait()
            return "first"

        async def second_command(_resource, _state):
            second_started.set()
            await second_gate.wait()
            events.append("execution_finished")
            return "second"

        async def second_call():
            caller_started.set()
            return await manager.call(token, ("job",), second_command)

        first = None
        if phase == "queued":
            first = asyncio.create_task(manager.call(token, ("job",), first_command))
            await first_started.wait()
        second = asyncio.create_task(second_call())
        await caller_started.wait()
        if phase == "running":
            await second_started.wait()
        second.cancel()
        await asyncio.sleep(0)
        assert not second.done()
        assert await manager.expire() == 0
        if first is not None:
            first_gate.set()
            assert await first == "first"
        await second_started.wait()
        assert not second.done()
        second_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await second
        events.append("caller_cancelled")
        await manager.close(token, ("job",))
        assert events == [
            "borrow:resource", "execution_finished", "caller_cancelled",
            "closed", "release:resource",
        ]

    asyncio.run(asyncio.wait_for(run(), 10))


def test_failed_command_settles_queued_call_after_cleanup() -> None:
    async def run() -> None:
        events = []
        first_started, first_gate = asyncio.Event(), asyncio.Event()
        caller_started = asyncio.Event()
        closing, close_gate = asyncio.Event(), asyncio.Event()
        manager = TrackerSessionManager()

        async def close_session(_resource, _state):
            closing.set()
            await close_gate.wait()
            events.append("closed")

        token = await manager.start(
            binding=("job",), borrow=lambda: _borrow(events, "resource"),
            open_session=lambda _resource: _value({}), close_session=close_session,
        )

        async def fail(_resource, _state):
            first_started.set()
            await first_gate.wait()
            raise ValueError("inference failed")

        async def queued_call():
            caller_started.set()
            return await manager.call(
                token, ("job",),
                lambda _resource, _state: _event(events, "unexpected_execution"),
            )

        first = asyncio.create_task(manager.call(token, ("job",), fail))
        await first_started.wait()
        second = asyncio.create_task(queued_call())
        await caller_started.wait()
        first_gate.set()
        with pytest.raises(ValueError):
            await first
        await closing.wait()
        assert not second.done()
        close_gate.set()
        with pytest.raises(TrackerSessionLost):
            await second
        await manager.shutdown()
        assert events == ["borrow:resource", "closed", "release:resource"]

    asyncio.run(asyncio.wait_for(run(), 10))


@pytest.mark.parametrize("retire", ["close", "expire", "shutdown"])
def test_cleanup_failure_survives_retirement_and_waits_other_owners(retire) -> None:
    async def run() -> None:
        events = []
        second_closing, second_gate = asyncio.Event(), asyncio.Event()
        failure = RuntimeError("session cleanup failed")
        manager = TrackerSessionManager(ttl_seconds=0)

        async def close_session(resource, _state):
            if resource == "first":
                events.append("failed_close")
                raise failure
            second_closing.set()
            await second_gate.wait()
            events.append("second_closed")

        first = await manager.start(
            binding=("first",), borrow=lambda: _borrow(events, "first"),
            open_session=lambda _resource: _value({}), close_session=close_session,
        )
        await manager.start(
            binding=("second",), borrow=lambda: _borrow(events, "second"),
            open_session=lambda _resource: _value({}), close_session=close_session,
        )
        if retire == "close":
            with pytest.raises(RuntimeError) as closed:
                await manager.close(first, ("first",))
            assert closed.value is failure
            pending = asyncio.create_task(manager.shutdown())
        else:
            pending = asyncio.create_task(getattr(manager, retire)())
        await second_closing.wait()
        assert not pending.done()
        second_gate.set()
        with pytest.raises(RuntimeError) as retired:
            await pending
        assert retired.value is failure
        assert events == [
            "borrow:first", "borrow:second", "failed_close", "release:first",
            "second_closed", "release:second",
        ]
        with pytest.raises(RuntimeError) as repeated:
            await manager.shutdown()
        assert repeated.value is failure

    asyncio.run(asyncio.wait_for(run(), 10))


def test_cancelled_close_remains_visible_to_shutdown() -> None:
    async def run() -> None:
        events = []
        closing, close_gate = asyncio.Event(), asyncio.Event()
        manager = TrackerSessionManager()

        async def close_session(_resource, _state):
            closing.set()
            await close_gate.wait()
            events.append("closed")

        token = await manager.start(
            binding=("job",), borrow=lambda: _borrow(events, "resource"),
            open_session=lambda _resource: _value(None), close_session=close_session,
        )
        pending_close = asyncio.create_task(manager.close(token, ("job",)))
        await closing.wait()
        pending_close.cancel()
        pending_shutdown = asyncio.create_task(manager.shutdown())
        await asyncio.sleep(0)
        assert not pending_close.done()
        assert not pending_shutdown.done()
        with pytest.raises(TrackerSessionLost):
            await manager.call(token, ("job",), _increment)
        close_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await pending_close
        await pending_shutdown
        assert events == ["borrow:resource", "closed", "release:resource"]

    asyncio.run(asyncio.wait_for(run(), 10))


def test_failed_open_releases_borrow_without_poisoning_retirement() -> None:
    async def run() -> None:
        events = []
        manager = TrackerSessionManager()

        async def fail(_resource):
            raise FileNotFoundError("input unavailable")

        with pytest.raises(FileNotFoundError):
            await manager.start(
                binding=("job",), borrow=lambda: _borrow(events, "resource"),
                open_session=fail,
                close_session=lambda _resource, _state: _event(events, "unexpected_close"),
            )
        await manager.shutdown()
        assert events == ["borrow:resource", "release:resource"]

    asyncio.run(asyncio.wait_for(run(), 10))


@asynccontextmanager
async def _borrow(events, name):
    events.append(f"borrow:{name}")
    try:
        yield name
    finally:
        events.append(f"release:{name}")


async def _value(value):
    return value


async def _increment(_resource, state):
    state["count"] += 1
    return state["count"]


async def _event(events, value):
    events.append(value)
