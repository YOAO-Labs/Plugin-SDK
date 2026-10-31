import asyncio
import gc
import threading
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest

from aap_backend_runtime import (
    BuildArtifact,
    ManagedLruPool,
    ManagedPoolBusyError,
    TrackerSessionLost,
    TrackerSessionManager,
)


@dataclass
class Resource:
    key: str
    device: str = "cpu"
    cleanup_uncertain: bool = False


def make_pool(events):
    def build(key):
        events.append(("build", key))
        return BuildArtifact(Resource(key))

    return ManagedLruPool(1, build, str, lambda: events.append(("dispose",)))


def borrow(pool, key):
    @asynccontextmanager
    async def context():
        async with pool.borrow(key) as lease:
            yield lease, lease.resource
    return context


async def open_state(resource):
    return {"key": resource.key, "count": 0}


async def increment(resource, state):
    assert resource.key == state["key"]
    state["count"] += 1
    return state["count"]


def test_dispatches_release_borrower_and_close_once():
    async def run():
        events = []
        pool = make_pool(events)
        manager = TrackerSessionManager()

        async def close(resource, state):
            events.append(("close", resource.key, state["count"]))

        token = await manager.start(binding=("job",), borrow=borrow(pool, "model"),
                                    open_session=open_state, close_session=close)
        assert (await pool.snapshot())["borrowers"] == 0
        assert await manager.call(token, ("job",), increment) == 1
        assert (await pool.snapshot())["borrowers"] == 0
        assert await manager.call(token, ("job",), increment) == 2
        with pytest.raises(TrackerSessionLost):
            await manager.call(token, ("other",), increment)
        await manager.close(token, ("job",))
        assert (await pool.snapshot())["borrowers"] == 0
        with pytest.raises(TrackerSessionLost):
            await manager.close(token, ("job",))
        await pool.shutdown()
        assert events.count(("close", "model", 2)) == 1
        assert events.index(("close", "model", 2)) < events.index(("dispose",))

    asyncio.run(run())


@pytest.mark.parametrize("retire", ["lru", "idle", "drain", "shutdown"])
def test_pool_retirement_invalidates_and_cleans_before_disposal(retire):
    async def run():
        events = []
        pool = make_pool(events)
        manager = TrackerSessionManager()
        cleaning, gate = asyncio.Event(), asyncio.Event()

        async def close(resource, state):
            cleaning.set()
            await gate.wait()
            events.append(("close", resource.key, state["count"]))

        token = await manager.start(binding=("job",), borrow=borrow(pool, "model"),
                                    open_session=open_state, close_session=close)
        if retire == "lru":
            retirement = asyncio.create_task(pool.warmup("other"))
        elif retire == "idle":
            retirement = asyncio.create_task(pool.unload_idle(idle_before=time.monotonic() + 1))
        elif retire == "drain":
            retirement = asyncio.create_task(pool.unload_all(reason="drain"))
        else:
            retirement = asyncio.create_task(pool.shutdown())
        await cleaning.wait()
        with pytest.raises(TrackerSessionLost):
            await manager.call(token, ("job",), increment)
        assert not retirement.done()
        assert ("dispose",) not in events
        gate.set()
        await retirement
        assert events.count(("close", "model", 0)) == 1
        assert events.index(("close", "model", 0)) < events.index(("dispose",))
        if retire == "lru":
            await pool.warmup("model")
            with pytest.raises(TrackerSessionLost):
                await manager.call(token, ("job",), increment)
            assert events.count(("build", "model")) == 2
        await manager.shutdown()
        await pool.shutdown()

    asyncio.run(asyncio.wait_for(run(), 10))


def test_cancelled_command_waits_for_work_and_release():
    async def run():
        events = []
        pool = make_pool(events)
        manager = TrackerSessionManager(ttl_seconds=0)
        started, gate = asyncio.Event(), asyncio.Event()

        async def close(resource, state):
            events.append(("close", resource.key, state["count"]))

        async def slow(resource, state):
            started.set()
            await gate.wait()
            return await increment(resource, state)

        token = await manager.start(binding=("job",), borrow=borrow(pool, "model"),
                                    open_session=open_state, close_session=close)
        call = asyncio.create_task(manager.call(token, ("job",), slow))
        await started.wait()
        call.cancel()
        await asyncio.sleep(0)
        assert not call.done()
        assert (await pool.snapshot())["borrowers"] == 1
        assert await manager.expire() == 0
        with pytest.raises(ManagedPoolBusyError):
            await pool.unload_all(reason="drain")
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert (await pool.snapshot())["borrowers"] == 0
        assert await manager.expire() == 1
        assert events.count(("close", "model", 1)) == 1
        await pool.shutdown()

    asyncio.run(asyncio.wait_for(run(), 10))


def test_cancelled_start_waits_for_open_and_cleanup():
    async def run():
        events = []
        pool = make_pool(events)
        manager = TrackerSessionManager()
        opening, open_gate = asyncio.Event(), asyncio.Event()
        closing, close_gate = asyncio.Event(), asyncio.Event()

        async def opening_session(resource):
            opening.set()
            await open_gate.wait()
            return await open_state(resource)

        async def close(resource, state):
            closing.set()
            await close_gate.wait()
            events.append(("close", resource.key, state["count"]))

        start = asyncio.create_task(manager.start(binding=("job",), borrow=borrow(pool, "model"),
                            open_session=opening_session, close_session=close))
        await opening.wait()
        start.cancel()
        open_gate.set()
        await closing.wait()
        assert not start.done()
        close_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await start
        assert (await pool.snapshot())["borrowers"] == 0
        assert events.count(("close", "model", 0)) == 1
        await manager.shutdown()
        await pool.shutdown()

    asyncio.run(asyncio.wait_for(run(), 10))


@pytest.mark.parametrize("failure", ["raise", "uncertain"])
@pytest.mark.parametrize("retire", ["close", "lru", "drain"])
def test_cleanup_failure_never_reports_unloaded(failure, retire):
    async def run():
        events = []
        pool = make_pool(events)
        manager = TrackerSessionManager()
        attempts = 0

        async def close(resource, state):
            nonlocal attempts
            attempts += 1
            if failure == "raise":
                raise RuntimeError("reset failed")
            resource.cleanup_uncertain = True

        token = await manager.start(binding=("job",), borrow=borrow(pool, "model"),
                                    open_session=open_state, close_session=close)
        with pytest.raises(RuntimeError):
            if retire == "close":
                await manager.close(token, ("job",))
            elif retire == "lru":
                await pool.warmup("other")
            else:
                await pool.unload_all(reason="drain")
        # Wait for a failed LRU builder's bookkeeping before examining its status.
        while (await pool.snapshot())["builders"]:
            await asyncio.sleep(0)
        snapshot = await pool.snapshot()
        assert snapshot["cleanup_failed"] is True
        assert snapshot["gpu_resident"] is not False
        with pytest.raises(RuntimeError):
            await pool.unload_all(reason="drain", force_cleanup=True)
        assert attempts == 1
        assert ("dispose",) not in events

    asyncio.run(asyncio.wait_for(run(), 10))


def test_prestart_cancelled_builder_retires_owned_state():
    async def run():
        events = []
        pool = make_pool(events)
        manager = TrackerSessionManager()

        async def close(resource, state):
            events.append(("close", resource.key, state["count"]))

        token = await manager.start(binding=("job",), borrow=borrow(pool, "model"),
                                    open_session=open_state, close_session=close)
        async with pool._lock:
            builder = pool._get_or_start_builder_locked("other")
            builder.cancel()
        while (await pool.snapshot())["builders"]:
            await asyncio.sleep(0)
        with pytest.raises(TrackerSessionLost):
            await manager.call(token, ("job",), increment)
        assert events.count(("close", "model", 0)) == 1
        assert events.index(("close", "model", 0)) < events.index(("dispose",))
        await manager.shutdown()
        await pool.shutdown()

    asyncio.run(asyncio.wait_for(run(), 10))


def test_cancelled_thread_work_settles_before_release():
    async def run():
        events = []
        pool = make_pool(events)
        manager = TrackerSessionManager()
        entered = asyncio.Event()
        gate = threading.Event()
        loop = asyncio.get_running_loop()

        async def close(resource, state):
            events.append(("close", resource.key, state["count"]))

        def work():
            loop.call_soon_threadsafe(entered.set)
            if not gate.wait(5):
                raise TimeoutError("worker was not released")
            return 42

        async def command(_resource, _state):
            return await asyncio.to_thread(work)

        token = await manager.start(binding=("job",), borrow=borrow(pool, "model"),
                                    open_session=open_state, close_session=close)
        call = asyncio.create_task(manager.call(token, ("job",), command))
        await entered.wait()
        call.cancel()
        await asyncio.sleep(0)
        assert not call.done()
        assert (await pool.snapshot())["borrowers"] == 1
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert (await pool.snapshot())["borrowers"] == 0
        await manager.close(token, ("job",))
        await pool.shutdown()

    asyncio.run(asyncio.wait_for(run(), 10))


def test_failed_command_invalidates_queued_call_after_cleanup():
    async def run():
        events = []
        pool = make_pool(events)
        manager = TrackerSessionManager()
        entered, gate = asyncio.Event(), asyncio.Event()
        closing, close_gate = asyncio.Event(), asyncio.Event()

        async def close(resource, state):
            closing.set()
            await close_gate.wait()
            events.append(("close", resource.key, state["count"]))

        async def fail(_resource, _state):
            entered.set()
            await gate.wait()
            raise ValueError("inference failed")

        token = await manager.start(binding=("job",), borrow=borrow(pool, "model"),
                                    open_session=open_state, close_session=close)
        first = asyncio.create_task(manager.call(token, ("job",), fail))
        await entered.wait()
        second = asyncio.create_task(manager.call(token, ("job",), increment))
        await asyncio.sleep(0)
        gate.set()
        with pytest.raises(ValueError, match="inference failed"):
            await first
        await closing.wait()
        assert not second.done()
        close_gate.set()
        with pytest.raises(TrackerSessionLost):
            await second
        await manager.shutdown()
        await pool.shutdown()
        assert events.count(("close", "model", 0)) == 1

    asyncio.run(asyncio.wait_for(run(), 10))


def test_cancelled_close_remains_owned_by_shutdown():
    async def run():
        events = []
        pool = make_pool(events)
        manager = TrackerSessionManager()
        closing, gate = asyncio.Event(), asyncio.Event()

        async def close(resource, state):
            closing.set()
            await gate.wait()
            events.append(("close", resource.key, state["count"]))

        token = await manager.start(binding=("job",), borrow=borrow(pool, "model"),
                                    open_session=open_state, close_session=close)
        pending = asyncio.create_task(manager.close(token, ("job",)))
        await closing.wait()
        pending.cancel()
        shutdown = asyncio.create_task(manager.shutdown())
        await asyncio.sleep(0)
        assert not pending.done()
        assert not shutdown.done()
        with pytest.raises(TrackerSessionLost):
            await manager.call(token, ("job",), increment)
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await shutdown
        assert (await pool.snapshot())["borrowers"] == 0
        assert events.count(("close", "model", 0)) == 1
        await pool.shutdown()

    asyncio.run(asyncio.wait_for(run(), 10))


def test_retiring_session_remains_counted_until_cleanup_finishes():
    async def run():
        events = []
        pool = make_pool(events)
        manager = TrackerSessionManager()
        closing, gate = asyncio.Event(), asyncio.Event()

        async def open_counted(resource):
            resource.active_sessions = 1
            return {}

        async def close(resource, _state):
            closing.set()
            await gate.wait()
            resource.active_sessions = 0

        await manager.start(binding=("job",), borrow=borrow(pool, "model"),
                            open_session=open_counted, close_session=close)
        assert (await pool.snapshot())["active_sessions"] == 1
        unload = asyncio.create_task(pool.unload_all(reason="drain"))
        await closing.wait()
        assert (await pool.snapshot())["active_sessions"] == 1
        gate.set()
        await unload
        assert (await pool.snapshot())["active_sessions"] == 0
        await manager.shutdown()
        await pool.shutdown()

    asyncio.run(asyncio.wait_for(run(), 10))


def test_close_drops_callback_resource_wrapper():
    async def run():
        events = []
        pool = make_pool(events)
        manager = TrackerSessionManager()
        wrapper_ref = None

        class Wrapper:
            def __init__(self, resource):
                self.resource = resource

        @asynccontextmanager
        async def wrapped_borrow():
            nonlocal wrapper_ref
            async with pool.borrow("model") as lease:
                wrapper = Wrapper(lease.resource)
                wrapper_ref = weakref.ref(wrapper)
                yield lease, wrapper

        async def open_wrapper(wrapper):
            return {"key": wrapper.resource.key}

        async def close_wrapper(wrapper, state):
            assert wrapper.resource.key == state["key"]

        token = await manager.start(binding=("job",), borrow=wrapped_borrow,
                                    open_session=open_wrapper, close_session=close_wrapper)
        assert wrapper_ref() is not None
        await manager.close(token, ("job",))
        gc.collect()
        assert wrapper_ref() is None
        await pool.shutdown()

    asyncio.run(asyncio.wait_for(run(), 10))


def test_failed_open_releases_borrow_without_poisoning_pool():
    async def run():
        events = []
        pool = make_pool(events)
        manager = TrackerSessionManager()

        async def failed_open(_resource):
            raise FileNotFoundError("input unavailable")

        async def unexpected_close(_resource, _state):
            raise AssertionError("failed open must not close a nonexistent state")

        with pytest.raises(FileNotFoundError):
            await manager.start(binding=("job",), borrow=borrow(pool, "model"),
                                open_session=failed_open, close_session=unexpected_close)
        assert (await pool.snapshot())["borrowers"] == 0
        await manager.shutdown()
        await pool.shutdown()

    asyncio.run(asyncio.wait_for(run(), 10))
