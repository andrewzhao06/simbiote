"""Whoever drives IsaacBackend must pump MainThreadBridge from the main thread.

This is not a property of the bridge so much as a contract on its callers, and
breaking it produced a freeze that looked like a graphics problem: the window
kept its last frame and the robot stopped a second into the first command.

The shape is fixed by two constraints that cannot both be relaxed. Kit aborts
the process if the simulator is stepped from any thread but the one that
created it, so simulator work has to come back to the main thread. And
`task_executor` deliberately runs each skill on a `ThreadPoolExecutor` worker
so a wedged skill can be abandoned on timeout. The bridge is what joins them.

The consequence is that a driver may not sit on the main thread waiting for the
session to finish -- that is precisely the thread the session needs. It has to
run the session elsewhere and spend the main thread pumping. `hospital_demo.py`
(run_session_pumped) and `hospital_server.py` both do this; these tests pin the
requirement down so a third caller does not rediscover it the hard way.

No Isaac Sim needed: MainThreadBridge is plain stdlib.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import pytest

from simbiote.sim_env.isaac_nav import MainThreadBridge


def test_worker_blocks_until_the_main_thread_pumps():
    """Nobody pumping means the simulator work never runs at all.

    This is the freeze, reproduced: the skill worker parks in `call()` and the
    caller's `future.result()` expires without a single simulator step.
    """

    bridge = MainThreadBridge()
    executed = []

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(lambda: bridge.call(lambda: executed.append("step")))

        with pytest.raises(FutureTimeout):
            future.result(timeout=0.3)

        assert executed == [], "simulator work must not have run -- nothing pumped it"
    finally:
        # Release the parked worker before leaving. ThreadPoolExecutor's
        # threads are not daemons and the interpreter joins them at exit, so a
        # worker still blocked in call() would hang the whole test run -- which
        # is the same failure mode this test is about, arriving via pytest.
        while not bridge.pump():
            time.sleep(0.005)
        pool.shutdown(wait=True)

    assert executed == ["step"], "the pump in the cleanup should have run it"


def test_pumping_from_the_main_thread_lets_the_worker_finish():
    """The pattern hospital_demo/hospital_server use: worker + main-thread pump."""

    bridge = MainThreadBridge()
    executed = []
    box = {}

    def session():
        box["result"] = bridge.call(lambda: (executed.append("step"), "arrived")[1])

    worker = threading.Thread(target=session, daemon=True)
    worker.start()

    deadline = time.time() + 5
    while worker.is_alive() and time.time() < deadline:
        if not bridge.pump():
            time.sleep(0.005)
    worker.join(timeout=1)

    assert not worker.is_alive(), "worker should have been released by the pump"
    assert executed == ["step"]
    assert box["result"] == "arrived"


def test_call_on_the_main_thread_runs_inline():
    """Direct use without a driver around it must behave exactly as before."""

    bridge = MainThreadBridge()
    assert bridge.call(lambda: "inline") == "inline"
    assert bridge.pump() is False, "nothing should have been queued"


def test_exceptions_travel_back_to_the_calling_thread():
    """A failing skill must surface where it was called, not vanish in the pump."""

    bridge = MainThreadBridge()
    box = {}

    def session():
        try:
            bridge.call(lambda: (_ for _ in ()).throw(RuntimeError("collided")))
        except RuntimeError as exc:
            box["error"] = str(exc)

    worker = threading.Thread(target=session, daemon=True)
    worker.start()
    deadline = time.time() + 5
    while worker.is_alive() and time.time() < deadline:
        if not bridge.pump():
            time.sleep(0.005)
    worker.join(timeout=1)

    assert box.get("error") == "collided"
