"""Tests for the UDP action bridge between teleop and Isaac Sim.

These run in-process on loopback -- no simulator and no camera needed. What
they pin down is the behaviour the Isaac side actually depends on: that an
action survives the wire intact, that a silent sender reads as "stop" rather
than "keep going", and that UDP reordering can't rewind the robot to an older
command.
"""

import time

import pytest

from simbiote.robot_iface.actions import GripperState, Pose, RobotAction
from simbiote.teleop.action_bridge import ActionReceiver, UdpActionSink

PORT = 47899


@pytest.fixture
def receiver():
    rx = ActionReceiver(host="127.0.0.1", port=PORT, stale_after=0.3)
    yield rx
    rx.close()


def _drain(rx, attempts=50):
    """Poll until a *newly arrived* datagram lands.

    Keyed off `rx.received` rather than "latest() is not None": the receiver
    keeps serving its last action until it goes stale, so a plain None-check
    returns the previous test phase's value the instant it is called and never
    waits for the datagram actually under test.
    """
    before = rx.received
    for _ in range(attempts):
        action = rx.latest()
        if action is not None and rx.received > before:
            return action
        time.sleep(0.01)
    return None


def _settle(rx, quiet_rounds=5):
    """Absorb everything still in flight so the next _drain sees only new sends.

    A burst of UDP frames arrives over several milliseconds; without this a
    later drain can trip on the tail of the previous burst.
    """
    quiet = 0
    while quiet < quiet_rounds:
        before = rx.received
        time.sleep(0.01)
        rx.latest()
        quiet = quiet + 1 if rx.received == before else 0


def test_action_survives_the_round_trip(receiver):
    sink = UdpActionSink(host="127.0.0.1", port=PORT)
    sent = RobotAction(
        base_velocity=(0.25, 0.0, -0.75),
        arm_target_pose=Pose(position=(0.4, -0.1, 0.55)),
        gripper_state=GripperState.CLOSED,
    )
    sink.apply_action(sent)

    got = _drain(receiver)
    assert got is not None
    assert got.base_velocity == pytest.approx(sent.base_velocity)
    assert got.arm_target_pose.position == pytest.approx(sent.arm_target_pose.position)
    assert got.gripper_state == GripperState.CLOSED
    sink.close()


def test_silence_reads_as_stale_so_the_base_stops(receiver):
    """The Isaac loop commands a full stop when latest() returns None.

    Without this a dropped teleop process would leave the robot coasting on
    its last velocity into a wall.
    """

    sink = UdpActionSink(host="127.0.0.1", port=PORT)
    sink.apply_action(RobotAction(base_velocity=(1.0, 0.0, 0.0)))
    assert _drain(receiver) is not None
    assert receiver.is_live

    time.sleep(0.35)  # longer than stale_after
    assert receiver.latest() is None
    assert not receiver.is_live
    sink.close()


def test_out_of_order_datagram_cannot_rewind_the_command(receiver):
    """UDP may reorder; an older action must never supersede a newer one."""

    sink = UdpActionSink(host="127.0.0.1", port=PORT)
    sink.apply_action(RobotAction(base_velocity=(0.1, 0.0, 0.0)))
    sink.apply_action(RobotAction(base_velocity=(0.2, 0.0, 0.0)))
    newest = _drain(receiver)
    assert newest.base_velocity[0] == pytest.approx(0.2)

    # Rewind this sender's counter, as a reordered datagram from the *same*
    # run would look. Same sender id, older sequence -> must be ignored.
    sink._sequence = 0
    sink.apply_action(RobotAction(base_velocity=(9.9, 0.0, 0.0)))
    time.sleep(0.05)

    assert receiver.latest().base_velocity[0] == pytest.approx(0.2)
    sink.close()


def test_restarted_teleop_is_not_locked_out(receiver):
    """A new teleop run restarts its sequence at 1 and must be accepted.

    Regression: the receiver's "never go backwards" rule was applied across
    senders, so after a long session the next run's low sequence numbers were
    all rejected -- teleop looked connected while the robot ignored it.
    """

    first = UdpActionSink(host="127.0.0.1", port=PORT)
    for _ in range(500):
        first.apply_action(RobotAction(base_velocity=(0.1, 0.0, 0.0)))
    assert _drain(receiver) is not None
    first.close()
    _settle(receiver)  # the rest of the burst is still on the wire

    # A fresh process: sequence back to 1, but a different sender id.
    second = UdpActionSink(host="127.0.0.1", port=PORT)
    assert second._sender_id != first._sender_id
    second.apply_action(RobotAction(base_velocity=(0.7, 0.0, 0.0)))

    got = _drain(receiver)
    assert got is not None, "restarted sender was locked out"
    assert got.base_velocity[0] == pytest.approx(0.7)
    second.close()


def test_sending_with_no_listener_does_not_raise():
    """Teleop must survive Isaac not being up yet, or exiting first."""

    sink = UdpActionSink(host="127.0.0.1", port=PORT + 1)
    for _ in range(5):
        sink.apply_action(RobotAction(base_velocity=(0.1, 0.0, 0.0)))
    sink.close()
