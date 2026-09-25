# Import the ArmStatus enum and RoboticArm class from the robotic_arm module.
from simulation.robotic_arm import ArmStatus, RoboticArm


# Test that a new robotic arm starts in the idle state.
def test_arm_starts_idle():

    # Create a new robotic arm with ID 1.
    arm = RoboticArm(arm_id=1)

    # Check that the arm starts with idle status.
    assert arm.status == ArmStatus.IDLE

    # Check that the arm is available for a task.
    assert arm.is_available() is True


# Test that assigning a task to an idle arm succeeds.
def test_assign_task_when_idle_succeeds():

    # Create a new robotic arm with ID 1.
    arm = RoboticArm(arm_id=1)

    # Assign order 101 to the robotic arm.
    ok = arm.assign_task(order_id=101, processing_time=2.0)

    # Check that the task was assigned successfully.
    assert ok is True

    # Check that the arm is now processing.
    assert arm.status == ArmStatus.PROCESSING

    # Check that the correct order is assigned.
    assert arm.current_task == 101

    # Check that the busy arm is no longer available.
    assert arm.is_available() is False


# Test that assigning a second task while the arm is busy fails.
def test_assign_task_when_busy_fails():

    # Create a new robotic arm with ID 1.
    arm = RoboticArm(arm_id=1)

    # Assign the first task to the arm.
    arm.assign_task(order_id=101, processing_time=2.0)

    # Try to assign another task while the arm is busy.
    ok = arm.assign_task(order_id=102, processing_time=1.0)

    # Check that the second task was not assigned.
    assert ok is False

    # Check that the original task is still assigned.
    assert arm.current_task == 101  # unchanged


# Test that the arm completes a task after enough processing time.
def test_step_completes_task_after_enough_time():

    # Create a new robotic arm with ID 1.
    arm = RoboticArm(arm_id=1)

    # Assign order 55 to the robotic arm.
    arm.assign_task(order_id=55, processing_time=2.0)

    # Process the task for one second.
    completed = arm.step(dt=1.0)

    # Check that the task is not finished yet.
    assert completed is None  # not finished yet

    # Process the task for another 1.5 seconds.
    completed = arm.step(dt=1.5)

    # Check that order 55 has been completed.
    assert completed == 55

    # Check that the arm is idle again.
    assert arm.status == ArmStatus.IDLE

    # Check that one task has been completed.
    assert arm.tasks_completed == 1


# Test that utilization increases when the arm is busy.
def test_utilization_increases_with_busy_time():

    # Create a new robotic arm with ID 1.
    arm = RoboticArm(arm_id=1)

    # Assign a long processing task to the arm.
    arm.assign_task(order_id=1, processing_time=10.0)

    # Advance the arm by one second five times.
    for _ in range(5):
        arm.step(dt=1.0)

    # Check that the utilization is greater than zero and not above 100%.
    assert 0.0 < arm.utilization <= 1.0


# Test that the energy usage increases over time.
def test_energy_accumulates():

    # Create a new robotic arm with ID 1.
    arm = RoboticArm(arm_id=1)

    # Store the initial energy usage.
    initial_energy = arm.energy_used

    # Advance the arm by one second.
    arm.step(dt=1.0)

    # Check that energy usage has increased.
    assert arm.energy_used > initial_energy
