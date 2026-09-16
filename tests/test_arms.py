from simulation.robotic_arm import ArmStatus, RoboticArm


def test_arm_starts_idle():
    arm = RoboticArm(arm_id=1)
    assert arm.status == ArmStatus.IDLE
    assert arm.is_available() is True


def test_assign_task_when_idle_succeeds():
    arm = RoboticArm(arm_id=1)
    ok = arm.assign_task(order_id=101, processing_time=2.0)
    assert ok is True
    assert arm.status == ArmStatus.PROCESSING
    assert arm.current_task == 101
    assert arm.is_available() is False


def test_assign_task_when_busy_fails():
    arm = RoboticArm(arm_id=1)
    arm.assign_task(order_id=101, processing_time=2.0)
    ok = arm.assign_task(order_id=102, processing_time=1.0)
    assert ok is False
    assert arm.current_task == 101  # unchanged


def test_step_completes_task_after_enough_time():
    arm = RoboticArm(arm_id=1)
    arm.assign_task(order_id=55, processing_time=2.0)
    completed = arm.step(dt=1.0)
    assert completed is None  # not finished yet
    completed = arm.step(dt=1.5)
    assert completed == 55
    assert arm.status == ArmStatus.IDLE
    assert arm.tasks_completed == 1


def test_utilization_increases_with_busy_time():
    arm = RoboticArm(arm_id=1)
    arm.assign_task(order_id=1, processing_time=10.0)
    for _ in range(5):
        arm.step(dt=1.0)
    assert 0.0 < arm.utilization <= 1.0


def test_energy_accumulates():
    arm = RoboticArm(arm_id=1)
    initial_energy = arm.energy_used
    arm.step(dt=1.0)
    assert arm.energy_used > initial_energy
