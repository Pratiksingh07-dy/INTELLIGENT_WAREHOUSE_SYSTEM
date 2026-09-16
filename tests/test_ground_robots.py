from simulation.ground_robot import GroundRobot, RobotStatus


def test_robot_starts_idle_at_storage():
    robot = GroundRobot(robot_id=1)
    assert robot.status == RobotStatus.IDLE
    assert robot.location == "storage"


def test_assign_transport_moves_to_moving_status():
    robot = GroundRobot(robot_id=1)
    ok = robot.assign_transport(order_id=5, destination="dispatch", travel_time=2.0)
    assert ok is True
    assert robot.status == RobotStatus.MOVING
    assert robot.destination == "dispatch"


def test_full_transport_cycle_completes():
    robot = GroundRobot(robot_id=1)
    robot.assign_transport(order_id=5, destination="dispatch", travel_time=1.0)
    # First step: finishes moving -> transitions to loading
    completed = robot.step(dt=1.5)
    assert completed is None
    assert robot.status == RobotStatus.LOADING
    assert robot.location == "dispatch"
    # Second step: finishes loading -> task completed
    completed = robot.step(dt=1.0)
    assert completed == 5
    assert robot.status == RobotStatus.IDLE


def test_battery_drains_while_busy():
    robot = GroundRobot(robot_id=1)
    robot.assign_transport(order_id=1, destination="packing", travel_time=5.0)
    initial_battery = robot.battery
    robot.step(dt=1.0)
    assert robot.battery < initial_battery


def test_low_battery_triggers_charging():
    robot = GroundRobot(robot_id=1, battery=10.0)
    robot.step(dt=1.0)
    assert robot.status == RobotStatus.CHARGING


def test_charging_increases_battery():
    robot = GroundRobot(robot_id=1, battery=10.0, status=RobotStatus.CHARGING)
    initial = robot.battery
    robot.step(dt=1.0)
    assert robot.battery > initial


def test_unavailable_when_battery_too_low():
    robot = GroundRobot(robot_id=1, battery=5.0)
    assert robot.is_available() is False
