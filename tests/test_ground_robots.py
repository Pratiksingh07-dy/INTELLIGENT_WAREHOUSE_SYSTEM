# Import the GroundRobot class and RobotStatus enum from the ground_robot module.
from simulation.ground_robot import GroundRobot, RobotStatus


# Test that a new robot starts idle and is located in storage.
def test_robot_starts_idle_at_storage():

    # Create a new ground robot with ID 1.
    robot = GroundRobot(robot_id=1)

    # Check that the robot starts with idle status.
    assert robot.status == RobotStatus.IDLE

    # Check that the robot starts in the storage zone.
    assert robot.location == "storage"


# Test that assigning a transport task changes the robot status to moving.
def test_assign_transport_moves_to_moving_status():

    # Create a new ground robot with ID 1.
    robot = GroundRobot(robot_id=1)

    # Assign order 5 to travel to the dispatch zone.
    ok = robot.assign_transport(order_id=5, destination="dispatch", travel_time=2.0)

    # Check that the task was assigned successfully.
    assert ok is True

    # Check that the robot is now moving.
    assert robot.status == RobotStatus.MOVING

    # Check that the correct destination was stored.
    assert robot.destination == "dispatch"


# Test that the complete transport cycle finishes successfully.
def test_full_transport_cycle_completes():

    # Create a new ground robot with ID 1.
    robot = GroundRobot(robot_id=1)

    # Assign order 5 to travel to the dispatch zone.
    robot.assign_transport(order_id=5, destination="dispatch", travel_time=1.0)

    # First step: finishes moving -> transitions to loading
    completed = robot.step(dt=1.5)

    # Check that the order is not completed yet.
    assert completed is None

    # Check that the robot is now loading.
    assert robot.status == RobotStatus.LOADING

    # Check that the robot has reached the dispatch zone.
    assert robot.location == "dispatch"

    # Second step: finishes loading -> task completed
    completed = robot.step(dt=1.0)

    # Check that order 5 has been completed.
    assert completed == 5

    # Check that the robot is idle again.
    assert robot.status == RobotStatus.IDLE


# Test that the robot battery decreases while it is busy.
def test_battery_drains_while_busy():

    # Create a new ground robot with ID 1.
    robot = GroundRobot(robot_id=1)

    # Assign order 1 to travel to the packing zone.
    robot.assign_transport(order_id=1, destination="packing", travel_time=5.0)

    # Store the initial battery level.
    initial_battery = robot.battery

    # Advance the robot by one second.
    robot.step(dt=1.0)

    # Check that the battery level has decreased.
    assert robot.battery < initial_battery


# Test that low battery causes the robot to start charging.
def test_low_battery_triggers_charging():

    # Create a robot with a low battery level.
    robot = GroundRobot(robot_id=1, battery=10.0)

    # Advance the robot by one second.
    robot.step(dt=1.0)

    # Check that the robot is now charging.
    assert robot.status == RobotStatus.CHARGING


# Test that charging increases the robot battery level.
def test_charging_increases_battery():

    # Create a robot with a low battery and charging status.
    robot = GroundRobot(robot_id=1, battery=10.0, status=RobotStatus.CHARGING)

    # Store the initial battery level.
    initial = robot.battery

    # Advance the charging process by one second.
    robot.step(dt=1.0)

    # Check that the battery level has increased.
    assert robot.battery > initial


# Test that the robot is unavailable when its battery is too low.
def test_unavailable_when_battery_too_low():

    # Create a robot with a very low battery level.
    robot = GroundRobot(robot_id=1, battery=5.0)

    # Check that the robot is not available.
    assert robot.is_available() is False
