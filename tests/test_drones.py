# Import the Drone class and DroneStatus enum from the drone module.
from simulation.drone import Drone, DroneStatus


# Test that a new drone starts as available.
def test_drone_starts_available():

    # Create a new drone with ID 1.
    drone = Drone(drone_id=1)

    # Check that the drone starts with available status.
    assert drone.status == DroneStatus.AVAILABLE

    # Check that the drone is available for a task.
    assert drone.is_available() is True


# Test that assigning a scan changes the drone status to scanning.
def test_assign_scan_sets_scanning_status():

    # Create a new drone with ID 1.
    drone = Drone(drone_id=1)

    # Assign a scanning task to the drone.
    ok = drone.assign_scan(zone="storage", duration=2.0)

    # Check that the task was assigned successfully.
    assert ok is True

    # Check that the drone is now scanning.
    assert drone.status == DroneStatus.SCANNING

    # Check that the drone is in the correct zone.
    assert drone.current_zone == "storage"


# Test that a scan is completed after the required duration.
def test_scan_completes_after_duration():

    # Create a new drone with ID 1.
    drone = Drone(drone_id=1)

    # Assign a scan in the packing zone.
    drone.assign_scan(zone="packing", duration=2.0)

    # Move the simulation forward by one second.
    completed = drone.step(dt=1.0)

    # Check that the scan is not completed yet.
    assert completed is False

    # Move the simulation forward by another 1.5 seconds.
    completed = drone.step(dt=1.5)

    # Check that the scan has now been completed.
    assert completed is True

    # Check that the drone is available again.
    assert drone.status == DroneStatus.AVAILABLE

    # Check that one task has been completed.
    assert drone.tasks_completed == 1


# Test that the battery decreases while the drone is scanning.
def test_battery_drains_while_scanning():

    # Create a new drone with ID 1.
    drone = Drone(drone_id=1)

    # Assign a scanning task in the dispatch zone.
    drone.assign_scan(zone="dispatch", duration=5.0)

    # Store the battery level before scanning.
    initial = drone.battery

    # Advance the simulation by one second.
    drone.step(dt=1.0)

    # Check that the battery has decreased.
    assert drone.battery < initial


# Test that low battery causes the drone to start charging.
def test_low_battery_triggers_charging():

    # Create a drone with a low battery level.
    drone = Drone(drone_id=1, battery=8.0)

    # Advance the simulation by one second.
    drone.step(dt=1.0)

    # Check that the drone is now charging.
    assert drone.status == DroneStatus.CHARGING


# Test that a task cannot be assigned when the drone is unavailable.
def test_cannot_assign_when_unavailable():

    # Create a drone that is already scanning.
    drone = Drone(drone_id=1, status=DroneStatus.SCANNING)

    # Try to assign another scanning task.
    ok = drone.assign_scan(zone="storage", duration=1.0)

    # Check that the task was not assigned.
    assert ok is False
