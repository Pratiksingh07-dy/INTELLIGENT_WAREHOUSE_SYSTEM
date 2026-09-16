from simulation.drone import Drone, DroneStatus


def test_drone_starts_available():
    drone = Drone(drone_id=1)
    assert drone.status == DroneStatus.AVAILABLE
    assert drone.is_available() is True


def test_assign_scan_sets_scanning_status():
    drone = Drone(drone_id=1)
    ok = drone.assign_scan(zone="storage", duration=2.0)
    assert ok is True
    assert drone.status == DroneStatus.SCANNING
    assert drone.current_zone == "storage"


def test_scan_completes_after_duration():
    drone = Drone(drone_id=1)
    drone.assign_scan(zone="packing", duration=2.0)
    completed = drone.step(dt=1.0)
    assert completed is False
    completed = drone.step(dt=1.5)
    assert completed is True
    assert drone.status == DroneStatus.AVAILABLE
    assert drone.tasks_completed == 1


def test_battery_drains_while_scanning():
    drone = Drone(drone_id=1)
    drone.assign_scan(zone="dispatch", duration=5.0)
    initial = drone.battery
    drone.step(dt=1.0)
    assert drone.battery < initial


def test_low_battery_triggers_charging():
    drone = Drone(drone_id=1, battery=8.0)
    drone.step(dt=1.0)
    assert drone.status == DroneStatus.CHARGING


def test_cannot_assign_when_unavailable():
    drone = Drone(drone_id=1, status=DroneStatus.SCANNING)
    ok = drone.assign_scan(zone="storage", duration=1.0)
    assert ok is False
