
from __future__ import annotations

# Import dataclass to create a simple data-holding class.
from dataclasses import dataclass

# Import Enum to create fixed status values.
from enum import Enum

# Import Optional to allow a variable to have a value or None.
from typing import Optional


# List of all zones where drones can operate.
ZONES = ["storage", "packing", "dispatch"]


# Define the possible statuses of a drone.
class DroneStatus(str, Enum):
    # Drone is ready to receive a task.
    AVAILABLE = "available"

    # Drone is currently scanning.
    SCANNING = "scanning"

    # Drone is currently monitoring.
    MONITORING = "monitoring"

    # Drone is currently charging its battery.
    CHARGING = "charging"


# Create a data class to store information about one drone.
@dataclass
class Drone:

    # Unique ID number of the drone.
    drone_id: int

    # Zone where the drone is currently located.
    current_zone: str = "storage"

    # Current status of the drone.
    status: DroneStatus = DroneStatus.AVAILABLE

    # Task currently assigned to the drone.
    current_task: Optional[str] = None

    # Current battery level of the drone.
    battery: float = 100.0

    # Time remaining for the current task.
    remaining_time: float = 0.0

    # Total time the drone has been busy.
    total_busy_time: float = 0.0

    # Total time tracked for the drone.
    total_time: float = 0.0

    # Number of tasks completed by the drone.
    tasks_completed: int = 0

    # Total energy used by the drone.
    energy_used: float = 0.0


    # Calculate how much of the total time the drone was busy.
    @property
    def utilization(self) -> float:

        # Avoid division by zero when no time has passed.
        if self.total_time <= 0:
            return 0.0

        # Calculate the percentage of time the drone was busy.
        return round(min(1.0, self.total_busy_time / self.total_time), 3)


    # Check whether the drone can receive a new task.
    def is_available(self) -> bool:

        # Drone must be available and have more than 10% battery.
        return self.status == DroneStatus.AVAILABLE and self.battery > 10.0


    # Assign a scanning task to the drone.
    def assign_scan(self, zone: str, duration: float) -> bool:

        # Do not assign the task if the drone is not available.
        if not self.is_available():
            return False

        # Change the drone status to scanning.
        self.status = DroneStatus.SCANNING

        # Move the drone to the selected zone.
        self.current_zone = zone

        # Store the current scanning task.
        self.current_task = f"scan:{zone}"

        # Store how much time the scan will take.
        self.remaining_time = max(0.1, duration)

        # Confirm that the task was assigned successfully.
        return True


    # Move the drone forward by a specific amount of time.
    def step(self, dt: float = 1.0) -> bool:

        # Advance the drone's tracked time.
        self.total_time += dt

        # Assume no task has been completed yet.
        completed = False


        # Check whether the drone is scanning or monitoring.
        if self.status in (DroneStatus.SCANNING, DroneStatus.MONITORING):

            # Add this time to the drone's busy time.
            self.total_busy_time += dt

            # Add energy used during scanning or monitoring.
            self.energy_used += 0.9 * dt

            # Reduce the battery during active work.
            self.battery = max(0.0, self.battery - 1.2 * dt)

            # Reduce the remaining task time.
            self.remaining_time -= dt


            # Check whether the task has finished.
            if self.remaining_time <= 0:

                # Mark the task as completed.
                completed = True

                # Increase the number of completed tasks.
                self.tasks_completed += 1

                # Remove the current task.
                self.current_task = None

                # Make the drone available again.
                self.status = DroneStatus.AVAILABLE

                # Reset remaining task time.
                self.remaining_time = 0.0


        # Check whether the drone is charging.
        elif self.status == DroneStatus.CHARGING:

            # Increase the battery while charging.
            self.battery = min(100.0, self.battery + 6.0 * dt)

            # Add the small amount of energy used while charging.
            self.energy_used += 0.2 * dt

            # Make the drone available after reaching 95% battery.
            if self.battery >= 95.0:
                self.status = DroneStatus.AVAILABLE


        # This section handles an available or other idle drone.
        else:

            # Add a small amount of energy used while idle.
            self.energy_used += 0.05 * dt

            # Start charging when the battery becomes low.
            if self.battery < 15.0:
                self.status = DroneStatus.CHARGING


        # Return whether a task was completed during this step.
        return completed


    # Convert the drone information into a dictionary.
    def to_dict(self) -> dict:

        # Return the important drone information.
        return {
            # Store the drone's ID.
            "drone_id": self.drone_id,

            # Store the drone's current zone.
            "current_zone": self.current_zone,

            # Store the drone's status as text.
            "status": self.status.value,

            # Store the current task.
            "current_task": self.current_task,

            # Store the battery rounded to one decimal place.
            "battery": round(self.battery, 1),

            # Store the drone's utilization.
            "utilization": self.utilization,

            # Store the number of completed tasks.
            "tasks_completed": self.tasks_completed,

            # Store the total energy used.
            "energy_used": round(self.energy_used, 2),
        }
