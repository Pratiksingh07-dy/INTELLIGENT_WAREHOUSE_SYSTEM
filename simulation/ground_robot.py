from __future__ import annotations

# Import dataclass to create a data-holding class.
from dataclasses import dataclass

# Import Enum to create fixed status values.
from enum import Enum

# Import Optional to allow a variable to have a value or None.
from typing import Optional

# List of all zones where robots can operate.
ZONES = ["storage", "packing", "dispatch"]


# Define the possible statuses of a robot.
class RobotStatus(str, Enum):
    # Robot is not currently performing a task.
    IDLE = "idle"

    # Robot is moving to a destination.
    MOVING = "moving"

    # Robot is loading or unloading an order.
    LOADING = "loading"

    # Robot is charging its battery.
    CHARGING = "charging"


# Create a data class to store information about one ground robot.
@dataclass
class GroundRobot:
    # Unique ID number of the robot.
    robot_id: int

    # Current location of the robot.
    location: str = "storage"

    # Destination where the robot needs to go.
    destination: Optional[str] = None

    # Current status of the robot.
    status: RobotStatus = RobotStatus.IDLE

    # ID of the order currently assigned to the robot.
    current_task: Optional[int] = None

    # Current battery level of the robot.
    battery: float = 100.0

    # Time remaining for the current task.
    remaining_time: float = 0.0

    # Total time the robot has been busy.
    total_busy_time: float = 0.0

    # Total tracked time for the robot.
    total_time: float = 0.0

    # Number of tasks completed by the robot.
    tasks_completed: int = 0

    # Total energy used by the robot.
    energy_used: float = 0.0

    # Calculate the robot's utilization.
    @property
    def utilization(self) -> float:
        # Return zero if no time has been recorded.
        if self.total_time <= 0:
            return 0.0

        # Calculate the percentage of time the robot was busy.
        return round(min(1.0, self.total_busy_time / self.total_time), 3)

    # Check whether the robot is available for a new task.
    def is_available(self) -> bool:
        # Robot must be idle and have more than 15% battery.
        return self.status == RobotStatus.IDLE and self.battery > 15.0

    # Assign a transportation task to the robot.
    def assign_transport(self, order_id: int, destination: str, travel_time: float) -> bool:
        # Do not assign the task if the robot is not available.
        if not self.is_available():
            return False

        # Change the robot status to moving.
        self.status = RobotStatus.MOVING

        # Store the order ID as the current task.
        self.current_task = order_id

        # Store the destination of the robot.
        self.destination = destination

        # Store the time required to travel.
        self.remaining_time = max(0.1, travel_time)

        # Confirm that the task was assigned successfully.
        return True

    # Move the robot forward by a specific amount of time.
    def step(self, dt: float = 1.0) -> Optional[int]:
        """Advance the robot by dt time units; return completed order_id if any."""
        # Add the elapsed time to the robot's total time.
        self.total_time += dt

        # Assume no order has been completed yet.
        completed_order = None

        # Check whether the robot is moving or loading.
        if self.status in (RobotStatus.MOVING, RobotStatus.LOADING):

            # Add the elapsed time to the robot's busy time.
            self.total_busy_time += dt

            # Add energy used while moving or loading.
            self.energy_used += 1.0 * dt

            # Reduce the robot's battery.
            self.battery = max(0.0, self.battery - 0.8 * dt)

            # Reduce the remaining task time.
            self.remaining_time -= dt

            # Check whether the current step has finished.
            if self.remaining_time <= 0:

                # Check whether the robot has finished moving.
                if self.status == RobotStatus.MOVING:

                    # Update the robot's location to the destination.
                    self.location = self.destination or self.location

                    # Change the robot status to loading.
                    self.status = RobotStatus.LOADING

                    # Give a short amount of time for loading or unloading.
                    self.remaining_time = 0.5  # brief loading/unloading step

                # Otherwise, the loading or unloading is complete.
                else:

                    # Store the completed order ID.
                    completed_order = self.current_task

                    # Increase the number of completed tasks.
                    self.tasks_completed += 1

                    # Remove the current task.
                    self.current_task = None

                    # Remove the destination.
                    self.destination = None

                    # Make the robot idle.
                    self.status = RobotStatus.IDLE

                    # Reset the remaining task time.
                    self.remaining_time = 0.0

        # Check whether the robot is charging.
        elif self.status == RobotStatus.CHARGING:

            # Increase the battery while charging.
            self.battery = min(100.0, self.battery + 5.0 * dt)

            # Add the energy used while charging.
            self.energy_used += 0.3 * dt

            # Make the robot idle after reaching 95% battery.
            if self.battery >= 95.0:
                self.status = RobotStatus.IDLE

        # Handle the robot when it is idle.
        else:

            # Add a small amount of energy used while idle.
            self.energy_used += 0.05 * dt

            # Start charging when the battery becomes low.
            if self.battery < 20.0:
                self.status = RobotStatus.CHARGING

        # Return the completed order ID, if an order was completed.
        return completed_order

    # Convert the robot information into a dictionary.
    def to_dict(self) -> dict:
        return {
            # Store the robot's ID.
            "robot_id": self.robot_id,

            # Store the robot's current location.
            "location": self.location,

            # Store the robot's destination.
            "destination": self.destination,

            # Store the robot's status as text.
            "status": self.status.value,

            # Store the current task.
            "current_task": self.current_task,

            # Store the battery rounded to one decimal place.
            "battery": round(self.battery, 1),

            # Store the robot's utilization.
            "utilization": self.utilization,

            # Store the number of completed tasks.
            "tasks_completed": self.tasks_completed,

            # Store the total energy used.
            "energy_used": round(self.energy_used, 2),
        }
