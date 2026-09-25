from __future__ import annotations

# Import dataclass to create a data-holding class.
from dataclasses import dataclass

# Import Enum to create fixed status values.
from enum import Enum

# Import Optional to allow a variable to have a value or None.
from typing import Optional


# Define the possible statuses of a robotic arm.
class ArmStatus(str, Enum):
    # The arm is not currently performing a task.
    IDLE = "idle"

    # The arm is currently processing a task.
    PROCESSING = "processing"

    # The arm is currently under maintenance.
    MAINTENANCE = "maintenance"


# Create a data class to store information about one robotic arm.
@dataclass
class RoboticArm:
    # Unique ID number of the robotic arm.
    arm_id: int

    # Current status of the robotic arm.
    status: ArmStatus = ArmStatus.IDLE

    # ID of the order currently being processed.
    current_task: Optional[int] = None       # order_id being processed

    # Amount of time remaining for the current task.
    remaining_time: float = 0.0              # time left to finish current task

    # Total time the arm has been busy.
    total_busy_time: float = 0.0

    # Total tracked time for the arm.
    total_time: float = 0.0

    # Number of tasks completed by the arm.
    tasks_completed: int = 0

    # Total energy used by the arm.
    energy_used: float = 0.0

    # Calculate the arm's utilization.
    @property
    def utilization(self) -> float:
        # Return zero if no time has been recorded.
        if self.total_time <= 0:
            return 0.0

        # Calculate the percentage of time the arm was busy.
        return round(min(1.0, self.total_busy_time / self.total_time), 3)

    # Check whether the arm is available for a new task.
    def is_available(self) -> bool:
        # The arm is available only when it is idle.
        return self.status == ArmStatus.IDLE

    # Assign a processing task to the robotic arm.
    def assign_task(self, order_id: int, processing_time: float) -> bool:
        # Do not assign the task if the arm is not available.
        if not self.is_available():
            return False

        # Change the arm status to processing.
        self.status = ArmStatus.PROCESSING

        # Store the order ID as the current task.
        self.current_task = order_id

        # Store the time required to process the task.
        self.remaining_time = max(0.1, processing_time)

        # Confirm that the task was assigned successfully.
        return True

    # Move the arm forward by a specific amount of time.
    def step(self, dt: float = 1.0) -> Optional[int]:
        """
        Advance the arm by dt time units.
        Returns the completed order_id if a task finished this step, else None.
        """
        # Add the elapsed time to the arm's total time.
        self.total_time += dt

        # Assume no order has been completed yet.
        completed_order = None

        # Check whether the arm is currently processing.
        if self.status == ArmStatus.PROCESSING:

            # Add the elapsed time to the arm's busy time.
            self.total_busy_time += dt

            # Add the energy used while processing.
            self.energy_used += 1.2 * dt  # active arms consume more energy

            # Reduce the remaining processing time.
            self.remaining_time -= dt

            # Check whether the processing task is complete.
            if self.remaining_time <= 0:

                # Store the completed order ID.
                completed_order = self.current_task

                # Increase the number of completed tasks.
                self.tasks_completed += 1

                # Remove the current task.
                self.current_task = None

                # Make the arm idle again.
                self.status = ArmStatus.IDLE

                # Reset the remaining task time.
                self.remaining_time = 0.0

        # Handle the arm when it is not processing.
        else:

            # Add the small amount of energy used while idle.
            self.energy_used += 0.1 * dt  # idle power draw

        # Return the completed order ID, if any.
        return completed_order

    # Convert the robotic arm information into a dictionary.
    def to_dict(self) -> dict:
        return {
            # Store the arm's ID.
            "arm_id": self.arm_id,

            # Store the arm's status as text.
            "status": self.status.value,

            # Store the current task.
            "current_task": self.current_task,

            # Store the remaining task time.
            "remaining_time": round(max(0.0, self.remaining_time), 2),

            # Store the arm's utilization.
            "utilization": self.utilization,

            # Store the number of completed tasks.
            "tasks_completed": self.tasks_completed,

            # Store the total energy used.
            "energy_used": round(self.energy_used, 2),
        }
