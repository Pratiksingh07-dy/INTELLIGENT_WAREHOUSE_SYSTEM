
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ArmStatus(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    MAINTENANCE = "maintenance"


@dataclass
class RoboticArm:
    arm_id: int
    status: ArmStatus = ArmStatus.IDLE
    current_task: Optional[int] = None       # order_id being processed
    remaining_time: float = 0.0              # time left to finish current task
    total_busy_time: float = 0.0
    total_time: float = 0.0
    tasks_completed: int = 0
    energy_used: float = 0.0

    @property
    def utilization(self) -> float:
        if self.total_time <= 0:
            return 0.0
        return round(min(1.0, self.total_busy_time / self.total_time), 3)

    def is_available(self) -> bool:
        return self.status == ArmStatus.IDLE

    def assign_task(self, order_id: int, processing_time: float) -> bool:
        if not self.is_available():
            return False
        self.status = ArmStatus.PROCESSING
        self.current_task = order_id
        self.remaining_time = max(0.1, processing_time)
        return True

    def step(self, dt: float = 1.0) -> Optional[int]:
        """
        Advance the arm by dt time units.
        Returns the completed order_id if a task finished this step, else None.
        """
        self.total_time += dt
        completed_order = None
        if self.status == ArmStatus.PROCESSING:
            self.total_busy_time += dt
            self.energy_used += 1.2 * dt  # active arms consume more energy
            self.remaining_time -= dt
            if self.remaining_time <= 0:
                completed_order = self.current_task
                self.tasks_completed += 1
                self.current_task = None
                self.status = ArmStatus.IDLE
                self.remaining_time = 0.0
        else:
            self.energy_used += 0.1 * dt  # idle power draw
        return completed_order

    def to_dict(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "status": self.status.value,
            "current_task": self.current_task,
            "remaining_time": round(max(0.0, self.remaining_time), 2),
            "utilization": self.utilization,
            "tasks_completed": self.tasks_completed,
            "energy_used": round(self.energy_used, 2),
        }
