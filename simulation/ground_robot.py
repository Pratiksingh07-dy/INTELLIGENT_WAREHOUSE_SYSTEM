
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

ZONES = ["storage", "packing", "dispatch"]


class RobotStatus(str, Enum):
    IDLE = "idle"
    MOVING = "moving"
    LOADING = "loading"
    CHARGING = "charging"


@dataclass
class GroundRobot:
    robot_id: int
    location: str = "storage"
    destination: Optional[str] = None
    status: RobotStatus = RobotStatus.IDLE
    current_task: Optional[int] = None
    battery: float = 100.0
    remaining_time: float = 0.0
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
        return self.status == RobotStatus.IDLE and self.battery > 15.0

    def assign_transport(self, order_id: int, destination: str, travel_time: float) -> bool:
        if not self.is_available():
            return False
        self.status = RobotStatus.MOVING
        self.current_task = order_id
        self.destination = destination
        self.remaining_time = max(0.1, travel_time)
        return True

    def step(self, dt: float = 1.0) -> Optional[int]:
        """Advance the robot by dt time units; return completed order_id if any."""
        self.total_time += dt
        completed_order = None

        if self.status in (RobotStatus.MOVING, RobotStatus.LOADING):
            self.total_busy_time += dt
            self.energy_used += 1.0 * dt
            self.battery = max(0.0, self.battery - 0.8 * dt)
            self.remaining_time -= dt
            if self.remaining_time <= 0:
                if self.status == RobotStatus.MOVING:
                    self.location = self.destination or self.location
                    self.status = RobotStatus.LOADING
                    self.remaining_time = 0.5  # brief loading/unloading step
                else:
                    completed_order = self.current_task
                    self.tasks_completed += 1
                    self.current_task = None
                    self.destination = None
                    self.status = RobotStatus.IDLE
                    self.remaining_time = 0.0
        elif self.status == RobotStatus.CHARGING:
            self.battery = min(100.0, self.battery + 5.0 * dt)
            self.energy_used += 0.3 * dt
            if self.battery >= 95.0:
                self.status = RobotStatus.IDLE
        else:
            self.energy_used += 0.05 * dt
            if self.battery < 20.0:
                self.status = RobotStatus.CHARGING

        return completed_order

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "location": self.location,
            "destination": self.destination,
            "status": self.status.value,
            "current_task": self.current_task,
            "battery": round(self.battery, 1),
            "utilization": self.utilization,
            "tasks_completed": self.tasks_completed,
            "energy_used": round(self.energy_used, 2),
        }
