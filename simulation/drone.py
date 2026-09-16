
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

ZONES = ["storage", "packing", "dispatch"]


class DroneStatus(str, Enum):
    AVAILABLE = "available"
    SCANNING = "scanning"
    MONITORING = "monitoring"
    CHARGING = "charging"


@dataclass
class Drone:
    drone_id: int
    current_zone: str = "storage"
    status: DroneStatus = DroneStatus.AVAILABLE
    current_task: Optional[str] = None
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
        return self.status == DroneStatus.AVAILABLE and self.battery > 10.0

    def assign_scan(self, zone: str, duration: float) -> bool:
        if not self.is_available():
            return False
        self.status = DroneStatus.SCANNING
        self.current_zone = zone
        self.current_task = f"scan:{zone}"
        self.remaining_time = max(0.1, duration)
        return True

    def step(self, dt: float = 1.0) -> bool:
        """Advance drone by dt; return True if a scan/task completed this step."""
        self.total_time += dt
        completed = False

        if self.status in (DroneStatus.SCANNING, DroneStatus.MONITORING):
            self.total_busy_time += dt
            self.energy_used += 0.9 * dt
            self.battery = max(0.0, self.battery - 1.2 * dt)
            self.remaining_time -= dt
            if self.remaining_time <= 0:
                completed = True
                self.tasks_completed += 1
                self.current_task = None
                self.status = DroneStatus.AVAILABLE
                self.remaining_time = 0.0
        elif self.status == DroneStatus.CHARGING:
            self.battery = min(100.0, self.battery + 6.0 * dt)
            self.energy_used += 0.2 * dt
            if self.battery >= 95.0:
                self.status = DroneStatus.AVAILABLE
        else:
            self.energy_used += 0.05 * dt
            if self.battery < 15.0:
                self.status = DroneStatus.CHARGING

        return completed

    def to_dict(self) -> dict:
        return {
            "drone_id": self.drone_id,
            "current_zone": self.current_zone,
            "status": self.status.value,
            "current_task": self.current_task,
            "battery": round(self.battery, 1),
            "utilization": self.utilization,
            "tasks_completed": self.tasks_completed,
            "energy_used": round(self.energy_used, 2),
        }
