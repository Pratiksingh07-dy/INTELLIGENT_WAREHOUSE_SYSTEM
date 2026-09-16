from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class OrderStage(str, Enum):
    QUEUED = "queued"
    PICKING = "picking"          # being picked/packed by a robotic arm
    TRANSPORT = "transport"      # being carried by a ground robot
    DISPATCH_READY = "dispatch_ready"
    COMPLETED = "completed"


class OrderPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


PRODUCT_CATALOG = [
    "Widget-A", "Widget-B", "Gadget-C", "Component-D",
    "Package-E", "Module-F", "Part-G", "Kit-H",
]

_order_id_counter = itertools.count(1)


@dataclass
class Order:
    order_id: int
    product: str
    quantity: int
    priority: OrderPriority
    arrival_time: float
    processing_time: float
    stage: OrderStage = OrderStage.QUEUED
    assigned_resource: Optional[str] = None
    completion_time: Optional[float] = None
    wait_started: float = field(default=0.0)

    def waiting_time(self, current_time: float) -> float:
        """Time an order has spent waiting since arrival (or since completion)."""
        if self.stage == OrderStage.COMPLETED and self.completion_time is not None:
            return max(0.0, self.completion_time - self.arrival_time)
        return max(0.0, current_time - self.arrival_time)

    def to_dict(self, current_time: float) -> dict:
        return {
            "order_id": self.order_id,
            "product": self.product,
            "quantity": self.quantity,
            "priority": self.priority.value,
            "arrival_time": round(self.arrival_time, 2),
            "stage": self.stage.value,
            "waiting_time": round(self.waiting_time(current_time), 2),
            "assigned_resource": self.assigned_resource,
            "processing_time": round(self.processing_time, 2),
        }


class OrderGenerator:
    """
    Generates synthetic orders using configurable probability distributions.

    Parameters
    ----------
    arrival_rate : float
        Lambda (mean number of order arrivals) per simulation step,
        used to drive a Poisson process: P(k arrivals) = e^-l * l^k / k!
    high_priority_prob : float
        Probability that a newly generated order is HIGH priority
        (models urgent/rush orders).
    seed : Optional[int]
        RNG seed for reproducibility (useful for experiments/tests).
    """

    def __init__(self, arrival_rate: float = 0.45,
                 high_priority_prob: float = 0.2,
                 low_priority_prob: float = 0.3,
                 seed: Optional[int] = None):
        self.arrival_rate = arrival_rate
        self.high_priority_prob = high_priority_prob
        self.low_priority_prob = low_priority_prob
        self._rng = random.Random(seed)
        self._np_seed = seed

        import numpy as np
        self._np_rng = np.random.default_rng(seed)

    def set_arrival_rate(self, rate: float) -> None:
        self.arrival_rate = max(0.0, rate)

    def _sample_priority(self) -> OrderPriority:
        r = self._rng.random()
        if r < self.high_priority_prob:
            return OrderPriority.HIGH
        if r < self.high_priority_prob + self.low_priority_prob:
            return OrderPriority.LOW
        return OrderPriority.NORMAL

    def _sample_processing_time(self, priority: OrderPriority) -> float:
        # Triangular distribution: (low, high, mode) - simple, bounded,
        # and easy to reason about compared to a heavy-tailed distribution.
        base = self._rng.triangular(1.0, 6.0, 2.5)
        if priority == OrderPriority.HIGH:
            base *= 0.7  # urgent orders are expedited / smaller processing time
        return round(base, 2)

    def poisson_arrivals(self) -> int:
        """Sample the number of new orders arriving this step (Poisson process)."""
        return int(self._np_rng.poisson(lam=self.arrival_rate))

    def generate_order(self, current_time: float) -> Order:
        priority = self._sample_priority()
        order = Order(
            order_id=next(_order_id_counter),
            product=self._rng.choice(PRODUCT_CATALOG),
            quantity=self._rng.randint(1, 5),
            priority=priority,
            arrival_time=current_time,
            processing_time=self._sample_processing_time(priority),
        )
        return order

    def generate_batch(self, current_time: float) -> List[Order]:
        """Generate the batch of orders that arrive during this simulation step."""
        n_arrivals = self.poisson_arrivals()
        return [self.generate_order(current_time) for _ in range(n_arrivals)]
