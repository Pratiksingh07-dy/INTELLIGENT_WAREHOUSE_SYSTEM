from __future__ import annotations

import random
from typing import Dict

from simulation.orders import PRODUCT_CATALOG


class Inventory:
    def __init__(self, seed: int = None):
        rng = random.Random(seed)
        self.stock: Dict[str, int] = {p: rng.randint(50, 200) for p in PRODUCT_CATALOG}
        self.last_scanned: Dict[str, float] = {p: 0.0 for p in PRODUCT_CATALOG}

    def consume(self, product: str, quantity: int) -> bool:
        if self.stock.get(product, 0) >= quantity:
            self.stock[product] -= quantity
            return True
        return False

    def restock(self, product: str, quantity: int) -> None:
        self.stock[product] = self.stock.get(product, 0) + quantity

    def mark_scanned(self, product: str, current_time: float) -> None:
        self.last_scanned[product] = current_time

    def average_staleness(self, current_time: float) -> float:
        if not self.last_scanned:
            return 0.0
        return sum(max(0.0, current_time - t) for t in self.last_scanned.values()) / len(self.last_scanned)

    def to_dict(self) -> dict:
        return {"stock": dict(self.stock)}
