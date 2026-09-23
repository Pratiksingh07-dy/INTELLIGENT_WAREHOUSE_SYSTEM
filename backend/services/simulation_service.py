from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from rl.environment.warehouse_env import WarehouseEnv
from simulation.warehouse import (ACTION_ASSIGN_ARM, ACTION_ASSIGN_DRONE,
                                   ACTION_ASSIGN_ROBOT, ACTION_NOOP, ACTION_PRIORITIZE)

# maximum number of history values stored
HISTORY_LEN = 300


class SimulationService:
    def __init__(self):
        # create a long-running environment for live visualization
        self.env = WarehouseEnv(max_steps=10 ** 9)  # effectively unbounded for live viz
        self.env.reset(seed=42)
        # protect simulation state from concurrent access
        self.lock = threading.RLock()

        # track simulation control state
        self.running = False
        self.speed = 2.0  # simulation steps per second
        self.policy_table: Optional[np.ndarray] = None
        self.algorithm_name = "Rule-based heuristic (untrained)"

        # store recent simulation metrics
        self.history = {
            "reward": deque(maxlen=HISTORY_LEN),
            "queue_length": deque(maxlen=HISTORY_LEN),
            "waiting_time": deque(maxlen=HISTORY_LEN),
            "throughput": deque(maxlen=HISTORY_LEN),
            "energy": deque(maxlen=HISTORY_LEN),
            "utilization": deque(maxlen=HISTORY_LEN),
            "time": deque(maxlen=HISTORY_LEN),
        }

        # start the background simulation thread
        self._stop_flag = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # choose an action using the rule-based policy
    def _heuristic_action(self) -> int:
        
        # access the current warehouse simulation
        sim = self.env.sim

        # assign available robots to pending transport tasks
        if sim.transport_queue and any(r.is_available() for r in sim.robots):
            return ACTION_ASSIGN_ROBOT

        # assign available arms to pending orders
        if sim.queue and any(a.is_available() for a in sim.arms):
            if sim.has_high_priority_waiting() and sim.orders[sim.queue[0]].priority.value != "high":
                return ACTION_PRIORITIZE
            return ACTION_ASSIGN_ARM

        # assign available drones when no higher priority task exists
        if any(d.is_available() for d in sim.drones):
            return ACTION_ASSIGN_DRONE

        # perform no action when no resource is available
        return ACTION_NOOP

    def _choose_action(self) -> int:
        # use the learned policy when available
        if self.policy_table is not None:
            state = self.env.encode_state()
            return int(np.argmax(self.policy_table[state]))

        # use the heuristic policy before training
        return self._heuristic_action()

    def _loop(self):
        # continuously run the live simulation
        while not self._stop_flag:
            if self.running:
                with self.lock:
                    # choose and execute the next warehouse action
                    action = self._choose_action()
                    self.env.step(action)

                    # store current simulation metrics
                    self._record_history()

                # control the simulation speed
                time.sleep(max(0.02, 1.0 / max(self.speed, 0.1)))
            else:
                time.sleep(0.1)

    def _record_history(self):
        # get current warehouse state and metrics
        d = self.env.sim.to_dict()
        m = d["metrics"]

        # record recent performance values
        self.history["reward"].append(m["last_reward"])
        self.history["queue_length"].append(m["pending_orders"])
        self.history["waiting_time"].append(m["average_waiting_time"])
        self.history["throughput"].append(m["throughput"])
        self.history["energy"].append(m["energy_used"])
        self.history["utilization"].append(m["resource_utilization"])
        self.history["time"].append(d["time"])

    # ------------------------------------------------------------------ #
    # Public control API
    # ------------------------------------------------------------------ #
    def start(self):
        # start the warehouse simulation
        with self.lock:
            self.running = True

    def pause(self):
        # pause the warehouse simulation
        with self.lock:
            self.running = False

    def reset(self, arrival_rate: Optional[float] = None):
        # reset the simulation while preserving its running state
        with self.lock:
            was_running = self.running
            self.running = False
            seed = int(time.time()) % 100000

            # recreate the environment with the requested arrival rate
            if arrival_rate is not None:
                self.env.sim.order_generator.set_arrival_rate(arrival_rate)
                self.env = WarehouseEnv(max_steps=10 ** 9, arrival_rate=arrival_rate)
            else:
                self.env = WarehouseEnv(max_steps=10 ** 9)

            # reset environment and clear previous history
            self.env.reset(seed=seed)
            for k in self.history:
                self.history[k].clear()

            # restore previous running state
            self.running = was_running

    def set_speed(self, speed: float):
        # update simulation speed within allowed limits
        with self.lock:
            self.speed = max(0.1, min(speed, 20.0))

    def set_arrival_rate(self, rate: float):
        # update the warehouse order arrival rate
        with self.lock:
            self.env.sim.order_generator.set_arrival_rate(rate)

    def generate_orders(self, count: int):
        # generate and add new orders to the warehouse queue
        with self.lock:
            sim = self.env.sim
            for _ in range(count):
                order = sim.order_generator.generate_order(sim.time)
                sim.orders[order.order_id] = order
                sim.queue.append(order.order_id)
                sim.total_orders_generated += 1

    def set_policy(self, policy_table: Optional[np.ndarray], algorithm_name: str):
        # activate a trained policy for the simulation
        with self.lock:
            self.policy_table = policy_table
            self.algorithm_name = algorithm_name

    def get_state(self) -> dict:
        # return the current simulation state for the API
        with self.lock:
            d = self.env.sim.to_dict()
            d["running"] = self.running
            d["speed"] = self.speed
            d["active_algorithm"] = self.algorithm_name
            d["history"] = {k: list(v) for k, v in self.history.items()}
            d["arrival_rate"] = self.env.sim.order_generator.arrival_rate
            d["state_description"] = self.env.state_description(self.env.encode_state())
            return d


# Single shared instance used by the whole backend (simple, explicit singleton -
# avoids the complexity of a full dependency-injection framework for this
# academic project).
simulation_service = SimulationService()