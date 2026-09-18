"""
warehouse_env.py
-----------------
Gymnasium-compatible environment that formalizes the warehouse resource
allocation problem as an MDP = (S, A, P, R, gamma).

STATE (discretized so tabular RL algorithms are tractable on a laptop):

    pending_bucket   in {0,1,2,3}   : queued+in-transport orders, capped/bucketed
    arms_busy        in {0..NUM_ARMS}
    robots_busy      in {0..NUM_ROBOTS}
    drones_busy      in {0..NUM_DRONES}
    wait_bucket      in {0,1,2}     : low / medium / high average waiting time
    priority_flag    in {0,1}       : whether a HIGH priority order is waiting

With NUM_ARMS = NUM_ROBOTS = NUM_DRONES = 2, the total number of discrete
states is 4 x 3 x 3 x 3 x 3 x 2 = 648, which is small enough for tabular
Dynamic Programming, Monte Carlo, TD, SARSA and Q-Learning to train in
seconds to minutes on a normal laptop, with NO neural networks and NO GPU.

ACTIONS: see simulation/warehouse.py (5 high-level scheduling actions).

This class is intentionally the ONLY place that converts between the
"raw" (continuous / dict-based) simulation state produced by
simulation.warehouse.Warehouse and the discrete (S, A) tuple used by
the RL algorithms - keeping the MDP formalization in one auditable file.
"""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation.orders import Order, OrderPriority, OrderStage
from simulation.robotic_arm import ArmStatus
from simulation.ground_robot import RobotStatus
from simulation.drone import DroneStatus
from simulation.warehouse import NUM_ACTIONS, Warehouse

# --- Discretization bucket sizes -------------------------------------------------
PENDING_BUCKETS = 4     # 0, 1, 2, 3+
WAIT_BUCKETS = 3        # low, medium, high
PRIORITY_BUCKETS = 2    # no high-priority waiting / at least one waiting

WAIT_LOW_THRESHOLD = 2.0
WAIT_HIGH_THRESHOLD = 5.0


class WarehouseEnv(gym.Env):
    """A high-level, tabular-friendly warehouse resource-allocation environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, num_arms: int = 2, num_robots: int = 2, num_drones: int = 2,
                 arrival_rate: float = 0.45, max_steps: int = 200,
                 seed: Optional[int] = None):
        super().__init__()
        self.num_arms = num_arms
        self.num_robots = num_robots
        self.num_drones = num_drones
        self.max_steps = max_steps
        self._seed = seed
        self._step_count = 0

        self.sim = Warehouse(num_arms=num_arms, num_robots=num_robots,
                              num_drones=num_drones, arrival_rate=arrival_rate, seed=seed)

        self._dims = (PENDING_BUCKETS, num_arms + 1, num_robots + 1,
                      num_drones + 1, WAIT_BUCKETS, PRIORITY_BUCKETS)
        self.n_states = int(np.prod(self._dims))
        self.n_actions = NUM_ACTIONS

        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[int, dict]:
        super().reset(seed=seed)
        self.sim.reset(seed=seed if seed is not None else self._seed)
        self._step_count = 0
        state = self.encode_state()
        return state, {"raw": self.sim.to_dict()}

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        reward = self.sim.step(int(action))
        self._step_count += 1
        next_state = self.encode_state()
        terminated = False  # this task is continuing/episodic-by-truncation
        truncated = self._step_count >= self.max_steps
        info = {"raw": self.sim.to_dict()}
        return next_state, reward, terminated, truncated, info

    def render(self):  # pragma: no cover - visualization happens in the web UI
        d = self.sim.to_dict()
        print(f"t={d['time']} pending={d['metrics']['pending_orders']} "
              f"reward={d['metrics']['last_reward']}")

    # ------------------------------------------------------------------ #
    # State encoding / decoding (the S in the MDP)
    # ------------------------------------------------------------------ #
    def _bucket_pending(self, n: int) -> int:
        return min(n, PENDING_BUCKETS - 1)

    def _bucket_wait(self, avg_wait: float) -> int:
        if avg_wait < WAIT_LOW_THRESHOLD:
            return 0
        if avg_wait < WAIT_HIGH_THRESHOLD:
            return 1
        return 2

    def encode_state(self) -> int:
        sim = self.sim
        pending = self._bucket_pending(len(sim.queue) + len(sim.transport_queue))
        arms_busy = sum(1 for a in sim.arms if a.status == ArmStatus.PROCESSING)
        robots_busy = sum(1 for r in sim.robots if r.status in
                           (RobotStatus.MOVING, RobotStatus.LOADING))
        drones_busy = sum(1 for d in sim.drones if d.status in
                           (DroneStatus.SCANNING, DroneStatus.MONITORING))
        wait_bucket = self._bucket_wait(sim.average_waiting_time())
        priority_flag = int(sim.has_high_priority_waiting())

        indices = (pending, arms_busy, robots_busy, drones_busy, wait_bucket, priority_flag)
        return int(np.ravel_multi_index(indices, self._dims))

    def decode_state(self, state_index: int) -> Tuple[int, int, int, int, int, int]:
        return tuple(int(x) for x in np.unravel_index(state_index, self._dims))

    def state_description(self, state_index: int) -> dict:
        pending, arms_busy, robots_busy, drones_busy, wait_bucket, priority = \
            self.decode_state(state_index)
        return {
            "pending_bucket": pending,
            "arms_busy": arms_busy,
            "robots_busy": robots_busy,
            "drones_busy": drones_busy,
            "wait_bucket": ["low", "medium", "high"][wait_bucket],
            "high_priority_waiting": bool(priority),
        }

    # ------------------------------------------------------------------ #
    # Used ONLY by Dynamic Programming methods, which require a model of
    # the environment P(s'|s,a) and R(s,a). Since the underlying simulator
    # is stochastic and continuous-time, we build an approximate discrete
    # model by resetting the simulator into a synthetic configuration that
    # matches a given discretized state and sampling real transitions from
    # it (see rl/utils.py: estimate_mdp_model). This keeps DP grounded in
    # the actual simulator dynamics rather than a hand-written table.
    # ------------------------------------------------------------------ #
    def set_discrete_state(self, state_index: int) -> None:
        pending, arms_busy, robots_busy, drones_busy, wait_bucket, priority_flag = \
            self.decode_state(state_index)

        self.sim.reset(seed=None)

        # Representative "pending" count for this bucket
        pending_count = {0: 0, 1: 1, 2: 2, 3: 4}[pending]
        wait_repr = {0: 0.5, 1: 3.5, 2: 7.5}[wait_bucket]

        self.sim.time = 10.0  # give some headroom so arrival_time - wait >= 0
        # Split the synthetic pending orders between the "waiting for an arm"
        # queue and the "waiting for a ground robot" transport queue, so that
        # BOTH ASSIGN_ARM and ASSIGN_ROBOT get meaningful, non-degenerate
        # samples during MDP model estimation (estimate_mdp_model). Without
        # this split, the transport queue would always be empty in the
        # synthetic states and Policy/Value Iteration would never learn that
        # ASSIGN_ROBOT is ever useful.
        num_to_arm_queue = (pending_count + 1) // 2
        for i in range(pending_count):
            priority = OrderPriority.HIGH if (priority_flag and i == 0) else OrderPriority.NORMAL
            to_arm_queue = i < num_to_arm_queue
            order = Order(
                order_id=-(i + 1),  # negative synthetic ids, won't collide with real orders
                product="Widget-A",
                quantity=1,
                priority=priority,
                arrival_time=self.sim.time - wait_repr,
                processing_time=2.0,
                stage=OrderStage.QUEUED if to_arm_queue else OrderStage.TRANSPORT,
            )
            self.sim.orders[order.order_id] = order
            if to_arm_queue:
                self.sim.queue.append(order.order_id)
            else:
                self.sim.transport_queue.append(order.order_id)

        for i, arm in enumerate(self.sim.arms):
            if i < arms_busy:
                arm.status = ArmStatus.PROCESSING
                arm.current_task = -100 - i
                arm.remaining_time = 1.5
            else:
                arm.status = ArmStatus.IDLE

        for i, robot in enumerate(self.sim.robots):
            if i < robots_busy:
                robot.status = RobotStatus.MOVING
                robot.current_task = -200 - i
                robot.remaining_time = 1.5
                robot.battery = 80.0
            else:
                robot.status = RobotStatus.IDLE
                robot.battery = 80.0

        for i, drone in enumerate(self.sim.drones):
            if i < drones_busy:
                drone.status = DroneStatus.SCANNING
                drone.current_task = "scan:storage"
                drone.remaining_time = 1.5
                drone.battery = 80.0
            else:
                drone.status = DroneStatus.AVAILABLE
                drone.battery = 80.0
