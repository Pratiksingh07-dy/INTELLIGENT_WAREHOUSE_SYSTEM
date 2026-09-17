from __future__ import annotations

import random
from typing import Dict, List, Optional

# Import all warehouse resources and supporting classes
from simulation.drone import Drone
from simulation.ground_robot import GroundRobot, RobotStatus
from simulation.inventory import Inventory
from simulation.orders import Order, OrderGenerator, OrderPriority, OrderStage
from simulation.robotic_arm import ArmStatus, RoboticArm

# Default number of resources in the warehouse
NUM_ARMS = 2
NUM_ROBOTS = 2
NUM_DRONES = 2

# RL action values used by the warehouse
ACTION_ASSIGN_ARM = 0
ACTION_ASSIGN_ROBOT = 1
ACTION_ASSIGN_DRONE = 2
ACTION_PRIORITIZE = 3
ACTION_NOOP = 4
NUM_ACTIONS = 5

# Action names for display and tracking
ACTION_NAMES = {
    ACTION_ASSIGN_ARM: "ASSIGN_ARM",
    ACTION_ASSIGN_ROBOT: "ASSIGN_ROBOT",
    ACTION_ASSIGN_DRONE: "ASSIGN_DRONE",
    ACTION_PRIORITIZE: "PRIORITIZE",
    ACTION_NOOP: "NOOP",
}


class Warehouse:
    # Initialize the complete warehouse simulation
    def __init__(self, num_arms: int = NUM_ARMS, num_robots: int = NUM_ROBOTS,
                 num_drones: int = NUM_DRONES, arrival_rate: float = 0.45,
                 seed: Optional[int] = None):

        # Store the number of available resources
        self.num_arms = num_arms
        self.num_robots = num_robots
        self.num_drones = num_drones

        # Random generator for simulation behaviour
        self._seed = seed
        self._rng = random.Random(seed)

        # Create robotic arms, ground robots and drones
        self.arms: List[RoboticArm] = [RoboticArm(arm_id=i + 1) for i in range(num_arms)]
        self.robots: List[GroundRobot] = [GroundRobot(robot_id=i + 1) for i in range(num_robots)]
        self.drones: List[Drone] = [Drone(drone_id=i + 1) for i in range(num_drones)]

        # Create inventory and order generator
        self.inventory = Inventory(seed=seed)
        self.order_generator = OrderGenerator(arrival_rate=arrival_rate, seed=seed)

        # Simulation clock and step size
        self.time: float = 0.0
        self.dt: float = 1.0

        # Store all orders and queues
        self.orders: Dict[int, Order] = {}
        self.queue: List[int] = []               # order_ids waiting for an arm
        self.transport_queue: List[int] = []      # order_ids waiting for a ground robot
        self.completed_orders: List[int] = []

        # Metrics used to track warehouse performance
        self.total_orders_generated = 0
        self.total_orders_completed = 0
        self.last_reward = 0.0
        self.cumulative_reward = 0.0
        self.last_action_name = "NOOP"
        self.last_action_valid = True
        self.energy_this_step = 0.0

    # Reset the warehouse back to its initial state
    def reset(self, seed: Optional[int] = None) -> None:
        seed = seed if seed is not None else self._seed
        self.__init__(self.num_arms, self.num_robots, self.num_drones,
                       self.order_generator.arrival_rate, seed=seed)

    # Run one complete simulation step using the selected action
    def step(self, action: int) -> float:
        """Apply one RL action, advance all resources by dt, return the reward."""
        self.last_action_name = ACTION_NAMES.get(action, "NOOP")

        # Generate new orders for the current time
        new_orders = self.order_generator.generate_batch(self.time)
        for order in new_orders:
            self.orders[order.order_id] = order
            self.queue.append(order.order_id)
        self.total_orders_generated += len(new_orders)

        # Apply the action selected by the RL agent
        assignments_made = 0
        action_valid = True
        if action == ACTION_ASSIGN_ARM:
            action_valid = self._assign_arm()
            assignments_made += int(action_valid)
        elif action == ACTION_ASSIGN_ROBOT:
            action_valid = self._assign_robot()
            assignments_made += int(action_valid)
        elif action == ACTION_ASSIGN_DRONE:
            action_valid = self._assign_drone()
            assignments_made += int(action_valid)
        elif action == ACTION_PRIORITIZE:
            action_valid = self._prioritize_queue()
        elif action == ACTION_NOOP:
            action_valid = True
        self.last_action_valid = action_valid

        # Move all resources forward by one time step
        completed_arm_orders = [oid for arm in self.arms if (oid := arm.step(self.dt)) is not None]
        completed_robot_orders = [oid for robot in self.robots if (oid := robot.step(self.dt)) is not None]
        for drone in self.drones:
            if drone.step(self.dt):
                self.inventory.mark_scanned(self._rng.choice(list(self.inventory.stock.keys())), self.time)

        orders_completed_this_step = 0

        # Move picked orders to the transport queue
        for oid in completed_arm_orders:
            order = self.orders.get(oid)
            if order is None:
                continue
            order.stage = OrderStage.TRANSPORT
            self.transport_queue.append(oid)

        # Mark transported orders as completed
        for oid in completed_robot_orders:
            order = self.orders.get(oid)
            if order is None:
                continue
            order.stage = OrderStage.COMPLETED
            order.completion_time = self.time
            self.completed_orders.append(oid)
            self.total_orders_completed += 1
            orders_completed_this_step += 1

        # Calculate reward for the current step
        reward = self._compute_reward(assignments_made, action_valid, orders_completed_this_step)

        self.last_reward = reward
        self.cumulative_reward += reward
        self.time += self.dt
        return reward

    # Assign the next waiting order to an available robotic arm
    def _assign_arm(self) -> bool:
        if not self.queue:
            return False
        idle_arm = next((a for a in self.arms if a.is_available()), None)
        if idle_arm is None:
            return False
        order_id = self.queue.pop(0)
        order = self.orders[order_id]
        order.stage = OrderStage.PICKING
        order.assigned_resource = f"Arm-{idle_arm.arm_id}"
        idle_arm.assign_task(order_id, order.processing_time)
        return True

    # Assign a picked order to an available ground robot
    def _assign_robot(self) -> bool:
        if not self.transport_queue:
            return False
        idle_robot = next((r for r in self.robots if r.is_available()), None)
        if idle_robot is None:
            return False
        order_id = self.transport_queue.pop(0)
        order = self.orders[order_id]
        order.assigned_resource = f"AGV-{idle_robot.robot_id}"
        travel_time = self._rng.uniform(1.5, 3.5)
        idle_robot.assign_transport(order_id, destination="dispatch", travel_time=travel_time)
        return True

    # Send an available drone to scan a warehouse zone
    def _assign_drone(self) -> bool:
        idle_drone = next((d for d in self.drones if d.is_available()), None)
        if idle_drone is None:
            return False
        zone = self._rng.choice(["storage", "packing", "dispatch"])
        duration = self._rng.uniform(1.0, 2.5)
        idle_drone.assign_scan(zone, duration)
        return True

    # Sort the queue based on order priority
    def _prioritize_queue(self) -> bool:
        if not self.queue:
            return False
        priority_rank = {OrderPriority.HIGH: 0, OrderPriority.NORMAL: 1, OrderPriority.LOW: 2}
        self.queue.sort(key=lambda oid: priority_rank.get(self.orders[oid].priority, 1))
        return True

    # Calculate reward using warehouse performance
    def _compute_reward(self, assignments_made: int, action_valid: bool,
                        orders_completed_this_step: int) -> float:
        queue_len = len(self.queue) + len(self.transport_queue)
        avg_wait = self.average_waiting_time()

        # Count idle resources when orders are waiting
        idle_resources = sum(1 for a in self.arms if a.is_available()) \
            + sum(1 for r in self.robots if r.is_available())
        idle_penalty_count = idle_resources if queue_len > 0 else 0

        self.energy_this_step = (
            sum(a.energy_used for a in self.arms) * 0.0  # cumulative, not per-step
        )

        # Calculate energy consumed during this step
        step_energy = self._instantaneous_energy()

        reward = 0.0
        reward += 10.0 * orders_completed_this_step
        reward += 2.0 * assignments_made
        if not action_valid:
            reward -= 3.0
        reward -= 0.4 * queue_len
        reward -= 0.05 * avg_wait
        reward -= 0.3 * idle_penalty_count
        reward -= 0.02 * step_energy

        return round(reward, 4)

    # Calculate current energy usage of active and idle resources
    def _instantaneous_energy(self) -> float:
        active = sum(1 for a in self.arms if a.status == ArmStatus.PROCESSING)
        active += sum(1 for r in self.robots if r.status in (RobotStatus.MOVING, RobotStatus.LOADING))
        active += sum(1 for d in self.drones if d.status.value in ("scanning", "monitoring"))
        idle = (self.num_arms + self.num_robots + self.num_drones) - active
        return active * 1.0 + idle * 0.1

    # Calculate average waiting time of pending orders
    def average_waiting_time(self) -> float:
        pending_ids = self.queue + self.transport_queue
        if not pending_ids:
            return 0.0
        return sum(self.orders[oid].waiting_time(self.time) for oid in pending_ids) / len(pending_ids)

    # Check whether any high priority order is waiting
    def has_high_priority_waiting(self) -> bool:
        pending_ids = self.queue + self.transport_queue
        return any(self.orders[oid].priority == OrderPriority.HIGH for oid in pending_ids)

    # Calculate completed orders per unit time
    def throughput(self) -> float:
        if self.time <= 0:
            return 0.0
        return self.total_orders_completed / self.time

    # Calculate average utilization of all resources
    def resource_utilization(self) -> float:
        utils = [a.utilization for a in self.arms] + [r.utilization for r in self.robots] \
            + [d.utilization for d in self.drones]
        return round(sum(utils) / len(utils), 3) if utils else 0.0

    # Calculate total energy used by all resources
    def total_energy_used(self) -> float:
        return round(sum(a.energy_used for a in self.arms)
                     + sum(r.energy_used for r in self.robots)
                     + sum(d.energy_used for d in self.drones), 2)

    # Convert the warehouse state into dictionary format
    def to_dict(self) -> dict:
        pending_orders = [self.orders[oid].to_dict(self.time) for oid in self.queue]
        transport_orders = [self.orders[oid].to_dict(self.time) for oid in self.transport_queue]
        recent_completed = [self.orders[oid].to_dict(self.time) for oid in self.completed_orders[-10:]]
        return {
            "time": round(self.time, 2),
            "arms": [a.to_dict() for a in self.arms],
            "robots": [r.to_dict() for r in self.robots],
            "drones": [d.to_dict() for d in self.drones],
            "inventory": self.inventory.to_dict(),
            "queue": pending_orders,
            "transport_queue": transport_orders,
            "recent_completed": recent_completed,
            "metrics": {
                "total_orders_generated": self.total_orders_generated,
                "total_orders_completed": self.total_orders_completed,
                "pending_orders": len(self.queue) + len(self.transport_queue),
                "average_waiting_time": round(self.average_waiting_time(), 2),
                "throughput": round(self.throughput(), 3),
                "resource_utilization": self.resource_utilization(),
                "energy_used": self.total_energy_used(),
                "last_reward": self.last_reward,
                "cumulative_reward": round(self.cumulative_reward, 2),
                "last_action": self.last_action_name,
                "last_action_valid": self.last_action_valid,
            },
        }