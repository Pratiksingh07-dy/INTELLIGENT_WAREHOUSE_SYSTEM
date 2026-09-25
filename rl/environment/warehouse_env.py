from __future__ import annotations

# Import Optional and Tuple for type hints.
from typing import Optional, Tuple

# Import Gymnasium to create the reinforcement learning environment.
import gymnasium as gym

# Import NumPy for numerical calculations and state handling.
import numpy as np

# Import spaces to define the observation and action spaces.
from gymnasium import spaces

# Import order-related classes used by the warehouse simulation.
from simulation.orders import Order, OrderPriority, OrderStage

# Import the robotic arm status values.
from simulation.robotic_arm import ArmStatus

# Import the ground robot status values.
from simulation.ground_robot import RobotStatus

# Import the drone status values.
from simulation.drone import DroneStatus

# Import the number of available actions and the Warehouse simulator.
from simulation.warehouse import NUM_ACTIONS, Warehouse

# --- Discretization bucket sizes -------------------------------------------------
# Number of groups used to represent pending orders.
PENDING_BUCKETS = 4     # 0, 1, 2, 3+

# Number of groups used to represent waiting time.
WAIT_BUCKETS = 3        # low, medium, high

# Number of groups used to represent high-priority orders.
PRIORITY_BUCKETS = 2    # no high-priority waiting / at least one waiting

# Waiting time below this value is considered low.
WAIT_LOW_THRESHOLD = 2.0

# Waiting time below this value is considered medium.
WAIT_HIGH_THRESHOLD = 5.0


# Create the warehouse environment for reinforcement learning.
class WarehouseEnv(gym.Env):
    """A high-level, tabular-friendly warehouse resource-allocation environment."""

    # Define the rendering modes supported by the environment.
    metadata = {"render_modes": ["human"]}

    # Initialize the warehouse environment.
    def __init__(self, num_arms: int = 2, num_robots: int = 2, num_drones: int = 2,
                 arrival_rate: float = 0.45, max_steps: int = 200,
                 seed: Optional[int] = None):

        # Initialize the parent Gym environment.
        super().__init__()

        # Store the number of robotic arms.
        self.num_arms = num_arms

        # Store the number of ground robots.
        self.num_robots = num_robots

        # Store the number of drones.
        self.num_drones = num_drones

        # Store the maximum number of simulation steps.
        self.max_steps = max_steps

        # Store the random seed.
        self._seed = seed

        # Start the step counter at zero.
        self._step_count = 0

        # Create the warehouse simulation.
        self.sim = Warehouse(num_arms=num_arms, num_robots=num_robots,
                              num_drones=num_drones, arrival_rate=arrival_rate, seed=seed)

        # Define the size of each part of the discrete state.
        self._dims = (PENDING_BUCKETS, num_arms + 1, num_robots + 1,
                      num_drones + 1, WAIT_BUCKETS, PRIORITY_BUCKETS)

        # Calculate the total number of possible states.
        self.n_states = int(np.prod(self._dims))

        # Store the total number of available actions.
        self.n_actions = NUM_ACTIONS

        # Define the observation space using discrete state numbers.
        self.observation_space = spaces.Discrete(self.n_states)

        # Define the action space using discrete action numbers.
        self.action_space = spaces.Discrete(self.n_actions)

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    # Reset the environment and return the initial state.
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[int, dict]:

        # Reset the parent Gym environment.
        super().reset(seed=seed)

        # Reset the warehouse simulation.
        self.sim.reset(seed=seed if seed is not None else self._seed)

        # Reset the environment step counter.
        self._step_count = 0

        # Create the initial encoded state.
        state = self.encode_state()

        # Return the state and the raw simulation information.
        return state, {"raw": self.sim.to_dict()}

    # Perform one action in the environment.
    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:

        # Apply the selected action to the warehouse simulation.
        reward = self.sim.step(int(action))

        # Increase the step counter.
        self._step_count += 1

        # Calculate the next encoded state.
        next_state = self.encode_state()

        # The task is continuing and ends by truncation.
        terminated = False  # this task is continuing/episodic-by-truncation

        # Check whether the maximum number of steps has been reached.
        truncated = self._step_count >= self.max_steps

        # Store the raw simulation information.
        info = {"raw": self.sim.to_dict()}

        # Return the next state, reward, termination information, and extra data.
        return next_state, reward, terminated, truncated, info

    # Display the current simulation information.
    def render(self):  # pragma: no cover - visualization happens in the web UI

        # Get the current warehouse information.
        d = self.sim.to_dict()

        # Print the current time, pending orders, and latest reward.
        print(f"t={d['time']} pending={d['metrics']['pending_orders']} "
              f"reward={d['metrics']['last_reward']}")

    # ------------------------------------------------------------------ #
    # State encoding / decoding (the S in the MDP)
    # ------------------------------------------------------------------ #

    # Convert the number of pending orders into a bucket.
    def _bucket_pending(self, n: int) -> int:

        # Keep the value within the allowed pending-order buckets.
        return min(n, PENDING_BUCKETS - 1)

    # Convert average waiting time into a bucket.
    def _bucket_wait(self, avg_wait: float) -> int:

        # Return the low waiting-time bucket.
        if avg_wait < WAIT_LOW_THRESHOLD:
            return 0

        # Return the medium waiting-time bucket.
        if avg_wait < WAIT_HIGH_THRESHOLD:
            return 1

        # Return the high waiting-time bucket.
        return 2

    # Convert the current warehouse situation into one discrete state number.
    def encode_state(self) -> int:

        # Store the current warehouse simulation.
        sim = self.sim

        # Calculate the bucket for pending orders.
        pending = self._bucket_pending(len(sim.queue) + len(sim.transport_queue))

        # Count robotic arms that are currently processing.
        arms_busy = sum(1 for a in sim.arms if a.status == ArmStatus.PROCESSING)

        # Count ground robots that are moving or loading.
        robots_busy = sum(1 for r in sim.robots if r.status in
                           (RobotStatus.MOVING, RobotStatus.LOADING))

        # Count drones that are scanning or monitoring.
        drones_busy = sum(1 for d in sim.drones if d.status in
                           (DroneStatus.SCANNING, DroneStatus.MONITORING))

        # Convert the average waiting time into a bucket.
        wait_bucket = self._bucket_wait(sim.average_waiting_time())

        # Check whether a high-priority order is waiting.
        priority_flag = int(sim.has_high_priority_waiting())

        # Store all state values together.
        indices = (pending, arms_busy, robots_busy, drones_busy, wait_bucket, priority_flag)

        # Convert the multi-part state into one integer state number.
        return int(np.ravel_multi_index(indices, self._dims))

    # Convert a single state number back into its individual state values.
    def decode_state(self, state_index: int) -> Tuple[int, int, int, int, int, int]:

        # Return each state component as an integer.
        return tuple(int(x) for x in np.unravel_index(state_index, self._dims))

    # Create a readable description of a discrete state.
    def state_description(self, state_index: int) -> dict:

        # Decode the state into its individual values.
        pending, arms_busy, robots_busy, drones_busy, wait_bucket, priority = \
            self.decode_state(state_index)

        # Return the state information as a dictionary.
        return {
            # Store the pending-order bucket.
            "pending_bucket": pending,

            # Store the number of busy robotic arms.
            "arms_busy": arms_busy,

            # Store the number of busy ground robots.
            "robots_busy": robots_busy,

            # Store the number of busy drones.
            "drones_busy": drones_busy,

            # Convert the waiting bucket number into a readable name.
            "wait_bucket": ["low", "medium", "high"][wait_bucket],

            # Store whether a high-priority order is waiting.
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

    # Set the simulator to a synthetic configuration matching a discrete state.
    def set_discrete_state(self, state_index: int) -> None:

        # Decode the requested state into its individual values.
        pending, arms_busy, robots_busy, drones_busy, wait_bucket, priority_flag = \
            self.decode_state(state_index)

        # Reset the warehouse simulation.
        self.sim.reset(seed=None)

        # Representative "pending" count for this bucket
        pending_count = {0: 0, 1: 1, 2: 2, 3: 4}[pending]

        # Choose a representative waiting time for the selected bucket.
        wait_repr = {0: 0.5, 1: 3.5, 2: 7.5}[wait_bucket]

        # Set the simulation time to provide enough time for synthetic orders.
        self.sim.time = 10.0  # give some headroom so arrival_time - wait >= 0

        # Split the synthetic pending orders between the "waiting for an arm"
        # queue and the "waiting for a ground robot" transport queue, so that
        # BOTH ASSIGN_ARM and ASSIGN_ROBOT get meaningful, non-degenerate
        # samples during MDP model estimation (estimate_mdp_model). Without
        # this split, the transport queue would always be empty in the
        # synthetic states and Policy/Value Iteration would never learn that
        # ASSIGN_ROBOT is ever useful.

        # Calculate how many orders should wait for an arm.
        num_to_arm_queue = (pending_count + 1) // 2

        # Create the required number of synthetic pending orders.
        for i in range(pending_count):

            # Give the first order high priority when required.
            priority = OrderPriority.HIGH if (priority_flag and i == 0) else OrderPriority.NORMAL

            # Check whether this order should go into the arm queue.
            to_arm_queue = i < num_to_arm_queue

            # Create a synthetic order.
            order = Order(
                order_id=-(i + 1),  # negative synthetic ids, won't collide with real orders
                product="Widget-A",
                quantity=1,
                priority=priority,
                arrival_time=self.sim.time - wait_repr,
                processing_time=2.0,
                stage=OrderStage.QUEUED if to_arm_queue else OrderStage.TRANSPORT,
            )

            # Add the synthetic order to the warehouse order collection.
            self.sim.orders[order.order_id] = order

            # Add the order to the appropriate queue.
            if to_arm_queue:
                self.sim.queue.append(order.order_id)
            else:
                self.sim.transport_queue.append(order.order_id)

        # Set the status of each robotic arm.
        for i, arm in enumerate(self.sim.arms):

            # Make the required number of arms busy.
            if i < arms_busy:
                arm.status = ArmStatus.PROCESSING
                arm.current_task = -100 - i
                arm.remaining_time = 1.5

            # Keep the remaining arms idle.
            else:
                arm.status = ArmStatus.IDLE

        # Set the status of each ground robot.
        for i, robot in enumerate(self.sim.robots):

            # Make the required number of robots busy.
            if i < robots_busy:
                robot.status = RobotStatus.MOVING
                robot.current_task = -200 - i
                robot.remaining_time = 1.5
                robot.battery = 80.0

            # Keep the remaining robots idle.
            else:
                robot.status = RobotStatus.IDLE
                robot.battery = 80.0

        # Set the status of each drone.
        for i, drone in enumerate(self.sim.drones):

            # Make the required number of drones busy.
            if i < drones_busy:
                drone.status = DroneStatus.SCANNING
                drone.current_task = "scan:storage"
                drone.remaining_time = 1.5
                drone.battery = 80.0

            # Keep the remaining drones available.
            else:
                drone.status = DroneStatus.AVAILABLE
                drone.battery = 80.0
