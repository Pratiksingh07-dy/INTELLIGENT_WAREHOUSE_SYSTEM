from __future__ import annotations

# Import the os module for working with folders and file paths.
import os

# Import Tuple for type hints.
from typing import Tuple

# Import NumPy for numerical calculations and arrays.
import numpy as np


# Select an action using the epsilon-greedy strategy.
def epsilon_greedy_action(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """
    epsilon-greedy exploration:
        with probability epsilon       -> pick a uniformly random action (explore)
        with probability (1 - epsilon) -> pick argmax_a Q(s, a)          (exploit)
    """

    # Generate a random number and check whether to explore.
    if rng.random() < epsilon:

        # Select a random action.
        return int(rng.integers(0, len(q_values)))

    # Find the highest Q-value.
    max_q = np.max(q_values)

    # Find all actions that have the highest Q-value.
    best_actions = np.flatnonzero(q_values == max_q)

    # Randomly choose one of the best actions.
    return int(rng.choice(best_actions))


# Estimate the transition model and reward model of the environment.
def estimate_mdp_model(env, n_samples_per_pair: int = 15,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:

    # Get the total number of states in the environment.
    n_states = env.n_states

    # Get the total number of actions in the environment.
    n_actions = env.n_actions

    # Create a random number generator using the given seed.
    rng = np.random.default_rng(seed)

    # Create an empty transition probability table.
    P = np.zeros((n_states, n_actions, n_states), dtype=np.float64)

    # Create an empty reward table.
    R = np.zeros((n_states, n_actions), dtype=np.float64)

    # Go through every possible state.
    for s in range(n_states):

        # Go through every possible action.
        for a in range(n_actions):

            # Start the accumulated reward at zero.
            reward_accum = 0.0

            # Repeat the experiment for the required number of samples.
            for _ in range(n_samples_per_pair):

                # Set the simulator to the selected discrete state.
                env.set_discrete_state(s)

                # Perform the selected action and record the result.
                next_state, reward, _, _, _ = env.step(a)

                # Count how often this next state occurs.
                P[s, a, next_state] += 1.0

                # Add the reward to the accumulated reward.
                reward_accum += reward

            # Calculate the total number of collected samples.
            total = P[s, a, :].sum()

            # Check whether samples were collected.
            if total > 0:

                # Convert the collected counts into probabilities.
                P[s, a, :] /= total

            # Handle the case where no samples were collected.
            else:

                # No samples collected (shouldn't normally happen) -> self-loop
                P[s, a, s] = 1.0

            # Calculate the average reward for this state-action pair.
            R[s, a] = reward_accum / n_samples_per_pair

    # Return the estimated transition and reward tables.
    return P, R


# Save a NumPy array to a file.
def save_table(path: str, array: np.ndarray) -> None:

    # Create the required directory if it does not already exist.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the array to the specified path.
    np.save(path, array)


# Load a NumPy array from a file.
def load_table(path: str) -> np.ndarray:

    # Load and return the saved NumPy array.
    return np.load(path, allow_pickle=False)


# Calculate the moving average of a sequence of values.
def moving_average(values, window: int = 10) -> np.ndarray:

    # Convert the input values into a NumPy array of floating-point numbers.
    values = np.asarray(values, dtype=np.float64)

    # Adjust the window if there are fewer values than the requested window.
    if len(values) < window:
        window = max(1, len(values))

    # Create an averaging kernel for the selected window size.
    kernel = np.ones(window) / window

    # Calculate and return the moving average.
    return np.convolve(values, kernel, mode="valid")
