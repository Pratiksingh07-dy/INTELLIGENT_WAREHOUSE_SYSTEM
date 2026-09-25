from __future__ import annotations

# Import Tuple for type hints.
from typing import Tuple

# Import NumPy for creating and working with arrays.
import numpy as np

# Import the epsilon-greedy action selection function.
from rl.utils import epsilon_greedy_action


# Define the Q-learning algorithm.
def q_learning(env, num_episodes: int = 3000, alpha: float = 0.1, gamma: float = 0.95,
               epsilon_start: float = 1.0, epsilon_min: float = 0.05,
               epsilon_decay: float = 0.998, seed: int = 0
               ) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Returns
    -------
    Q               : (n_states, n_actions) learned action-value function (approx Q*)
    policy          : (n_states, n_actions) greedy policy derived from Q (one-hot)
    episode_rewards : list of total reward per training episode
    """

    # Get the number of states and actions from the environment.
    n_states, n_actions = env.n_states, env.n_actions

    # Create the Q-table and initialize all values to zero.
    Q = np.zeros((n_states, n_actions))

    # Create a random number generator using the given seed.
    rng = np.random.default_rng(seed)

    # Start epsilon at its initial value.
    epsilon = epsilon_start

    # Create a list to store the total reward from each episode.
    episode_rewards = []

    # Run the learning process for the specified number of episodes.
    for ep in range(num_episodes):

        # Reset the environment at the beginning of each episode.
        state, _ = env.reset(seed=seed + ep)

        # Mark the episode as not finished.
        done = False

        # Start the total reward for this episode at zero.
        total_reward = 0.0

        # Continue until the episode is finished.
        while not done:

            # Select an action using the epsilon-greedy strategy.
            action = epsilon_greedy_action(Q[state], epsilon, rng)

            # Perform the selected action in the environment.
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Mark the episode as finished if it is terminated or truncated.
            done = terminated or truncated

            # Get the best future Q-value for the next state.
            best_next = 0.0 if terminated else np.max(Q[next_state])

            # Calculate the target Q-value.
            td_target = reward + gamma * best_next

            # Calculate the temporal-difference error.
            td_error = td_target - Q[state, action]

            # Update the Q-value for the selected state and action.
            Q[state, action] += alpha * td_error

            # Add the current reward to the episode's total reward.
            total_reward += reward

            # Move to the next state.
            state = next_state

        # Store the total reward earned in this episode.
        episode_rewards.append(total_reward)

        # Reduce epsilon gradually while keeping it above the minimum.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Create an empty policy table.
    policy = np.zeros((n_states, n_actions))

    # Mark the action with the highest Q-value as the chosen action for each state.
    policy[np.arange(n_states), np.argmax(Q, axis=1)] = 1.0

    # Return the learned Q-table, policy, and episode rewards.
    return Q, policy, episode_rewards
