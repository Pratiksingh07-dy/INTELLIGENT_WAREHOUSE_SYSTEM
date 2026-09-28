from __future__ import annotations

from typing import Tuple

import numpy as np

from rl.utils import epsilon_greedy_action


def td_lambda_prediction(env, policy_q: np.ndarray, num_episodes: int = 1000,
                          alpha: float = 0.1, gamma: float = 0.95, lam: float = 0.8,
                          epsilon: float = 0.1, seed: int = 0
                          ) -> Tuple[np.ndarray, list]:
    # Get the total number of states from the environment.
    n_states = env.n_states

    # Initialize the value function V(s) for all states to zero.
    V = np.zeros(n_states)

    # Create a random number generator using the provided seed
    # so that action selection is reproducible.
    rng = np.random.default_rng(seed)

    # Store the total reward obtained in each episode.
    episode_rewards = []

    # Run the TD(lambda) prediction algorithm for the specified number of episodes.
    for ep in range(num_episodes):
        # Reset the environment at the beginning of each episode.
        # A different seed is used for each episode.
        state, _ = env.reset(seed=seed + ep)

        # Initialize eligibility traces for all states to zero.
        eligibility = np.zeros(n_states)

        # Track whether the current episode has ended.
        done = False

        # Keep track of the total reward collected in this episode.
        total_reward = 0.0

        # Continue interacting with the environment until the episode ends.
        while not done:
            # Select an action using an epsilon-greedy policy based
            # on the Q-values for the current state.
            action = epsilon_greedy_action(policy_q[state], epsilon, rng)

            # Take the selected action and observe the next state,
            # reward, and termination/truncation information.
            next_state, reward, terminated, truncated, _ = env.step(action)

            # The episode is finished if it is either naturally terminated
            # or truncated by the environment.
            done = terminated or truncated

            # Calculate the TD target.
            # If the episode terminates, there is no future value to include.
            td_target = reward + gamma * V[next_state] * (0.0 if terminated else 1.0)

            # Calculate the TD error (difference between the TD target
            # and the current estimate of the state's value).
            td_error = td_target - V[state]

            # Decay the eligibility traces according to gamma and lambda.
            eligibility *= gamma * lam

            # Increase the eligibility of the current state.
            eligibility[state] += 1.0

            # Update the value function using the TD error
            # weighted by the eligibility traces.
            V += alpha * td_error * eligibility

            # Add the current reward to the episode's total reward.
            total_reward += reward

            # Move to the next state.
            state = next_state

        # Store the total reward collected during this episode.
        episode_rewards.append(total_reward)

    # Return the learned value function and the rewards from all episodes.
    return V, episode_rewards
