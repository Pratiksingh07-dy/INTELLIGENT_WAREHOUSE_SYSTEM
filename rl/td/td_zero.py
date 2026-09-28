"""
td_zero.py
----------
TD(0): the simplest Temporal-Difference prediction method.

Unlike Monte Carlo (which waits for the full episode return G_t), TD(0)
updates its estimate of V(s) after every single step, using the
observed reward plus its OWN current estimate of the next state's value
as a "bootstrap" target:

    TD target : r_{t+1} + gamma * V(s_{t+1})
    TD error  : delta_t = r_{t+1} + gamma * V(s_{t+1}) - V(s_t)
    Update    : V(s_t) <- V(s_t) + alpha * delta_t

This file evaluates a FIXED policy (policy prediction, not control) -
this is the classical formulation of TD(0) and lets us clearly separate
the "prediction" problem (evaluate a given policy) from the "control"
problem (find a good policy), which SARSA and Q-Learning solve.

In this project we evaluate the epsilon-greedy behaviour policy induced
by a Q-table (e.g. the one learned by Q-Learning) - this doubles as a
sanity check that Q-Learning's greedy policy has a sensible value
function under TD(0) prediction.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from rl.utils import epsilon_greedy_action


def td_zero_prediction(env, policy_q: np.ndarray, num_episodes: int = 1000,
                        alpha: float = 0.1, gamma: float = 0.95,
                        epsilon: float = 0.1, seed: int = 0
                        ) -> Tuple[np.ndarray, list]:
    """
    Parameters
    ----------
    policy_q : (n_states, n_actions) Q-values defining the (epsilon-greedy)
               policy to evaluate. Pass a Q-table learned elsewhere
               (e.g. from Q-Learning) to evaluate that policy's state values.

    Returns
    -------
    V               : (n_states,) estimated state-value function
    episode_rewards : list of total reward per training/evaluation episode
    """
    n_states = env.n_states
    V = np.zeros(n_states)
    rng = np.random.default_rng(seed)
    episode_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0

        while not done:
            action = epsilon_greedy_action(policy_q[state], epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TD(0) bootstrap update
            td_target = reward + gamma * V[next_state] * (0.0 if terminated else 1.0)
            td_error = td_target - V[state]
            V[state] += alpha * td_error

            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

    return V, episode_rewards
