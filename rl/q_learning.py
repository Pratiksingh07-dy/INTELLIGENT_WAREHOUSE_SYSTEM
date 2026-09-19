"""
q_learning.py
-------------
Q-Learning: OFF-POLICY Temporal-Difference control (Watkins, 1989).

The agent behaves epsilon-greedily (to explore), but the UPDATE always
bootstraps off the GREEDY (max) action-value at the next state,
regardless of which action is actually taken next. This decouples the
behaviour policy from the policy being learned/evaluated (hence
"off-policy"), and Q-Learning directly approximates the OPTIMAL
action-value function Q*.

Update rule:
    delta_t = r_{t+1} + gamma * max_a' Q(s_{t+1}, a') - Q(s_t, a_t)
    Q(s_t, a_t) <- Q(s_t, a_t) + alpha * delta_t
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from rl.utils import epsilon_greedy_action


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
    n_states, n_actions = env.n_states, env.n_actions
    Q = np.zeros((n_states, n_actions))
    rng = np.random.default_rng(seed)
    epsilon = epsilon_start
    episode_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0

        while not done:
            action = epsilon_greedy_action(Q[state], epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next = 0.0 if terminated else np.max(Q[next_state])
            td_target = reward + gamma * best_next
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    policy = np.zeros((n_states, n_actions))
    policy[np.arange(n_states), np.argmax(Q, axis=1)] = 1.0
    return Q, policy, episode_rewards
