"""
sarsa.py
--------
SARSA (State-Action-Reward-State-Action): ON-POLICY Temporal-Difference
control. The name comes from the quintuple (s, a, r, s', a') used in the
update: the NEXT action a' is the one actually chosen by the current
(epsilon-greedy) policy, so SARSA learns the value of the policy it is
actually following - including the effects of its own exploration.

Update rule:
    delta_t = r_{t+1} + gamma * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
    Q(s_t, a_t) <- Q(s_t, a_t) + alpha * delta_t

where a_{t+1} is sampled epsilon-greedily from Q(s_{t+1}, .) - i.e. the
SAME policy used to act is used in the update target (on-policy).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from rl.utils import epsilon_greedy_action


def sarsa(env, num_episodes: int = 3000, alpha: float = 0.1, gamma: float = 0.95,
          epsilon_start: float = 1.0, epsilon_min: float = 0.05,
          epsilon_decay: float = 0.998, seed: int = 0
          ) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Returns
    -------
    Q               : (n_states, n_actions) learned action-value function
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
        action = epsilon_greedy_action(Q[state], epsilon, rng)
        done = False
        total_reward = 0.0

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                td_target = reward
            else:
                next_action = epsilon_greedy_action(Q[next_state], epsilon, rng)
                td_target = reward + gamma * Q[next_state, next_action]

            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            total_reward += reward
            state = next_state
            if not done:
                action = next_action

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    policy = np.zeros((n_states, n_actions))
    policy[np.arange(n_states), np.argmax(Q, axis=1)] = 1.0
    return Q, policy, episode_rewards
