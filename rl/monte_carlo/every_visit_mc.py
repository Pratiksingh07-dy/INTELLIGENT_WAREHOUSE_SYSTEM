from __future__ import annotations

from typing import Tuple

import numpy as np

# epsilon-greedy action selection
from rl.utils import epsilon_greedy_action


# every-visit Monte Carlo control
def every_visit_mc_control(env, num_episodes: int = 2000, gamma: float = 0.95,
                            epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                            epsilon_decay: float = 0.999, seed: int = 0
                            ) -> Tuple[np.ndarray, np.ndarray, list]:

    # get number of states and actions
    n_states, n_actions = env.n_states, env.n_actions
    Q = np.zeros((n_states, n_actions))
    N = np.zeros((n_states, n_actions), dtype=np.int64)

    # initialize random generator and epsilon
    rng = np.random.default_rng(seed)
    epsilon = epsilon_start
    episode_rewards = []

    # run training episodes
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        episode = []
        done = False
        total_reward = 0.0

        # generate one episode
        while not done:
            action = epsilon_greedy_action(Q[state], epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            total_reward += reward
            state = next_state
            done = terminated or truncated

        episode_rewards.append(total_reward)

        # calculate returns for each time step
        G = 0.0
        returns_from_t = [0.0] * len(episode)
        for t in reversed(range(len(episode))):
            _, _, r_t1 = episode[t]
            G = r_t1 + gamma * G
            returns_from_t[t] = G

        # update Q values for every visit
        for t, (s_t, a_t, _) in enumerate(episode):
            G_t = returns_from_t[t]
            N[s_t, a_t] += 1
            Q[s_t, a_t] += (G_t - Q[s_t, a_t]) / N[s_t, a_t]

        # reduce epsilon over episodes
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # create greedy policy from Q values
    policy = np.zeros((n_states, n_actions))
    policy[np.arange(n_states), np.argmax(Q, axis=1)] = 1.0

    return Q, policy, episode_rewards