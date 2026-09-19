"""
utils.py
--------
Shared RL utilities:

* epsilon_greedy_action  - the classic exploration/exploitation rule
* estimate_mdp_model     - builds an empirical (P, R) model of the
  WarehouseEnv by resetting it into every discretized state and sampling
  real transitions, so that the Dynamic Programming methods (which need
  an explicit model of the environment) can operate on real simulator
  dynamics rather than a hand-authored table.
* save_table / load_table - tiny numpy (de)serialization helpers.
* moving_average         - smoothing for reward-per-episode plots.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def epsilon_greedy_action(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """
    epsilon-greedy exploration:
        with probability epsilon       -> pick a uniformly random action (explore)
        with probability (1 - epsilon) -> pick argmax_a Q(s, a)          (exploit)
    """
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_values)))
    max_q = np.max(q_values)
    best_actions = np.flatnonzero(q_values == max_q)
    return int(rng.choice(best_actions))


def estimate_mdp_model(env, n_samples_per_pair: int = 15,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an empirical model of the environment for Dynamic Programming.

    Dynamic Programming (Policy Iteration / Value Iteration) requires full
    knowledge of the transition dynamics P(s'|s,a) and expected reward
    R(s,a). Our simulator is stochastic (Poisson arrivals, random service
    times) and does not expose a closed-form transition matrix, so we
    ESTIMATE one empirically: for every discretized state s and every
    action a, we reset the simulator into state s (see
    WarehouseEnv.set_discrete_state) and take the action `n_samples_per_pair`
    times, recording the resulting next state and reward. Averaging over
    these samples gives an unbiased Monte-Carlo estimate of P(s'|s,a) and
    R(s,a), which is exactly what iterative policy evaluation / policy
    iteration / value iteration then operate on.

    Returns
    -------
    P : np.ndarray of shape (n_states, n_actions, n_states)  transition probabilities
    R : np.ndarray of shape (n_states, n_actions)             expected immediate reward
    """
    n_states = env.n_states
    n_actions = env.n_actions
    rng = np.random.default_rng(seed)

    P = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    R = np.zeros((n_states, n_actions), dtype=np.float64)

    for s in range(n_states):
        for a in range(n_actions):
            reward_accum = 0.0
            for _ in range(n_samples_per_pair):
                env.set_discrete_state(s)
                next_state, reward, _, _, _ = env.step(a)
                P[s, a, next_state] += 1.0
                reward_accum += reward
            total = P[s, a, :].sum()
            if total > 0:
                P[s, a, :] /= total
            else:
                # No samples collected (shouldn't normally happen) -> self-loop
                P[s, a, s] = 1.0
            R[s, a] = reward_accum / n_samples_per_pair

    return P, R


def save_table(path: str, array: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def load_table(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def moving_average(values, window: int = 10) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        window = max(1, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")
