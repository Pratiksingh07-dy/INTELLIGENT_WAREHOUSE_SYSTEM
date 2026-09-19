from __future__ import annotations

from typing import Tuple

import numpy as np


# evaluate the current policy
def iterative_policy_evaluation(policy: np.ndarray, P: np.ndarray, R: np.ndarray,
                                 gamma: float = 0.95, theta: float = 1e-4,
                                 max_iterations: int = 1000) -> np.ndarray:

    # get number of states and actions
    n_states, n_actions = R.shape
    V = np.zeros(n_states)

    for _ in range(max_iterations):
        delta = 0.0

        # calculate expected returns
        expected_returns = R + gamma * P.dot(V)          # shape (n_states, n_actions)

        # update state values using the policy
        new_V = np.sum(policy * expected_returns, axis=1)  # weighted by pi(a|s)

        # calculate maximum change
        delta = np.max(np.abs(new_V - V))
        V = new_V

        # stop when values have converged
        if delta < theta:
            break

    return V


# find the optimal policy using value iteration
def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float = 0.95,
                     theta: float = 1e-4, max_iterations: int = 1000
                     ) -> Tuple[np.ndarray, np.ndarray]:

    # get number of states and actions
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    history = []

    for it in range(max_iterations):

        # calculate action values
        Q = R + gamma * P.dot(V)          # shape (n_states, n_actions)

        # select the highest value for each state
        new_V = np.max(Q, axis=1)

        # calculate maximum change
        delta = np.max(np.abs(new_V - V))
        V = new_V
        history.append(delta)

        # stop when values have converged
        if delta < theta:
            break

    # calculate final action values
    Q_final = R + gamma * P.dot(V)

    # select best action for each state
    best_actions = np.argmax(Q_final, axis=1)

    # create deterministic policy
    policy = np.zeros((n_states, n_actions))
    policy[np.arange(n_states), best_actions] = 1.0

    return V, policy


# create a policy from the value function
def policy_from_value(V: np.ndarray, P: np.ndarray, R: np.ndarray,
                       gamma: float = 0.95) -> np.ndarray:

    # get number of states and actions
    n_states, n_actions = R.shape

    # calculate action values
    Q = R + gamma * P.dot(V)

    # select best action for each state
    best_actions = np.argmax(Q, axis=1)

    # create deterministic policy
    policy = np.zeros((n_states, n_actions))
    policy[np.arange(n_states), best_actions] = 1.0

    return policy