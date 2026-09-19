from __future__ import annotations

from typing import Tuple

import numpy as np

# policy evaluation
from rl.dp.value_iteration import iterative_policy_evaluation


# policy iteration
def policy_iteration(P: np.ndarray, R: np.ndarray, gamma: float = 0.95,
                      theta: float = 1e-4, max_iterations: int = 100
                      ) -> Tuple[np.ndarray, np.ndarray, int]:


    # gamma: gives importance to future rewards
    # 0.95 keeps future warehouse rewards important

    # theta: convergence threshold
    # 0.0001 means stop evaluation when value changes are very small

    # max_iterations: maximum number of policy improvement rounds
    # 100 prevents the algorithm from running for too long

    
    # number of states and actions
    n_states, n_actions = R.shape

    # initial random policy
    policy = np.ones((n_states, n_actions)) / n_actions

    for iteration in range(max_iterations):

        # policy evaluation
        V = iterative_policy_evaluation(policy, P, R, gamma=gamma, theta=theta)

        # calculate action values
        Q = R + gamma * P.dot(V)                # (n_states, n_actions)

        # select best action
        new_best_actions = np.argmax(Q, axis=1)

        # get old best actions
        old_best_actions = np.argmax(policy, axis=1)

        # create new policy
        new_policy = np.zeros((n_states, n_actions))

        # set best action probability to 1
        new_policy[np.arange(n_states), new_best_actions] = 1.0

        # check if policy changed
        policy_stable = np.array_equal(new_best_actions, old_best_actions)

        # update policy
        policy = new_policy

        # stop if policy is stable
        if policy_stable and iteration > 0:
            return V, policy, iteration + 1

    # return final policy
    return V, policy, max_iterations