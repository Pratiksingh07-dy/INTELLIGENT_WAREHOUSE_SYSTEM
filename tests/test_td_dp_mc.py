import numpy as np

from rl.dp.policy_iteration import policy_iteration
from rl.dp.value_iteration import iterative_policy_evaluation, value_iteration
from rl.environment.warehouse_env import WarehouseEnv
from rl.monte_carlo.every_visit_mc import every_visit_mc_control
from rl.monte_carlo.first_visit_mc import first_visit_mc_control
from rl.q_learning import q_learning
from rl.td.td_lambda import td_lambda_prediction
from rl.td.td_zero import td_zero_prediction
from rl.utils import estimate_mdp_model


def _tiny_model(seed=1):
    env = WarehouseEnv(seed=seed, max_steps=20)
    P, R = estimate_mdp_model(env, n_samples_per_pair=3, seed=seed)
    return env, P, R


def test_estimate_mdp_model_shapes_and_validity():
    env, P, R = _tiny_model()
    assert P.shape == (env.n_states, env.n_actions, env.n_states)
    assert R.shape == (env.n_states, env.n_actions)
    # Every (s,a) row of P should sum to (approximately) 1
    sums = P.sum(axis=2)
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_value_iteration_converges_to_finite_values():
    env, P, R = _tiny_model()
    V, policy = value_iteration(P, R, gamma=0.9, max_iterations=200)
    assert V.shape == (env.n_states,)
    assert np.all(np.isfinite(V))
    assert np.allclose(policy.sum(axis=1), 1.0)


def test_policy_iteration_converges_and_matches_value_iteration_ballpark():
    env, P, R = _tiny_model()
    V_pi, policy_pi, n_improve = policy_iteration(P, R, gamma=0.9)
    V_vi, policy_vi = value_iteration(P, R, gamma=0.9, max_iterations=200)
    assert n_improve >= 1
    # Both should converge to (approximately) the same optimal value function
    assert np.allclose(V_pi, V_vi, atol=1.0)


def test_iterative_policy_evaluation_runs():
    env, P, R = _tiny_model()
    uniform_policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
    V = iterative_policy_evaluation(uniform_policy, P, R, gamma=0.9)
    assert V.shape == (env.n_states,)
    assert np.all(np.isfinite(V))


def test_first_visit_mc_control_runs():
    env = WarehouseEnv(seed=1, max_steps=20)
    Q, policy, rewards = first_visit_mc_control(env, num_episodes=40, seed=1)
    assert Q.shape == (env.n_states, env.n_actions)
    assert len(rewards) == 40


def test_every_visit_mc_control_runs():
    env = WarehouseEnv(seed=1, max_steps=20)
    Q, policy, rewards = every_visit_mc_control(env, num_episodes=40, seed=1)
    assert Q.shape == (env.n_states, env.n_actions)
    assert len(rewards) == 40


def test_td_zero_prediction_runs():
    env = WarehouseEnv(seed=1, max_steps=20)
    Q, _, _ = q_learning(env, num_episodes=60, seed=1)
    env2 = WarehouseEnv(seed=2, max_steps=20)
    V, rewards = td_zero_prediction(env2, Q, num_episodes=40, seed=1)
    assert V.shape == (env2.n_states,)
    assert len(rewards) == 40
    assert np.all(np.isfinite(V))


def test_td_lambda_prediction_runs():
    env = WarehouseEnv(seed=1, max_steps=20)
    Q, _, _ = q_learning(env, num_episodes=60, seed=1)
    env2 = WarehouseEnv(seed=2, max_steps=20)
    V, rewards = td_lambda_prediction(env2, Q, num_episodes=40, lam=0.7, seed=1)
    assert V.shape == (env2.n_states,)
    assert len(rewards) == 40
    assert np.all(np.isfinite(V))
