import numpy as np

from rl.environment.warehouse_env import WarehouseEnv
from simulation.warehouse import NUM_ACTIONS


def test_reset_returns_valid_state():
    env = WarehouseEnv(seed=1, max_steps=50)
    state, info = env.reset(seed=1)
    assert isinstance(state, int)
    assert 0 <= state < env.n_states
    assert "raw" in info


def test_step_returns_valid_tuple():
    env = WarehouseEnv(seed=1, max_steps=50)
    env.reset(seed=1)
    next_state, reward, terminated, truncated, info = env.step(0)
    assert 0 <= next_state < env.n_states
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_episode_truncates_at_max_steps():
    env = WarehouseEnv(seed=1, max_steps=10)
    env.reset(seed=1)
    truncated = False
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if truncated:
            break
    assert truncated is True


def test_state_encode_decode_roundtrip():
    env = WarehouseEnv(seed=1, max_steps=50)
    for s in [0, 10, 100, 300, env.n_states - 1]:
        decoded = env.decode_state(s)
        re_encoded = int(np.ravel_multi_index(decoded, env._dims))
        assert re_encoded == s


def test_action_space_size():
    env = WarehouseEnv(seed=1, max_steps=50)
    assert env.n_actions == NUM_ACTIONS == 5


def test_set_discrete_state_matches_encoding():
    env = WarehouseEnv(seed=1, max_steps=50)
    target_state = 42
    env.set_discrete_state(target_state)
    encoded = env.encode_state()
    assert encoded == target_state
