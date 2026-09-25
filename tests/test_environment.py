# Import NumPy for numerical calculations.
import numpy as np

# Import the warehouse reinforcement learning environment.
from rl.environment.warehouse_env import WarehouseEnv

# Import the total number of available actions.
from simulation.warehouse import NUM_ACTIONS


# Test that resetting the environment returns a valid state.
def test_reset_returns_valid_state():

    # Create a warehouse environment.
    env = WarehouseEnv(seed=1, max_steps=50)

    # Reset the environment and get the initial state and information.
    state, info = env.reset(seed=1)

    # Check that the returned state is an integer.
    assert isinstance(state, int)

    # Check that the state is within the valid state range.
    assert 0 <= state < env.n_states

    # Check that raw simulation information is included.
    assert "raw" in info


# Test that one environment step returns valid values.
def test_step_returns_valid_tuple():

    # Create a warehouse environment.
    env = WarehouseEnv(seed=1, max_steps=50)

    # Reset the environment before taking a step.
    env.reset(seed=1)

    # Perform action 0 and store the returned values.
    next_state, reward, terminated, truncated, info = env.step(0)

    # Check that the next state is within the valid state range.
    assert 0 <= next_state < env.n_states

    # Check that the reward is a floating-point number.
    assert isinstance(reward, float)

    # Check that the terminated value is a boolean.
    assert isinstance(terminated, bool)

    # Check that the truncated value is a boolean.
    assert isinstance(truncated, bool)


# Test that the episode stops after reaching the maximum number of steps.
def test_episode_truncates_at_max_steps():

    # Create an environment with a maximum of 10 steps.
    env = WarehouseEnv(seed=1, max_steps=10)

    # Reset the environment before starting the episode.
    env.reset(seed=1)

    # Start with the assumption that the episode is not truncated.
    truncated = False

    # Perform up to 10 steps.
    for _ in range(10):

        # Take a randomly selected action.
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())

        # Stop the loop if the episode has been truncated.
        if truncated:
            break

    # Check that the episode was truncated.
    assert truncated is True


# Test that encoding and decoding a state gives the original state.
def test_state_encode_decode_roundtrip():

    # Create a warehouse environment.
    env = WarehouseEnv(seed=1, max_steps=50)

    # Test several different state numbers.
    for s in [0, 10, 100, 300, env.n_states - 1]:

        # Decode the state number into its individual values.
        decoded = env.decode_state(s)

        # Convert the decoded values back into a state number.
        re_encoded = int(np.ravel_multi_index(decoded, env._dims))

        # Check that the encoded state matches the original state.
        assert re_encoded == s


# Test that the action space contains the expected number of actions.
def test_action_space_size():

    # Create a warehouse environment.
    env = WarehouseEnv(seed=1, max_steps=50)

    # Check that the environment has exactly five actions.
    assert env.n_actions == NUM_ACTIONS == 5


# Test that setting a discrete state produces the same encoded state.
def test_set_discrete_state_matches_encoding():

    # Create a warehouse environment.
    env = WarehouseEnv(seed=1, max_steps=50)

    # Select the state that should be created.
    target_state = 42

    # Set the environment to the selected discrete state.
    env.set_discrete_state(target_state)

    # Encode the current environment state.
    encoded = env.encode_state()

    # Check that the encoded state matches the target state.
    assert encoded == target_state
