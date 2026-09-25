from __future__ import annotations

# Import argparse for reading command-line arguments.
import argparse

# Import glob for finding files that match a pattern.
import glob

# Import os for working with files and directories.
import os

# Import sys for modifying the Python module search path.
import sys

# Import NumPy for numerical calculations and arrays.
import numpy as np

# Add the project root directory to the Python module search path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the warehouse reinforcement learning environment.
from rl.environment.warehouse_env import WarehouseEnv  # noqa: E402

# Define the folder where saved results are stored.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# Evaluate a saved policy using multiple episodes.
def evaluate(policy_table: np.ndarray, n_episodes: int = 100, max_steps: int = 60,
             seed: int = 123) -> dict:

    # Create the warehouse environment for evaluation.
    env = WarehouseEnv(max_steps=max_steps, seed=seed, arrival_rate=0.45)

    # Create lists to store the results from each episode.
    total_rewards, waits, throughputs, utils = [], [], [], []

    # Run the policy for the requested number of episodes.
    for ep in range(n_episodes):

        # Reset the environment for the current episode.
        state, _ = env.reset(seed=seed + ep)

        # Mark the episode as not finished.
        done = False

        # Start the reward for this episode at zero.
        ep_reward = 0.0

        # Continue running until the episode finishes.
        while not done:

            # Select the action with the highest policy value.
            action = int(np.argmax(policy_table[state]))

            # Perform the selected action in the environment.
            state, reward, terminated, truncated, _ = env.step(action)

            # Add the reward to the episode total.
            ep_reward += reward

            # Check whether the episode has ended.
            done = terminated or truncated

        # Get the final simulation information for this episode.
        d = env.sim.to_dict()

        # Store the total reward for the episode.
        total_rewards.append(ep_reward)

        # Store the average waiting time.
        waits.append(d["metrics"]["average_waiting_time"])

        # Store the throughput.
        throughputs.append(d["metrics"]["throughput"])

        # Store the resource utilization.
        utils.append(d["metrics"]["resource_utilization"])

    # Return the average and standard deviation of the evaluation results.
    return {
        "avg_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "avg_waiting_time": float(np.mean(waits)),
        "avg_throughput": float(np.mean(throughputs)),
        "avg_utilization": float(np.mean(utils)),
    }


# Main function for evaluating saved policies.
def main():

    # Create the command-line argument parser.
    parser = argparse.ArgumentParser(description="Evaluate saved warehouse RL policies.")

    # Add an optional argument for evaluating one specific algorithm.
    parser.add_argument("--algorithm", type=str, default=None,
                         help="Evaluate a single algorithm's saved policy; omit to evaluate all")

    # Add an argument for the number of evaluation episodes.
    parser.add_argument("--episodes", type=int, default=100)

    # Add an argument for the maximum number of environment steps.
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=60)

    # Read all command-line arguments.
    args = parser.parse_args()

    # Check whether a specific algorithm was requested.
    if args.algorithm:

        # Create the path for that algorithm's saved policy.
        paths = [os.path.join(RESULTS_DIR, f"{args.algorithm}_policy.npy")]

    # Otherwise, find all saved policy files.
    else:

        # Find and sort all saved policy files in the results directory.
        paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_policy.npy")))

    # Check whether there are no saved policies available.
    if not paths or not any(os.path.exists(p) for p in paths):

        # Display instructions for creating saved policies.
        print("No saved policies found. Run experiments/train.py first, e.g.:\n"
              "    python experiments/train.py --algorithm q_learning --episodes 2000")

        # Stop the program.
        return

    # Display the headings for the evaluation results.
    print(f"{'Algorithm':<20} {'Avg Reward':>12} {'Std':>8} {'Avg Wait':>10} "
          f"{'Throughput':>12} {'Utilization':>12}")

    # Print a separator line below the headings.
    print("-" * 78)

    # Process each saved policy file.
    for path in paths:

        # Skip the file if it does not exist.
        if not os.path.exists(path):
            continue

        # Get the algorithm name from the policy file name.
        algo_name = os.path.basename(path).replace("_policy.npy", "")

        # Load the saved policy table.
        policy_table = np.load(path)

        # Evaluate the loaded policy.
        metrics = evaluate(policy_table, n_episodes=args.episodes, max_steps=args.max_steps)

        # Display the evaluation results.
        print(f"{algo_name:<20} {metrics['avg_reward']:>12.2f} {metrics['std_reward']:>8.2f} "
              f"{metrics['avg_waiting_time']:>10.2f} {metrics['avg_throughput']:>12.3f} "
              f"{metrics['avg_utilization']:>12.3f}")


# Run the main function when this file is executed directly.
if __name__ == "__main__":
    main()
