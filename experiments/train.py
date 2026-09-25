from __future__ import annotations

# Import argparse for reading command-line arguments.
import argparse

# Import os for working with files and directories.
import os

# Import sys for modifying the Python module search path.
import sys

# Import time for measuring training duration.
import time

# Import NumPy for numerical calculations.
import numpy as np

# Add the project root directory to the Python module search path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the database service.
from backend.services import db  # noqa: E402

# Import the Policy Iteration algorithm.
from rl.dp.policy_iteration import policy_iteration  # noqa: E402

# Import the Value Iteration algorithm.
from rl.dp.value_iteration import value_iteration  # noqa: E402

# Import the warehouse reinforcement learning environment.
from rl.environment.warehouse_env import WarehouseEnv  # noqa: E402

# Import the First-Visit Monte Carlo algorithm.
from rl.monte_carlo.every_visit_mc import every_visit_mc_control  # noqa: E402

# Import the Every-Visit Monte Carlo algorithm.
from rl.monte_carlo.first_visit_mc import first_visit_mc_control  # noqa: E402

# Import the Q-Learning algorithm.
from rl.q_learning import q_learning  # noqa: E402

# Import the SARSA algorithm.
from rl.sarsa import sarsa  # noqa: E402

# Import the TD-Lambda prediction algorithm.
from rl.td.td_lambda import td_lambda_prediction  # noqa: E402

# Import the TD-Zero prediction algorithm.
from rl.td.td_zero import td_zero_prediction  # noqa: E402

# Import functions used for estimating the MDP model and calculating moving averages.
from rl.utils import estimate_mdp_model, moving_average  # noqa: E402

# Define the folder where training results will be stored.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Define the list of all supported reinforcement learning algorithms.
ALL_ALGORITHMS = ["policy_iteration", "value_iteration", "first_visit_mc", "every_visit_mc",
                   "td_zero", "td_lambda", "sarsa", "q_learning"]


# Evaluate a trained policy by running it through several episodes.
def evaluate_policy(policy_table, max_steps, seed, n_eval_episodes=100):

    # Create the warehouse environment for evaluation.
    env = WarehouseEnv(max_steps=max_steps, seed=seed, arrival_rate=0.45)

    # Create a list to store rewards from each evaluation episode.
    rewards = []

    # Run the policy for the requested number of evaluation episodes.
    for ep in range(n_eval_episodes):

        # Reset the environment for the current episode.
        state, _ = env.reset(seed=seed + ep)

        # Mark the episode as not finished.
        done = False

        # Start the total reward at zero.
        total = 0.0

        # Continue until the episode finishes.
        while not done:

            # Select the action with the highest policy value.
            action = int(np.argmax(policy_table[state]))

            # Perform the selected action.
            state, reward, terminated, truncated, _ = env.step(action)

            # Add the reward to the episode total.
            total += reward

            # Check whether the episode has ended.
            done = terminated or truncated

        # Store the total reward for this episode.
        rewards.append(total)

    # Return all evaluation rewards.
    return rewards


# Train one selected reinforcement learning algorithm.
def train_one(algorithm: str, args) -> dict:

    # Display the name of the algorithm being trained.
    print(f"\n=== Training: {algorithm} ===")

    # Store the starting time for measuring training duration.
    t0 = time.time()

    # Create the warehouse environment.
    env = WarehouseEnv(max_steps=args.max_steps, seed=args.seed, arrival_rate=0.45)

    # Start with no policy table.
    policy_table = None

    # Create an empty list for storing reward values.
    reward_curve = []

    # Create an empty notes string.
    notes = ""

    # Handle the Dynamic Programming algorithms.
    if algorithm in ("policy_iteration", "value_iteration"):

        # Display a message while estimating the MDP model.
        print("Estimating empirical MDP model (this samples every state/action pair)...")

        # Estimate the transition and reward models.
        P, R = estimate_mdp_model(env, n_samples_per_pair=args.dp_samples, seed=args.seed)

        # Check whether Policy Iteration was selected.
        if algorithm == "policy_iteration":

            # Run Policy Iteration and get the resulting values and policy.
            V, policy_table, n_improve = policy_iteration(P, R, gamma=args.gamma)

            # Store information about the improvement sweeps.
            notes = f"{n_improve} improvement sweeps"

        # Otherwise, run Value Iteration.
        else:

            # Run Value Iteration and get the resulting values and policy.
            V, policy_table = value_iteration(P, R, gamma=args.gamma)

            # Store a note about the convergence method.
            notes = "converged via Bellman optimality backups"

        # Evaluate the resulting policy.
        reward_curve = evaluate_policy(policy_table, args.max_steps, args.seed)

    # Handle the First-Visit Monte Carlo algorithm.
    elif algorithm == "first_visit_mc":

        # Train the First-Visit Monte Carlo policy.
        _, policy_table, reward_curve = first_visit_mc_control(
            env, num_episodes=args.episodes, gamma=args.gamma, seed=args.seed)

    # Handle the Every-Visit Monte Carlo algorithm.
    elif algorithm == "every_visit_mc":

        # Train the Every-Visit Monte Carlo policy.
        _, policy_table, reward_curve = every_visit_mc_control(
            env, num_episodes=args.episodes, gamma=args.gamma, seed=args.seed)

    # Handle the SARSA algorithm.
    elif algorithm == "sarsa":

        # Train the SARSA policy.
        _, policy_table, reward_curve = sarsa(
            env, num_episodes=args.episodes, alpha=args.alpha, gamma=args.gamma, seed=args.seed)

    # Handle the Q-Learning algorithm.
    elif algorithm == "q_learning":

        # Train the Q-Learning policy.
        _, policy_table, reward_curve = q_learning(
            env, num_episodes=args.episodes, alpha=args.alpha, gamma=args.gamma, seed=args.seed)

    # Handle the TD-Zero and TD-Lambda algorithms.
    elif algorithm in ("td_zero", "td_lambda"):

        # Use a smaller number of episodes to create the initial Q-Learning policy.
        base_episodes = max(300, args.episodes // 3)

        # Train a Q-Learning model that will provide the policy being evaluated.
        Q, policy_table, _ = q_learning(env, num_episodes=base_episodes, alpha=args.alpha,
                                         gamma=args.gamma, seed=args.seed)

        # Create a second environment for TD prediction.
        env2 = WarehouseEnv(max_steps=args.max_steps, seed=args.seed + 1, arrival_rate=0.45)

        # Check whether TD-Zero was selected.
        if algorithm == "td_zero":

            # Run TD-Zero prediction.
            V, reward_curve = td_zero_prediction(env2, Q, num_episodes=args.episodes,
                                                  alpha=args.alpha, gamma=args.gamma, seed=args.seed)

        # Otherwise, run TD-Lambda prediction.
        else:

            # Run TD-Lambda prediction.
            V, reward_curve = td_lambda_prediction(env2, Q, num_episodes=args.episodes,
                                                     alpha=args.alpha, gamma=args.gamma,
                                                     lam=args.lam, seed=args.seed)

        # Store information about the value estimates.
        notes = f"mean V = {float(np.mean(V)):.3f} (evaluates Q-Learning's policy)"

    # Handle an unknown algorithm name.
    else:

        # Raise an error when the algorithm is not recognized.
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Calculate how long the training process took.
    duration = time.time() - t0

    # Calculate the average reward from the final part of the reward curve.
    final_avg = float(np.mean(reward_curve[-max(1, len(reward_curve) // 10):])) if reward_curve else None

    # Display the training results.
    print(f"Finished in {duration:.2f}s | episodes={len(reward_curve)} | "
          f"final avg reward={final_avg} | {notes}")

    # Make sure the results directory exists.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save the policy table if one was created.
    if policy_table is not None:

        # Create the file path for the policy table.
        model_path = os.path.join(RESULTS_DIR, f"{algorithm}_policy.npy")

        # Save the policy table as a NumPy file.
        np.save(model_path, policy_table)

        # Display where the policy table was saved.
        print(f"Saved policy table -> {model_path}")

    # No model path is available when there is no policy table.
    else:

        # Store None to indicate that no model was saved.
        model_path = None

    # Save the reward curve if it contains values.
    if reward_curve:

        # Create the file path for the reward plot.
        plot_path = os.path.join(RESULTS_DIR, f"{algorithm}_rewards.png")

        # Create and save the reward plot.
        if _save_plot(reward_curve, algorithm, plot_path):

            # Display where the reward plot was saved.
            print(f"Saved reward plot -> {plot_path}")

    # Store the training parameters.
    params = {"algorithm": algorithm, "num_episodes": args.episodes, "alpha": args.alpha,
              "gamma": args.gamma, "max_steps": args.max_steps, "seed": args.seed,
              "dp_samples_per_pair": args.dp_samples, "lam": args.lam}

    # Save the training run information in the database.
    run_id = db.insert_training_run(
        algorithm=algorithm, params=params, duration_seconds=duration,
        num_episodes=len(reward_curve) if reward_curve else args.episodes,
        final_avg_reward=final_avg, best_avg_reward=final_avg,
        reward_curve=[round(float(v), 4) for v in reward_curve][:300],
        model_path=model_path, notes=notes)

    # Display the database training run ID.
    print(f"Logged training run id={run_id} to database/warehouse.db")

    # Return the main training results.
    return {"algorithm": algorithm, "duration": duration, "final_avg_reward": final_avg}


# Save a reward curve as a plot.
def _save_plot(reward_curve, algorithm, path):

    # Try to import Matplotlib.
    try:
        # Import the Matplotlib package.
        import matplotlib

        # Use a non-interactive backend for saving plots.
        matplotlib.use("Agg")

        # Import the plotting module.
        import matplotlib.pyplot as plt

    # Handle the case where Matplotlib is not installed.
    except ImportError:

        # Display a message and skip creating the PNG plot.
        print("(matplotlib not installed - skipping PNG plot; reward curve is still "
              "saved in the database and visible in the web dashboard)")

        # Indicate that the plot was not created.
        return False

    # Calculate a smoothed version of the reward curve.
    smoothed = moving_average(reward_curve, window=max(1, len(reward_curve) // 20))

    # Create a new figure for the plot.
    plt.figure(figsize=(7, 4))

    # Plot the original reward values.
    plt.plot(reward_curve, alpha=0.3, label="raw")

    # Plot the moving average.
    plt.plot(smoothed, label="moving average")

    # Add a title to the plot.
    plt.title(f"Reward per episode - {algorithm}")

    # Label the horizontal axis.
    plt.xlabel("Episode")

    # Label the vertical axis.
    plt.ylabel("Total reward")

    # Display the plot legend.
    plt.legend()

    # Adjust the plot layout.
    plt.tight_layout()

    # Save the plot to the specified path.
    plt.savefig(path, dpi=110)

    # Close the plot.
    plt.close()

    # Indicate that the plot was saved successfully.
    return True


# Main function that handles command-line execution.
def main():

    # Create the command-line argument parser.
    parser = argparse.ArgumentParser(description="Train warehouse RL algorithms.")

    # Add an argument for selecting one algorithm.
    parser.add_argument("--algorithm", type=str, default="q_learning", choices=ALL_ALGORITHMS)

    # Add an option to train every algorithm.
    parser.add_argument("--all", action="store_true", help="Train every algorithm sequentially")

    # Add an argument for the number of training episodes.
    parser.add_argument("--episodes", type=int, default=2000)

    # Add an argument for the learning rate.
    parser.add_argument("--alpha", type=float, default=0.1)

    # Add an argument for the discount factor.
    parser.add_argument("--gamma", type=float, default=0.95)

    # Add an argument for the TD-Lambda parameter.
    parser.add_argument("--lam", type=float, default=0.8)

    # Add an argument for the maximum number of environment steps.
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=60)

    # Add an argument for the number of samples used by Dynamic Programming.
    parser.add_argument("--dp-samples", dest="dp_samples", type=int, default=10)

    # Add an argument for the random seed.
    parser.add_argument("--seed", type=int, default=1)

    # Read all command-line arguments.
    args = parser.parse_args()

    # Initialize the database.
    db.init_db()

    # Check whether all algorithms should be trained.
    if args.all:

        # Train every algorithm sequentially and store the results.
        results = [train_one(algo, args) for algo in ALL_ALGORITHMS]

        # Display the summary heading.
        print("\n=== Summary ===")

        # Display the results for each algorithm.
        for r in results:
            print(f"{r['algorithm']:20s} duration={r['duration']:.2f}s "
                  f"final_avg_reward={r['final_avg_reward']}")

    # Train only the selected algorithm.
    else:
        train_one(args.algorithm, args)


# Run the main function when this file is executed directly.
if __name__ == "__main__":
    main()
