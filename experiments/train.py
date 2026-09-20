"""
experiments/train.py
---------------------
Command-line script to train any of the 8 RL algorithms on the
WarehouseEnv, print a summary, and save:
    - the learned policy/value table (experiments/results/<algo>_policy.npy)
    - a reward-per-episode plot (experiments/results/<algo>_rewards.png)
    - a row in the SQLite training-run history (same DB the web app uses)

Usage
-----
    python experiments/train.py --algorithm q_learning --episodes 2000
    python experiments/train.py --algorithm policy_iteration
    python experiments/train.py --all          # trains every algorithm back-to-back

Run from the project root (so that `rl`, `simulation`, `backend` are importable).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services import db  # noqa: E402
from rl.dp.policy_iteration import policy_iteration  # noqa: E402
from rl.dp.value_iteration import value_iteration  # noqa: E402
from rl.environment.warehouse_env import WarehouseEnv  # noqa: E402
from rl.monte_carlo.every_visit_mc import every_visit_mc_control  # noqa: E402
from rl.monte_carlo.first_visit_mc import first_visit_mc_control  # noqa: E402
from rl.q_learning import q_learning  # noqa: E402
from rl.sarsa import sarsa  # noqa: E402
from rl.td.td_lambda import td_lambda_prediction  # noqa: E402
from rl.td.td_zero import td_zero_prediction  # noqa: E402
from rl.utils import estimate_mdp_model, moving_average  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ALL_ALGORITHMS = ["policy_iteration", "value_iteration", "first_visit_mc", "every_visit_mc",
                   "td_zero", "td_lambda", "sarsa", "q_learning"]


def evaluate_policy(policy_table, max_steps, seed, n_eval_episodes=100):
    env = WarehouseEnv(max_steps=max_steps, seed=seed, arrival_rate=0.45)
    rewards = []
    for ep in range(n_eval_episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            action = int(np.argmax(policy_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
    return rewards


def train_one(algorithm: str, args) -> dict:
    print(f"\n=== Training: {algorithm} ===")
    t0 = time.time()
    env = WarehouseEnv(max_steps=args.max_steps, seed=args.seed, arrival_rate=0.45)
    policy_table = None
    reward_curve = []
    notes = ""

    if algorithm in ("policy_iteration", "value_iteration"):
        print("Estimating empirical MDP model (this samples every state/action pair)...")
        P, R = estimate_mdp_model(env, n_samples_per_pair=args.dp_samples, seed=args.seed)
        if algorithm == "policy_iteration":
            V, policy_table, n_improve = policy_iteration(P, R, gamma=args.gamma)
            notes = f"{n_improve} improvement sweeps"
        else:
            V, policy_table = value_iteration(P, R, gamma=args.gamma)
            notes = "converged via Bellman optimality backups"
        reward_curve = evaluate_policy(policy_table, args.max_steps, args.seed)

    elif algorithm == "first_visit_mc":
        _, policy_table, reward_curve = first_visit_mc_control(
            env, num_episodes=args.episodes, gamma=args.gamma, seed=args.seed)

    elif algorithm == "every_visit_mc":
        _, policy_table, reward_curve = every_visit_mc_control(
            env, num_episodes=args.episodes, gamma=args.gamma, seed=args.seed)

    elif algorithm == "sarsa":
        _, policy_table, reward_curve = sarsa(
            env, num_episodes=args.episodes, alpha=args.alpha, gamma=args.gamma, seed=args.seed)

    elif algorithm == "q_learning":
        _, policy_table, reward_curve = q_learning(
            env, num_episodes=args.episodes, alpha=args.alpha, gamma=args.gamma, seed=args.seed)

    elif algorithm in ("td_zero", "td_lambda"):
        base_episodes = max(300, args.episodes // 3)
        Q, policy_table, _ = q_learning(env, num_episodes=base_episodes, alpha=args.alpha,
                                         gamma=args.gamma, seed=args.seed)
        env2 = WarehouseEnv(max_steps=args.max_steps, seed=args.seed + 1, arrival_rate=0.45)
        if algorithm == "td_zero":
            V, reward_curve = td_zero_prediction(env2, Q, num_episodes=args.episodes,
                                                  alpha=args.alpha, gamma=args.gamma, seed=args.seed)
        else:
            V, reward_curve = td_lambda_prediction(env2, Q, num_episodes=args.episodes,
                                                     alpha=args.alpha, gamma=args.gamma,
                                                     lam=args.lam, seed=args.seed)
        notes = f"mean V = {float(np.mean(V)):.3f} (evaluates Q-Learning's policy)"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    duration = time.time() - t0
    final_avg = float(np.mean(reward_curve[-max(1, len(reward_curve) // 10):])) if reward_curve else None
    print(f"Finished in {duration:.2f}s | episodes={len(reward_curve)} | "
          f"final avg reward={final_avg} | {notes}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if policy_table is not None:
        model_path = os.path.join(RESULTS_DIR, f"{algorithm}_policy.npy")
        np.save(model_path, policy_table)
        print(f"Saved policy table -> {model_path}")
    else:
        model_path = None

    if reward_curve:
        plot_path = os.path.join(RESULTS_DIR, f"{algorithm}_rewards.png")
        if _save_plot(reward_curve, algorithm, plot_path):
            print(f"Saved reward plot -> {plot_path}")

    params = {"algorithm": algorithm, "num_episodes": args.episodes, "alpha": args.alpha,
              "gamma": args.gamma, "max_steps": args.max_steps, "seed": args.seed,
              "dp_samples_per_pair": args.dp_samples, "lam": args.lam}
    run_id = db.insert_training_run(
        algorithm=algorithm, params=params, duration_seconds=duration,
        num_episodes=len(reward_curve) if reward_curve else args.episodes,
        final_avg_reward=final_avg, best_avg_reward=final_avg,
        reward_curve=[round(float(v), 4) for v in reward_curve][:300],
        model_path=model_path, notes=notes)
    print(f"Logged training run id={run_id} to database/warehouse.db")

    return {"algorithm": algorithm, "duration": duration, "final_avg_reward": final_avg}


def _save_plot(reward_curve, algorithm, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping PNG plot; reward curve is still "
              "saved in the database and visible in the web dashboard)")
        return False
    smoothed = moving_average(reward_curve, window=max(1, len(reward_curve) // 20))
    plt.figure(figsize=(7, 4))
    plt.plot(reward_curve, alpha=0.3, label="raw")
    plt.plot(smoothed, label="moving average")
    plt.title(f"Reward per episode - {algorithm}")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Train warehouse RL algorithms.")
    parser.add_argument("--algorithm", type=str, default="q_learning", choices=ALL_ALGORITHMS)
    parser.add_argument("--all", action="store_true", help="Train every algorithm sequentially")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lam", type=float, default=0.8)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=60)
    parser.add_argument("--dp-samples", dest="dp_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    db.init_db()

    if args.all:
        results = [train_one(algo, args) for algo in ALL_ALGORITHMS]
        print("\n=== Summary ===")
        for r in results:
            print(f"{r['algorithm']:20s} duration={r['duration']:.2f}s "
                  f"final_avg_reward={r['final_avg_reward']}")
    else:
        train_one(args.algorithm, args)


if __name__ == "__main__":
    main()
