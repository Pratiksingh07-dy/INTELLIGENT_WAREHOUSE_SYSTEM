"""
experiments/evaluate.py
-------------------------
Loads previously trained policies (experiments/results/<algo>_policy.npy)
and evaluates them over a batch of fresh episodes, printing a comparison
table of average reward, average waiting time, throughput and resource
utilization - the same kind of metrics shown on the web dashboard's
"Algorithm Comparison" page.

Usage
-----
    python experiments/evaluate.py                       # evaluate all saved policies
    python experiments/evaluate.py --algorithm q_learning  # evaluate just one
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.environment.warehouse_env import WarehouseEnv  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def evaluate(policy_table: np.ndarray, n_episodes: int = 100, max_steps: int = 60,
             seed: int = 123) -> dict:
    env = WarehouseEnv(max_steps=max_steps, seed=seed, arrival_rate=0.45)
    total_rewards, waits, throughputs, utils = [], [], [], []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        while not done:
            action = int(np.argmax(policy_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        d = env.sim.to_dict()
        total_rewards.append(ep_reward)
        waits.append(d["metrics"]["average_waiting_time"])
        throughputs.append(d["metrics"]["throughput"])
        utils.append(d["metrics"]["resource_utilization"])

    return {
        "avg_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "avg_waiting_time": float(np.mean(waits)),
        "avg_throughput": float(np.mean(throughputs)),
        "avg_utilization": float(np.mean(utils)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved warehouse RL policies.")
    parser.add_argument("--algorithm", type=str, default=None,
                         help="Evaluate a single algorithm's saved policy; omit to evaluate all")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=60)
    args = parser.parse_args()

    if args.algorithm:
        paths = [os.path.join(RESULTS_DIR, f"{args.algorithm}_policy.npy")]
    else:
        paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_policy.npy")))

    if not paths or not any(os.path.exists(p) for p in paths):
        print("No saved policies found. Run experiments/train.py first, e.g.:\n"
              "    python experiments/train.py --algorithm q_learning --episodes 2000")
        return

    print(f"{'Algorithm':<20} {'Avg Reward':>12} {'Std':>8} {'Avg Wait':>10} "
          f"{'Throughput':>12} {'Utilization':>12}")
    print("-" * 78)
    for path in paths:
        if not os.path.exists(path):
            continue
        algo_name = os.path.basename(path).replace("_policy.npy", "")
        policy_table = np.load(path)
        metrics = evaluate(policy_table, n_episodes=args.episodes, max_steps=args.max_steps)
        print(f"{algo_name:<20} {metrics['avg_reward']:>12.2f} {metrics['std_reward']:>8.2f} "
              f"{metrics['avg_waiting_time']:>10.2f} {metrics['avg_throughput']:>12.3f} "
              f"{metrics['avg_utilization']:>12.3f}")


if __name__ == "__main__":
    main()
