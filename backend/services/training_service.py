
from __future__ import annotations

import os
import threading
import time
import traceback
from typing import Optional

import numpy as np

from backend.services import db
from backend.services.simulation_service import simulation_service
from rl.dp.policy_iteration import policy_iteration
from rl.dp.value_iteration import value_iteration
from rl.environment.warehouse_env import WarehouseEnv
from rl.monte_carlo.every_visit_mc import every_visit_mc_control
from rl.monte_carlo.first_visit_mc import first_visit_mc_control
from rl.q_learning import q_learning
from rl.sarsa import sarsa
from rl.td.td_lambda import td_lambda_prediction
from rl.td.td_zero import td_zero_prediction
from rl.utils import estimate_mdp_model

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            "experiments", "results")

ALGORITHM_NAMES = {
    "policy_iteration": "Policy Iteration (DP)",
    "value_iteration": "Value Iteration (DP)",
    "first_visit_mc": "First-Visit Monte Carlo",
    "every_visit_mc": "Every-Visit Monte Carlo",
    "td_zero": "TD(0)",
    "td_lambda": "TD(lambda)",
    "sarsa": "SARSA",
    "q_learning": "Q-Learning",
}


class TrainingService:
    def __init__(self):
        self.lock = threading.RLock()
        self.status = {
            "running": False,
            "algorithm": None,
            "progress": 0.0,
            "message": "Idle",
            "last_run_id": None,
            "error": None,
        }

    def get_status(self) -> dict:
        with self.lock:
            return dict(self.status)

    def start_training(self, params: dict):
        with self.lock:
            if self.status["running"]:
                raise RuntimeError("A training job is already running. Please wait for it to finish.")
            self.status.update({"running": True, "algorithm": params["algorithm"],
                                 "progress": 0.0, "message": "Starting...", "error": None})
        thread = threading.Thread(target=self._run_job, args=(params,), daemon=True)
        thread.start()

    # ------------------------------------------------------------------ #
    def _set_progress(self, progress: float, message: str):
        with self.lock:
            self.status["progress"] = round(progress, 3)
            self.status["message"] = message

    def _run_job(self, params: dict):
        algorithm = params["algorithm"]
        try:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            t0 = time.time()

            env = WarehouseEnv(max_steps=params["max_steps"], seed=params["seed"],
                                arrival_rate=0.45)

            policy_table = None
            reward_curve = []
            extra_notes = ""

            if algorithm in ("policy_iteration", "value_iteration"):
                self._set_progress(0.05, "Estimating empirical MDP model P(s'|s,a), R(s,a)...")
                P, R = estimate_mdp_model(env, n_samples_per_pair=params["dp_samples_per_pair"],
                                           seed=params["seed"])
                self._set_progress(0.6, f"Running {algorithm}...")
                if algorithm == "policy_iteration":
                    V, policy_table, n_improve = policy_iteration(P, R, gamma=params["gamma"])
                    extra_notes = f"converged after {n_improve} improvement sweeps"
                else:
                    V, policy_table = value_iteration(P, R, gamma=params["gamma"])
                    extra_notes = "value iteration convergence via Bellman optimality backup"
                # Evaluate the resulting policy on fresh episodes for a fair reward curve
                reward_curve = self._evaluate_policy(policy_table, params, n_eval_episodes=100)

            elif algorithm == "first_visit_mc":
                Q, policy_table, reward_curve = first_visit_mc_control(
                    env, num_episodes=params["num_episodes"], gamma=params["gamma"],
                    epsilon_start=params["epsilon_start"], epsilon_min=params["epsilon_min"],
                    epsilon_decay=params["epsilon_decay"], seed=params["seed"])

            elif algorithm == "every_visit_mc":
                Q, policy_table, reward_curve = every_visit_mc_control(
                    env, num_episodes=params["num_episodes"], gamma=params["gamma"],
                    epsilon_start=params["epsilon_start"], epsilon_min=params["epsilon_min"],
                    epsilon_decay=params["epsilon_decay"], seed=params["seed"])

            elif algorithm == "sarsa":
                Q, policy_table, reward_curve = sarsa(
                    env, num_episodes=params["num_episodes"], alpha=params["alpha"],
                    gamma=params["gamma"], epsilon_start=params["epsilon_start"],
                    epsilon_min=params["epsilon_min"], epsilon_decay=params["epsilon_decay"],
                    seed=params["seed"])

            elif algorithm == "q_learning":
                Q, policy_table, reward_curve = q_learning(
                    env, num_episodes=params["num_episodes"], alpha=params["alpha"],
                    gamma=params["gamma"], epsilon_start=params["epsilon_start"],
                    epsilon_min=params["epsilon_min"], epsilon_decay=params["epsilon_decay"],
                    seed=params["seed"])

            elif algorithm in ("td_zero", "td_lambda"):
                # TD(0)/TD(lambda) are PREDICTION methods: they evaluate a given
                # policy rather than search for one. We first learn a policy with
                # Q-Learning (quickly) and then use TD to evaluate its value
                # function - this is documented in rl/td/td_zero.py.
                self._set_progress(0.1, "Learning a base policy with Q-Learning to evaluate...")
                base_episodes = max(300, params["num_episodes"] // 3)
                Q, policy_table, _ = q_learning(env, num_episodes=base_episodes,
                                                 alpha=params["alpha"], gamma=params["gamma"],
                                                 seed=params["seed"])
                self._set_progress(0.5, f"Running {algorithm} policy evaluation...")
                env2 = WarehouseEnv(max_steps=params["max_steps"], seed=params["seed"] + 1,
                                     arrival_rate=0.45)
                if algorithm == "td_zero":
                    V, reward_curve = td_zero_prediction(
                        env2, Q, num_episodes=params["num_episodes"], alpha=params["alpha"],
                        gamma=params["gamma"], seed=params["seed"])
                else:
                    V, reward_curve = td_lambda_prediction(
                        env2, Q, num_episodes=params["num_episodes"], alpha=params["alpha"],
                        gamma=params["gamma"], lam=params["lam"], seed=params["seed"])
                extra_notes = f"mean estimated state-value = {float(np.mean(V)):.3f}"
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            duration = time.time() - t0
            final_avg = float(np.mean(reward_curve[-max(1, len(reward_curve) // 10):])) \
                if reward_curve else None
            best_avg = float(np.max(_rolling_mean(reward_curve, 20))) if len(reward_curve) >= 5 \
                else final_avg

            model_path = None
            if policy_table is not None:
                model_path = os.path.join(RESULTS_DIR, f"{algorithm}_policy.npy")
                np.save(model_path, policy_table)

            run_id = db.insert_training_run(
                algorithm=algorithm, params=params, duration_seconds=duration,
                num_episodes=len(reward_curve) if reward_curve else params["num_episodes"],
                final_avg_reward=final_avg, best_avg_reward=best_avg,
                reward_curve=_downsample(reward_curve, 300), model_path=model_path,
                notes=extra_notes)

            if policy_table is not None:
                simulation_service.set_policy(policy_table, ALGORITHM_NAMES.get(algorithm, algorithm))

            with self.lock:
                self.status.update({"running": False, "progress": 1.0,
                                     "message": f"Finished {algorithm} in {duration:.1f}s",
                                     "last_run_id": run_id, "error": None})

        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            with self.lock:
                self.status.update({"running": False, "progress": 1.0,
                                     "message": "Training failed", "error": str(exc)})

    def _evaluate_policy(self, policy_table: np.ndarray, params: dict,
                          n_eval_episodes: int = 100) -> list:
        """Roll out a fixed (greedy) policy to build a reward curve for DP methods,
        which don't naturally produce an episode-by-episode training curve."""
        env = WarehouseEnv(max_steps=params["max_steps"], seed=params["seed"] + 7,
                            arrival_rate=0.45)
        rewards = []
        for ep in range(n_eval_episodes):
            state, _ = env.reset(seed=params["seed"] + 7 + ep)
            done = False
            total = 0.0
            while not done:
                action = int(np.argmax(policy_table[state]))
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
            rewards.append(total)
        return rewards


def _rolling_mean(values, window):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.array([0.0])
    if len(values) < window:
        window = len(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _downsample(values: list, max_points: int) -> list:
    if len(values) <= max_points:
        return [round(float(v), 4) for v in values]
    step = len(values) / max_points
    indices = [int(i * step) for i in range(max_points)]
    return [round(float(values[i]), 4) for i in indices]


training_service = TrainingService()
