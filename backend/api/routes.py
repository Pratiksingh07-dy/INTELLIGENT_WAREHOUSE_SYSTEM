from __future__ import annotations

import os

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.models.schemas import (ArrivalRateRequest, GenerateOrdersRequest,
                                     PolicySelectRequest, SpeedRequest, TrainRequest)
from backend.services import db
from backend.services.simulation_service import simulation_service
from backend.services.training_service import ALGORITHM_NAMES, training_service
from simulation.warehouse import ACTION_NAMES, NUM_ACTIONS

# create API router with /api prefix
router = APIRouter(prefix="/api")

# metadata used by the frontend to display RL algorithms
ALGORITHM_METADATA = {
    "policy_iteration": {"label": "Policy Iteration", "type": "Dynamic Programming",
                          "model_required": True, "learning_style": "Planning (policy eval + improve)",
                          "on_off_policy": "N/A"},
    "value_iteration": {"label": "Value Iteration", "type": "Dynamic Programming",
                         "model_required": True, "learning_style": "Planning (Bellman optimality)",
                         "on_off_policy": "N/A"},
    "first_visit_mc": {"label": "First-Visit Monte Carlo", "type": "Monte Carlo",
                        "model_required": False, "learning_style": "Episode-based",
                        "on_off_policy": "On-policy"},
    "every_visit_mc": {"label": "Every-Visit Monte Carlo", "type": "Monte Carlo",
                        "model_required": False, "learning_style": "Episode-based",
                        "on_off_policy": "On-policy"},
    "td_zero": {"label": "TD(0)", "type": "Temporal Difference",
                "model_required": False, "learning_style": "Bootstrapping (prediction)",
                "on_off_policy": "On-policy (evaluates given policy)"},
    "td_lambda": {"label": "TD(lambda)", "type": "Temporal Difference",
                  "model_required": False, "learning_style": "Bootstrapping + eligibility traces",
                  "on_off_policy": "On-policy (evaluates given policy)"},
    "sarsa": {"label": "SARSA", "type": "Temporal Difference",
              "model_required": False, "learning_style": "Bootstrapping (control)",
              "on_off_policy": "On-policy"},
    "q_learning": {"label": "Q-Learning", "type": "Temporal Difference",
                   "model_required": False, "learning_style": "Bootstrapping (control)",
                   "on_off_policy": "Off-policy"},
}

# get current warehouse state
@router.get("/state")
def get_state():
    return simulation_service.get_state()


# control simulation start
@router.post("/control/start")
def control_start():
    simulation_service.start()
    return {"status": "running"}


# control simulation pause
@router.post("/control/pause")
def control_pause():
    simulation_service.pause()
    return {"status": "paused"}


# reset simulation state
@router.post("/control/reset")
def control_reset():
    simulation_service.reset()
    return {"status": "reset"}


# update simulation speed
@router.post("/control/speed")
def control_speed(req: SpeedRequest):
    simulation_service.set_speed(req.speed)
    return {"status": "ok", "speed": req.speed}


# update order arrival rate
@router.post("/control/arrival-rate")
def control_arrival_rate(req: ArrivalRateRequest):
    simulation_service.set_arrival_rate(req.arrival_rate)
    return {"status": "ok", "arrival_rate": req.arrival_rate}


# generate new warehouse orders
@router.post("/control/generate-orders")
def control_generate_orders(req: GenerateOrdersRequest):
    simulation_service.generate_orders(req.count)
    return {"status": "ok", "generated": req.count}


# return warehouse and RL configuration
@router.get("/config")
def get_config():
    return {
        "num_arms": simulation_service.env.num_arms,
        "num_robots": simulation_service.env.num_robots,
        "num_drones": simulation_service.env.num_drones,
        "num_states": simulation_service.env.n_states,
        "num_actions": NUM_ACTIONS,
        "action_names": ACTION_NAMES,
        "algorithms": ALGORITHM_METADATA,
    }


# start RL training
@router.post("/train")
def start_training(req: TrainRequest):
    if req.algorithm not in ALGORITHM_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm '{req.algorithm}'")
    try:
        training_service.start_training(req.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "started", "algorithm": req.algorithm}


# get current training status
@router.get("/train/status")
def train_status():
    return training_service.get_status()


# get previous training runs
@router.get("/train/history")
def train_history(limit: int = 50):
    runs = db.list_training_runs(limit=limit)
    for r in runs:
        r.pop("reward_curve_json", None)  # keep list endpoint light
    return {"runs": runs}


# get details of one training run
@router.get("/train/run/{run_id}")
def train_run_detail(run_id: int):
    run = db.get_training_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


# activate a saved RL policy
@router.post("/train/activate")
def activate_policy(req: PolicySelectRequest):
    if req.run_id is None:
        raise HTTPException(status_code=400, detail="run_id is required")
    run = db.get_training_run(req.run_id)
    if run is None or not run.get("model_path") or not os.path.exists(run["model_path"]):
        raise HTTPException(status_code=404, detail="No saved policy found for that run")
    policy_table = np.load(run["model_path"])
    label = ALGORITHM_NAMES.get(run["algorithm"], run["algorithm"])
    simulation_service.set_policy(policy_table, label)
    return {"status": "activated", "algorithm": run["algorithm"]}


# compare latest results of all RL algorithms
@router.get("/algorithms/compare")
def algorithms_compare():
    latest_runs = db.latest_run_per_algorithm()
    rows = []
    for algo_key, meta in ALGORITHM_METADATA.items():
        run = latest_runs.get(algo_key)
        row = dict(meta)
        row["algorithm"] = algo_key
        if run:
            row["has_results"] = True
            row["final_avg_reward"] = run["final_avg_reward"]
            row["best_avg_reward"] = run["best_avg_reward"]
            row["duration_seconds"] = round(run["duration_seconds"], 2)
            row["num_episodes"] = run["num_episodes"]
            row["notes"] = run["notes"]
        else:
            row["has_results"] = False
        rows.append(row)
    return {"algorithms": rows}