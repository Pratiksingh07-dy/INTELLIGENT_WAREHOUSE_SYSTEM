from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# define training request parameters
class TrainRequest(BaseModel):
    algorithm: str = Field(..., description=(
        "One of: policy_iteration, value_iteration, first_visit_mc, "
        "every_visit_mc, td_zero, td_lambda, sarsa, q_learning"))
    num_episodes: int = Field(1000, ge=10, le=20000)
    alpha: float = Field(0.1, gt=0, le=1.0, description="Learning rate")
    gamma: float = Field(0.95, gt=0, le=0.999, description="Discount factor")
    epsilon_start: float = Field(1.0, ge=0, le=1.0)
    epsilon_min: float = Field(0.05, ge=0, le=1.0)
    epsilon_decay: float = Field(0.998, gt=0, le=1.0)
    lam: float = Field(0.8, ge=0, le=1.0, description="Lambda for TD(lambda)")
    dp_samples_per_pair: int = Field(6, ge=1, le=50,
                                      description="Monte-Carlo samples used to "
                                                   "estimate the MDP model for DP methods")
    max_steps: int = Field(60, ge=10, le=500, description="Steps per training episode")
    seed: int = Field(1, ge=0)


# define simulation speed request
class SpeedRequest(BaseModel):
    speed: float = Field(1.0, gt=0, le=20.0, description="Simulation steps per second")


# define order arrival rate request
class ArrivalRateRequest(BaseModel):
    arrival_rate: float = Field(0.45, ge=0.0, le=5.0)


# define order generation request
class GenerateOrdersRequest(BaseModel):
    count: int = Field(3, ge=1, le=50)


# define policy selection request
class PolicySelectRequest(BaseModel):
    run_id: Optional[int] = Field(None, description="Training run id to load as the active policy")