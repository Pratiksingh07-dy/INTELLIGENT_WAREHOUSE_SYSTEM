<div align="center">

# 🏭 Intelligent Hybrid Aerial-Grounded Warehouse Automation using Reinforcement Learning

**A reinforcement-learning-based warehouse simulation that coordinates robotic arms, ground robots and aerial drones, with a live web dashboard.**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Environment-0F9D58)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Frontend](https://img.shields.io/badge/Frontend-HTML%20%7C%20CSS%20%7C%20JS-orange)
![Tests](https://img.shields.io/badge/Tests-pytest-yellow?logo=pytest&logoColor=white)

</div>

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Problem Statement](#-problem-statement)
3. [Objectives](#-objectives)
4. [Key Features](#-key-features)
5. [System Architecture](#-system-architecture)
6. [Warehouse Model](#-warehouse-model)
7. [Reinforcement Learning Formulation](#-reinforcement-learning-formulation)
8. [RL Algorithms Implemented](#-rl-algorithms-implemented)
9. [Backend API](#-backend-api)
10. [Frontend Dashboard](#-frontend-dashboard)
11. [Project Structure](#-project-structure)
12. [Installation & Setup](#-installation--setup)
13. [Running the Application](#-running-the-application)
14. [Testing](#-testing)
15. [Limitations](#-limitations)
16. [Future Scope](#-future-scope)
17. [Team](#-team)

---

## 📌 Overview

This project is an **Intelligent Hybrid Aerial-Grounded Warehouse Automation** system built as a reinforcement-learning-based simulation. The simulated warehouse contains three types of autonomous resources:

| Resource | Default Count | Primary Responsibility |
| --- | :---: | --- |
| 🦾 Robotic Arms | 2 | Picking / processing orders |
| 🚚 Ground Robots (AGV/AMR) | 2 | Transporting picked orders toward dispatch |
| 🚁 Aerial Drones | 2 | Inventory scanning / monitoring |

The RL agent does **not** control physical robot motors. Instead, it acts as a **high-level scheduler and resource allocator**. It decides which resource should receive work next, which order to prioritize, or whether to take no action.

The project combines a discrete warehouse simulation, a Gymnasium environment, multiple tabular RL algorithms, a FastAPI backend, a database service and a browser-based dashboard, so both the theoretical RL model and a practical interactive system can be demonstrated.

### High-Level Pipeline

```text
Orders + Inventory
        ↓
Warehouse Simulation
        ↓
Gymnasium Environment
        ↓
RL Agent / Policy
        ↓
High-Level Warehouse Action
        ↓
Robotic Arm / Ground Robot / Drone
        ↓
New Warehouse State + Reward
        ↓
Backend + Database
        ↓
Web Dashboard
```

---

## ❓ Problem Statement

A warehouse runs several resource types at the same time. Orders arrive continuously, some have higher priority than others, resources can be busy or idle, and each resource type consumes different amounts of time and energy. Static scheduling rules may not respond well to changing workload conditions.

This project models warehouse resource allocation as a **sequential decision-making problem**. At every simulation step, the RL controller observes a compact representation of the warehouse and selects a high-level action. The resulting state and reward are used to evaluate or improve the policy.

---

## 🎯 Objectives

| Objective | Meaning in the Project |
| --- | --- |
| Resource allocation | Decide which warehouse resource should receive attention next |
| Task scheduling | Manage order and transport queues according to workload and priority |
| Reduced waiting | Penalize long queues and waiting time in the reward |
| Improved throughput | Reward completed orders |
| Resource utilization | Encourage useful activity while considering idle resources |
| Energy awareness | Include resource energy usage in the reward and metrics |
| RL comparison | Provide DP, Monte Carlo, TD, SARSA and Q-Learning methods |
| Visualization | Display live warehouse state, metrics, charts and training results |

---

## ✨ Key Features

- **Heterogeneous resources**: robotic arms, ground robots and drones in one simulation
- **Standard RL interface**: Gymnasium-compatible `WarehouseEnv` with `reset` / `step`
- **8 RL algorithm variants** across Dynamic Programming, Monte Carlo and Temporal Difference families
- **Compact tabular state space** (~648 states) suited to academic experimentation
- **Multi-objective reward** covering throughput, queueing, waiting time, idle resources and energy
- **Live web dashboard** with warehouse floor canvas, resource panels, order queue and analytics charts
- **Training via API**: launch training, view history, and activate saved policies from the browser
- **Synthetic order generation**, so no external dataset or hardware is required
- **Modular, layered architecture** separating simulation, RL, backend, database and frontend
- **Automated pytest suite** covering the simulation and RL layers

---

## 🏗 System Architecture

```text
                    ┌──────────────────────────────┐
                    │       Browser Dashboard      │
                    │   HTML + CSS + JavaScript    │
                    └──────────────┬───────────────┘
                                   │ HTTP / JSON
                    ┌──────────────▼───────────────┐
                    │        FastAPI Backend       │
                    │  routes + schemas + services │
                    └──────────────┬───────────────┘
                                   │
             ┌─────────────────────┼─────────────────────┐
             │                     │                     │
     ┌───────▼────────┐   ┌────────▼────────┐   ┌────────▼─────┐
     │  Simulation    │   │    Training     │   │   Database   │
     │   Service      │   │    Service      │   │   Service    │
     └───────┬────────┘   └────────┬────────┘   └──────────────┘
             │                     │
             ▼                     ▼
     ┌──────────────────────────────────────────┐
     │          Gymnasium WarehouseEnv          │
     └──────────────────┬───────────────────────┘
                        │
                        ▼
     ┌──────────────────────────────────────────┐
     │           Warehouse Simulation           │
     │ orders + inventory + resources + reward  │
     └───────┬───────────────┬──────────────────┘
             │               │
       ┌─────▼─────┐   ┌─────▼────────────────┐
       │ Resources │   │ Orders / Inventory   │
       │ Arm/AGV/  │   │ queue / stock / stage│
       │ Drone     │   │                      │
       └───────────┘   └──────────────────────┘
```

The architecture is intentionally layered. The simulation is independent of the web interface, the Gymnasium environment exposes it in a standard RL form, the backend provides web-accessible control and training, and the frontend visualizes the state.

**Dependency direction**

```text
simulation → rl.environment → RL algorithms / experiments
simulation + rl.environment → backend services → FastAPI routes → frontend
```

---

## 🏭 Warehouse Model

### Order Flow

```text
Order generated → Pending order queue → Robotic arm assigned → Picking / processing
      → Transport queue → Ground robot assigned → Transport / dispatch → Order completed
```

### Resources

| Module | Description |
| --- | --- |
| `simulation/robotic_arm.py` | Picking and processing. Tracks status (IDLE / PROCESSING), assigned task, utilization, energy and completed tasks |
| `simulation/ground_robot.py` | Transport. Tracks location, destination, battery, travel progress, energy and utilization |
| `simulation/drone.py` | Inventory scanning. Tracks zones, status, battery, energy, utilization and completed scans |
| `simulation/orders.py` | Order model (ID, product, quantity, priority, stage, timing) and the synthetic `OrderGenerator` |
| `simulation/inventory.py` | Warehouse stock and product state |
| `simulation/warehouse.py` | Main integration layer: queues, simulation time, actions, rewards, metrics, and `to_dict()` serialization for the backend |

---

## 🧠 Reinforcement Learning Formulation

The problem is formulated as a **Markov Decision Process** `(S, A, P, R, γ)`.

| MDP Element | Project Meaning |
| --- | --- |
| **S**: State | Discretized warehouse condition: queues, busy resources, waiting time and priority |
| **A**: Action | Five high-level scheduling actions |
| **P**: Transition | Determined by the warehouse simulation dynamics |
| **R**: Reward | Combines completion, assignment, queue, waiting, idle-resource and energy effects |
| **γ**: Discount factor | Controls the importance of future rewards |

### State Space (~648 discretized states)

| State Component | Buckets / Values |
| --- | --- |
| Pending orders | 0, 1, 2, 3+ |
| Busy robotic arms | 0 to 2 |
| Busy ground robots | 0 to 2 |
| Busy drones | 0 to 2 |
| Average waiting time | Low / Medium / High |
| High-priority order waiting | Yes / No |

### Action Space

| ID | Action | Meaning |
| :---: | --- | --- |
| 0 | `ASSIGN_ARM` | Assign the next suitable queued order to an idle robotic arm |
| 1 | `ASSIGN_ROBOT` | Assign a picked order to an idle ground robot |
| 2 | `ASSIGN_DRONE` | Send an idle drone to scan a warehouse zone |
| 3 | `PRIORITIZE` | Reorder the queue to prioritize an important waiting order |
| 4 | `NOOP` | Take no scheduling action for this decision |

### Reward Function

| Component | Effect |
| --- | --- |
| Order completion | **+10** per completion |
| Successful assignment | **+2** |
| Invalid action | **−3** |
| Queue length | **−0.4** × queue-related term |
| Average waiting | **−0.05** × waiting-related term |
| Idle resources while work exists | **−0.3** |
| Energy | **−0.02** × step energy |

### Gymnasium Environment

`rl/environment/warehouse_env.py` bridges the simulation and the RL algorithms:

```text
Observation/state → WarehouseEnv → RL algorithm selects action → env.step(action)
      → Warehouse simulation advances → next state + reward + termination info
```

---

## 🤖 RL Algorithms Implemented

| Algorithm | Family | Role in the Project |
| --- | --- | --- |
| Policy Iteration | Dynamic Programming | Alternates policy evaluation and improvement using an estimated model |
| Value Iteration | Dynamic Programming | Bellman optimality updates to estimate optimal values and derive a policy |
| First-Visit Monte Carlo | Monte Carlo | Learns from returns after the first visit to a state/action in an episode |
| Every-Visit Monte Carlo | Monte Carlo | Uses returns from every occurrence of a state/action within episodes |
| TD(0) | Temporal Difference | One-step bootstrapping from the next state |
| TD(λ) | Temporal Difference | Eligibility traces spread learning over multiple steps |
| SARSA | TD Control | On-policy control using the selected next action |
| Q-Learning | TD Control | Off-policy control toward the best estimated next action |

**Notes**

- **Dynamic Programming** methods estimate a transition/reward model by sampling from the warehouse environment, then apply DP.
- **Monte Carlo** methods learn from complete episodes and use epsilon-greedy exploration for control.
- **SARSA vs Q-Learning**: SARSA is on-policy (learns from the actions the current policy takes), while Q-Learning is off-policy (updates toward the best estimated next action regardless of exploration).

### Experiments & Evaluation

The `experiments/` directory holds the training and evaluation entry points (`train.py`, `evaluate.py`). Training can also be triggered from the backend API. Algorithm comparison uses **recorded training runs** rather than manually entered values.

---

## 🔌 Backend API

Built with **FastAPI** and served through **Uvicorn**. `backend/main.py` creates the app, initializes the database on startup, configures CORS, mounts the frontend assets, serves `frontend/index.html` at the root URL, and exposes a `/health` endpoint.

| Endpoint | Purpose |
| --- | --- |
| `/api/state` | Current warehouse and dashboard state |
| `/api/control/start` | Start live simulation |
| `/api/control/pause` | Pause live simulation |
| `/api/control/reset` | Reset live simulation |
| `/api/control/speed` | Change simulation speed |
| `/api/control/arrival-rate` | Change order arrival rate |
| `/api/control/generate-orders` | Generate a requested number of orders |
| `/api/config` | Configuration and available algorithms |
| `/api/train` | Start RL training |
| `/api/train/status` | Training status / progress |
| `/api/train/history` | Training history |
| `/api/train/run/{run_id}` | Details of a specific training run |
| `/api/train/activate` | Activate a saved policy for live simulation |
| `/api/algorithms/compare` | Algorithm metadata and recorded results |
| `/health` | Health check |

**Service layer**

| Service | Responsibility |
| --- | --- |
| `simulation_service.py` | Live-control bridge between the API and `WarehouseEnv`: start/pause, speed, arrival rate, order generation, history, active policy |
| `training_service.py` | Connects the training API to the RL algorithms |
| `db.py` | Stores and retrieves training runs for history and comparison |

Request validation is handled by Pydantic models in `backend/models/schemas.py`.

---

## 🖥 Frontend Dashboard

A browser-based dashboard built with **HTML, CSS and vanilla JavaScript**, served by FastAPI so the whole system runs from a single local URL.

- Dashboard, training and algorithm-comparison views
- Simulation controls and status indicators
- Warehouse floor canvas
- Robotic arm, ground robot and drone panels
- Order queue table
- RL analytics charts (reward, queue length, waiting time, throughput, energy, utilization)
- Training controls and training history
- Algorithm comparison table
- MDP formalization summary

| File | Role |
| --- | --- |
| `frontend/index.html` | Dashboard layout and views |
| `frontend/css/style.css` | Layout, panels, tables, charts and progress indicators |
| `frontend/js/app.js` | Main controller: API calls, state polling, training controls, policy activation |
| `frontend/js/charts.js` | RL performance charts |
| `frontend/js/visualization.js` | Warehouse floor canvas rendering |

---

## 📂 Project Structure

```text
INTELLIGENT_WAREHOUSE_HOUSE/
│
├── backend/
│   ├── api/
│   │   └── routes.py
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   ├── db.py
│   │   ├── simulation_service.py
│   │   └── training_service.py
│   └── main.py
│
├── database/
│
├── experiments/
│   ├── evaluate.py
│   └── train.py
│
├── frontend/
│   ├── assets/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── app.js
│   │   ├── charts.js
│   │   └── visualization.js
│   └── index.html
│
├── rl/
│   ├── dp/
│   │   ├── policy_iteration.py
│   │   └── value_iteration.py
│   ├── environment/
│   │   └── warehouse_env.py
│   ├── monte_carlo/
│   │   ├── every_visit_mc.py
│   │   └── first_visit_mc.py
│   ├── td/
│   │   ├── td_zero.py
│   │   └── td_lambda.py
│   ├── q_learning.py
│   ├── sarsa.py
│   └── utils.py
│
├── simulation/
│   ├── drone.py
│   ├── ground_robot.py
│   ├── inventory.py
│   ├── orders.py
│   ├── robotic_arm.py
│   └── warehouse.py
│
├── tests/
│   ├── test_arms.py
│   ├── test_drones.py
│   ├── test_environment.py
│   ├── test_ground_robots.py
│   ├── test_orders.py
│   ├── test_td_dp_mc.py
│   └── test_warehouse.py
│
├── .gitignore
├── .gitattributes
├── README.md
├── requirements.txt
└── run.py
```

---

## ⚙️ Installation & Setup

### Prerequisites

- **Python 3.10** (developed and tested on 3.10.11)
- `pip` and `venv`

### Steps

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd INTELLIGENT_WAREHOUSE_HOUSE

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
python run.py
```

Then open **http://127.0.0.1:8000** in your browser. The frontend is served by FastAPI, so the dashboard and API share the same address.

### What Happens During a Live Run

1. FastAPI starts and initializes the database.
2. The simulation service creates and resets `WarehouseEnv`.
3. The browser loads the dashboard and begins polling `/api/state`.
4. When you click **Start**, the simulation service picks actions using the active policy (or the default rule-based heuristic).
5. The warehouse advances resources, updates orders and metrics, and the dashboard redraws.
6. You can launch **RL training** from the training interface. Results are stored and shown in history.
7. A saved policy can be **activated** and used for the live simulation.

### End-to-End Data Flow

```text
Order generated → queued → state encoded → policy selects action → warehouse applies action
     → resources advance one step → orders progress → reward + metrics computed
     → WarehouseEnv returns next state + reward → SimulationService records history
     → FastAPI returns JSON → dashboard updates
```

---

## 🧪 Testing

The project uses **pytest**.

```bash
python -m pytest -v
```

| Test File | Area Covered |
| --- | --- |
| `test_arms.py` | Robotic arm initialization, assignment, progression, utilization, energy |
| `test_drones.py` | Drone behavior and scanning-related state |
| `test_ground_robots.py` | Ground robot task, movement and resource behavior |
| `test_orders.py` | Order generation and properties |
| `test_warehouse.py` | Warehouse integration, queues and simulation behavior |
| `test_environment.py` | Gymnasium `WarehouseEnv` behavior |
| `test_td_dp_mc.py` | MDP estimation, DP, Monte Carlo and TD prediction |

---

## ⚠️ Limitations

This is a **simulation and academic demonstration**, not a production warehouse-control system.

- Warehouse dynamics are simplified compared with a real industrial facility
- Robot movement and collision behavior are abstracted
- The discretized state limits resolution
- Resource counts are small and intended for simulation
- Synthetic orders do not capture every real-world warehouse pattern
- The action space is high-level, not low-level robot control
- Real deployment would need sensors, localization, path planning, safety systems and hardware validation

---

## 🚀 Future Scope

- Multi-agent reinforcement learning
- Richer spatial state representations
- Realistic path planning
- Digital-twin simulation
- Training and evaluation on real warehouse datasets

---

## 👥 Team

| Name | Role |
| --- | --- |
| **Pratik Singh** | Co-developer |
| **Suraj Venkataraman** | Co-developer |

**Subject:** Reinforcement Learning  
**Project Type:** Web-based Warehouse Simulation

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>