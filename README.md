# Duckfollow Handoff (for local Codex)

## Goal
Build a Duckietown demo where the user clicks a target location in bird-eye view, the system generates a path, and an existing CasADi MPC controller drives the robot to that target.

## Current State
Implemented:
- `visual_goal_nav.py`: click-goal UI + centerline path extraction.
- `mpc_goal_follower.py`: control loop that loads a `.casadi` controller and steps the simulator.
- `controllers_N2_for_poliduckies.casadi`: vendored controller from `duckrace`.
- `smoke_test_controller.py`: standalone controller load/forward test (no Duckietown env required).
- `compat_check.py`: quick import/version check.

Not yet fully validated end-to-end on local laptop:
- Full `gym-duckietown` runtime with rendering and control loop in one session.

## Why this README exists
A local Codex instance should be able to continue from here without prior chat context.

## Repo Files
- `visual_goal_nav.py`
- `mpc_goal_follower.py`
- `controllers_N2_for_poliduckies.casadi`
- `smoke_test_controller.py`
- `compat_check.py`
- `requirements.txt`

## Environment Notes
You hit this issue on Ubuntu/Python 3.8:
- `gym-duckietown>=6.0.25` was not installable from PyPI.

Fix applied:
- `requirements.txt` now installs Duckietown from source:
  - `git+https://github.com/duckietown/gym-duckietown.git`

Python recommendations:
- Preferred: Python 3.10
- Acceptable fallback: Python 3.8 (may need extra system deps)

## Setup (local machine)
```bash
cd ~/prog/duckfollow
python3 -m venv .env
source .env/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

If `python3` is too old or packages fail, create a 3.10 env via `pyenv` or distro packages.

## Quick Checks
1. Import/dependency check:
```bash
python3 compat_check.py
```

2. Controller-only smoke test:
```bash
python3 smoke_test_controller.py
```
Expected:
- prints controller input/output signature
- prints `dummy_call_ok [...]`

## Notebook/Script run pattern
```python
from gym_duckietown.simulator import Simulator
from mpc_goal_follower import run_visual_goal_mpc

env = Simulator(
    map_name="ETH_large_loop",
    domain_rand=False,
    max_steps=float("inf"),
    frame_rate=30,
)

result = run_visual_goal_mpc(
    env,
    controller_path="./controllers_N2_for_poliduckies.casadi",
    num_points=80,
    max_steps=600,
)

print(len(result["logs"]))
```

## Logging
`mpc_goal_follower.py` logs:
- controller load path
- controller input/output signatures
- named-input vs positional fallback path
- raw action vs clipped action
- step progress (distance/reward/done)

If logs are too verbose, set logger level to `WARNING`.

## Known Risks / Next Work
1. Controller signature mismatch across different `.casadi` exports:
- current code maps known names and falls back to positional inputs.
- if a new controller uses unusual names/shapes, adapt `_prepare_inputs`.

2. `utils.py` dependency:
- code expects `get_position/get_top_view/get_trajectory` functions available.
- if missing, vendor `utils.py` from `duckrace` or package it cleanly.

3. End-to-end validation:
- run one full goal-click -> drive episode and save logs.
- optionally add video recording.

## Minimal command list for local Codex handoff
```bash
pip install -r requirements.txt
python3 smoke_test_controller.py
python3 compat_check.py
# then run notebook/script with Simulator + run_visual_goal_mpc
```
