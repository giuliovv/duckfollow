# Duckfollow

Visual-goal navigation + CasADi MPC control loop for Duckietown.

## What is included
- `visual_goal_nav.py`: click a target in bird-eye map and build a centerline path.
- `mpc_goal_follower.py`: run path tracking with an existing `.casadi` MPC controller.
- `compat_check.py`: quick Python/package compatibility check.

## Install
```bash
pip install -r requirements.txt
```

## Compatibility check
```bash
python3 compat_check.py
```

## Run from a notebook
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
    controller_path="/path/to/N2_for_poliduckies.casadi",
    num_points=80,
    max_steps=600,
)

print("steps logged:", len(result["logs"]))
```

## Notes
- Controller input/output signatures differ across exported `.casadi` files; this wrapper maps known names (`state/x/pose`, `goal/ref/target`) and falls back to positional calling.
- If you use duckrace utilities, ensure `utils.py` is importable in your environment.
