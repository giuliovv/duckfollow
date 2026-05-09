"""Visual-goal path follower using a precompiled CasADi MPC controller.

This module wires:
- visual goal selection/path planning from `visual_goal_nav.py`
- control from a `.casadi` controller exported in duckrace
- simulator stepping loop in gym-duckietown

The controller function is expected to output at least one value that can be
interpreted as action [velocity, steering]. Input signature can vary; we map by
argument names when possible and fall back to positional conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from visual_goal_nav import VisualGoalNavigator
from utils import get_position

try:
    import casadi as ca
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "casadi is required. Install with: pip install casadi"
    ) from exc


@dataclass
class MPCConfig:
    controller_path: str
    max_steps: int = 600
    goal_tolerance_m: float = 0.08
    action_min: Tuple[float, float] = (0.0, -1.0)
    action_max: Tuple[float, float] = (1.0, 1.0)
    debug: bool = True


class CasadiMPCFollower:
    def __init__(self, env, cfg: MPCConfig):
        self.env = env
        self.cfg = cfg
        self.controller = ca.Function.load(cfg.controller_path)

        self._in_names = [self.controller.name_in(i) for i in range(self.controller.n_in())]
        self._out_names = [self.controller.name_out(i) for i in range(self.controller.n_out())]

    def _build_state(self) -> np.ndarray:
        p = get_position(self.env)
        return np.array([p.x, p.y, p.theta], dtype=np.float32)

    def _prepare_inputs(self, state: np.ndarray, target_xy: np.ndarray) -> Dict[str, ca.DM]:
        inputs: Dict[str, ca.DM] = {}
        for i, name in enumerate(self._in_names):
            lname = name.lower()
            size = self.controller.size_in(i)
            n = int(size[0] * size[1])

            if any(k in lname for k in ["x", "state", "pose"]):
                val = np.zeros((n,), dtype=np.float32)
                val[: min(n, len(state))] = state[: min(n, len(state))]
                inputs[name] = ca.DM(val)
            elif any(k in lname for k in ["ref", "goal", "target", "traj"]):
                val = np.zeros((n,), dtype=np.float32)
                flat_t = np.asarray(target_xy, dtype=np.float32).reshape(-1)
                val[: min(n, len(flat_t))] = flat_t[: min(n, len(flat_t))]
                inputs[name] = ca.DM(val)
            elif any(k in lname for k in ["u", "action"]):
                inputs[name] = ca.DM.zeros(n, 1)
            else:
                inputs[name] = ca.DM.zeros(n, 1)

        return inputs

    def _extract_action(self, out) -> np.ndarray:
        if isinstance(out, dict):
            candidates: List[np.ndarray] = []
            for key in ("u", "action", "u0", "cmd", "control"):
                if key in out:
                    candidates.append(np.array(out[key]).reshape(-1))
            if not candidates:
                candidates = [np.array(v).reshape(-1) for v in out.values()]
            raw = max(candidates, key=lambda a: a.size)
        elif isinstance(out, (tuple, list)):
            raw = np.array(out[0]).reshape(-1)
        else:
            raw = np.array(out).reshape(-1)

        if raw.size == 0:
            action = np.array([0.0, 0.0], dtype=np.float32)
        elif raw.size == 1:
            action = np.array([float(raw[0]), 0.0], dtype=np.float32)
        else:
            action = np.array([float(raw[0]), float(raw[1])], dtype=np.float32)

        action = np.clip(action, np.array(self.cfg.action_min), np.array(self.cfg.action_max))
        return action

    def step_to_target(self, target_xy: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        state = self._build_state()
        ins = self._prepare_inputs(state, target_xy)

        try:
            out = self.controller.call(ins)
        except Exception:
            positional = []
            for i in range(self.controller.n_in()):
                name = self._in_names[i]
                if any(k in name.lower() for k in ["x", "state", "pose"]):
                    positional.append(ca.DM(state))
                elif any(k in name.lower() for k in ["ref", "goal", "target", "traj"]):
                    positional.append(ca.DM(np.asarray(target_xy).reshape(-1)))
                else:
                    positional.append(ca.DM.zeros(*self.controller.size_in(i)))
            out = self.controller(*positional)

        action = self._extract_action(out)
        obs, reward, done, info = self.env.step(action.tolist())
        return obs, reward, done, info

    def run_path(self, path_world: np.ndarray) -> List[dict]:
        logs: List[dict] = []
        for i, target in enumerate(path_world):
            if i >= self.cfg.max_steps:
                break

            pos = get_position(self.env)
            d = float(np.linalg.norm(np.array([pos.x, pos.y]) - np.array(target[:2])))
            if d <= self.cfg.goal_tolerance_m:
                continue

            _, reward, done, info = self.step_to_target(target[:2])
            logs.append({"i": i, "target": target[:2].tolist(), "dist": d, "reward": float(reward)})

            if self.cfg.debug and (i % 20 == 0):
                print(f"step={i:04d} dist={d:.3f}")

            if done:
                break

        return logs


def run_visual_goal_mpc(env, controller_path: str, num_points: int = 80, max_steps: int = 600):
    controller_path = str(Path(controller_path).expanduser().resolve())
    nav = VisualGoalNavigator(env)
    nav.show_and_pick_goal()
    path = nav.plan_to_goal(num_points=num_points)
    nav.plot_plan(path)

    follower = CasadiMPCFollower(
        env,
        MPCConfig(controller_path=controller_path, max_steps=max_steps),
    )
    logs = follower.run_path(path)
    return {"path": path, "logs": logs, "controller": controller_path}
