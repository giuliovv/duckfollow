"""Visual-goal path follower using a precompiled CasADi MPC controller."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from visual_goal_nav import VisualGoalNavigator
from utils import get_position

try:
    import casadi as ca
except ImportError as exc:  # pragma: no cover
    raise ImportError("casadi is required. Install with: pip install casadi") from exc

LOG = logging.getLogger("duckfollow.mpc")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


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
        LOG.info("Loaded controller %s", cfg.controller_path)
        LOG.info("Controller inputs: %s", [(n, self.controller.size_in(i)) for i, n in enumerate(self._in_names)])
        LOG.info("Controller outputs: %s", [(n, self.controller.size_out(i)) for i, n in enumerate(self._out_names)])

    def _build_state(self) -> np.ndarray:
        p = get_position(self.env)
        s = np.array([p.x, p.y, p.theta], dtype=np.float32)
        return s

    def _prepare_inputs(self, state: np.ndarray, target_xy: np.ndarray) -> Dict[str, ca.DM]:
        inputs: Dict[str, ca.DM] = {}
        for i, name in enumerate(self._in_names):
            lname = name.lower()
            r, c = self.controller.size_in(i)
            n = int(r * c)
            if any(k in lname for k in ["x", "state", "pose"]):
                val = np.zeros((n,), dtype=np.float32)
                val[: min(n, len(state))] = state[: min(n, len(state))]
                inputs[name] = ca.DM(val)
            elif any(k in lname for k in ["ref", "goal", "target", "traj"]):
                val = np.zeros((n,), dtype=np.float32)
                flat_t = np.asarray(target_xy, dtype=np.float32).reshape(-1)
                val[: min(n, len(flat_t))] = flat_t[: min(n, len(flat_t))]
                inputs[name] = ca.DM(val)
            else:
                inputs[name] = ca.DM.zeros(r, c)
        return inputs

    def _extract_action(self, out) -> np.ndarray:
        if isinstance(out, dict):
            arrs = [np.array(v).reshape(-1) for v in out.values()]
            raw = max(arrs, key=lambda a: a.size)
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

        clipped = np.clip(action, np.array(self.cfg.action_min), np.array(self.cfg.action_max))
        if self.cfg.debug:
            LOG.info("raw_action=%s clipped_action=%s", action.tolist(), clipped.tolist())
        return clipped

    def step_to_target(self, target_xy: np.ndarray):
        state = self._build_state()
        ins = self._prepare_inputs(state, target_xy)

        try:
            out = self.controller.call(ins)
            if self.cfg.debug:
                LOG.info("controller call: named-input path")
        except Exception as e:
            LOG.warning("named-input call failed (%s); retrying positional", e)
            positional = []
            for i in range(self.controller.n_in()):
                name = self._in_names[i].lower()
                if any(k in name for k in ["x", "state", "pose"]):
                    positional.append(ca.DM(state))
                elif any(k in name for k in ["ref", "goal", "target", "traj"]):
                    positional.append(ca.DM(np.asarray(target_xy).reshape(-1)))
                else:
                    positional.append(ca.DM.zeros(*self.controller.size_in(i)))
            out = self.controller(*positional)
            if self.cfg.debug:
                LOG.info("controller call: positional fallback path")

        action = self._extract_action(out)
        obs, reward, done, info = self.env.step(action.tolist())
        return obs, reward, done, info, state, action

    def run_path(self, path_world: np.ndarray) -> List[dict]:
        logs: List[dict] = []
        LOG.info("Starting run_path with %d waypoints", len(path_world))
        for i, target in enumerate(path_world):
            if i >= self.cfg.max_steps:
                LOG.info("Stopping at max_steps=%d", self.cfg.max_steps)
                break

            pos = get_position(self.env)
            d = float(np.linalg.norm(np.array([pos.x, pos.y]) - np.array(target[:2])))
            if d <= self.cfg.goal_tolerance_m:
                if self.cfg.debug:
                    LOG.info("step=%d target already reached dist=%.4f", i, d)
                continue

            _, reward, done, info, state, action = self.step_to_target(target[:2])
            row = {
                "i": i,
                "state": state.tolist(),
                "target": target[:2].tolist(),
                "action": action.tolist(),
                "dist": d,
                "reward": float(reward),
                "done": bool(done),
            }
            logs.append(row)

            if self.cfg.debug and (i % 10 == 0):
                LOG.info("step=%04d dist=%.3f reward=%.4f done=%s", i, d, float(reward), done)

            if done:
                LOG.warning("Environment returned done=True at step=%d", i)
                break

        LOG.info("Run complete, logged %d control steps", len(logs))
        return logs


def run_visual_goal_mpc(env, controller_path: str, num_points: int = 80, max_steps: int = 600):
    controller_path = str(Path(controller_path).expanduser().resolve())
    nav = VisualGoalNavigator(env)
    goal = nav.show_and_pick_goal()
    LOG.info("Goal selected world=(%.3f, %.3f)", goal.world_x, goal.world_y)
    path = nav.plan_to_goal(num_points=num_points)
    LOG.info("Planned %d waypoints", len(path))
    nav.plot_plan(path)

    follower = CasadiMPCFollower(env, MPCConfig(controller_path=controller_path, max_steps=max_steps))
    logs = follower.run_path(path)
    return {"path": path, "logs": logs, "controller": controller_path}
