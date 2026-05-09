"""Visual goal -> path planning helper for Duckfollow.

Click a goal in bird-eye space and generate a path there using
existing DuckRace utility functions (top-view + centerline extraction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

# NOTE: expects `utils.py` from duckrace-style stack to be available in PYTHONPATH.
# You can vendor/copy it into this repo, or install as package in your environment.
from utils import get_position, get_top_view, get_trajectory


@dataclass
class GoalSelection:
    pixel_x: float
    pixel_y: float
    world_x: float
    world_y: float


class VisualGoalNavigator:
    def __init__(self, env, centerline_samples: int = 600):
        self.env = env
        self.centerline_samples = centerline_samples
        self.top_view = np.flip(get_top_view(env), [0])
        self.centerline_world = get_trajectory(env, samples=centerline_samples, scaled=True, method="distance")
        self.goal: Optional[GoalSelection] = None

        self._world_w = env.grid_width * env.road_tile_size
        self._world_h = env.grid_height * env.road_tile_size

    def pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        h, w = self.top_view.shape[:2]
        x = px * self._world_w / w
        y = py * self._world_h / h
        return float(x), float(y)

    def world_to_pixel(self, x, y):
        h, w = self.top_view.shape[:2]
        x = np.asarray(x)
        y = np.asarray(y)
        px = x * w / self._world_w
        py = y * h / self._world_h
        return px, py

    def show_and_pick_goal(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.top_view, origin="lower")

        cx, cy = self.centerline_world[:, 0], self.centerline_world[:, 1]
        cpx, cpy = self.world_to_pixel(cx, cy)
        ax.plot(cpx, cpy, "y-", lw=1.0, alpha=0.7, label="centerline")

        ax.set_title("Click target point in bird-eye view")
        ax.legend(loc="upper right")

        picked = {"ok": False}

        def _onclick(event):
            if event.inaxes is None:
                return
            wx, wy = self.pixel_to_world(event.xdata, event.ydata)
            self.goal = GoalSelection(event.xdata, event.ydata, wx, wy)
            picked["ok"] = True
            ax.scatter([event.xdata], [event.ydata], c="r", s=70, marker="x")
            fig.canvas.draw_idle()

        cid = fig.canvas.mpl_connect("button_press_event", _onclick)
        plt.show()
        fig.canvas.mpl_disconnect(cid)

        if not picked["ok"]:
            raise RuntimeError("No goal selected. Click on the map first.")

        return self.goal

    def _closest_centerline_index(self, x: float, y: float) -> int:
        pts = self.centerline_world[:, :2]
        d2 = np.sum((pts - np.array([x, y])) ** 2, axis=1)
        return int(np.argmin(d2))

    def plan_to_goal(self, num_points: int = 80) -> np.ndarray:
        if self.goal is None:
            raise RuntimeError("No goal set. Call show_and_pick_goal() first.")

        pos = get_position(self.env)
        start_idx = self._closest_centerline_index(pos.x, pos.y)
        goal_idx = self._closest_centerline_index(self.goal.world_x, self.goal.world_y)

        n = len(self.centerline_world)
        if goal_idx >= start_idx:
            forward = np.arange(start_idx, goal_idx + 1)
        else:
            forward = np.concatenate([np.arange(start_idx, n), np.arange(0, goal_idx + 1)])

        if start_idx >= goal_idx:
            backward = np.arange(start_idx, goal_idx - 1, -1)
        else:
            backward = np.concatenate([np.arange(start_idx, -1, -1), np.arange(n - 1, goal_idx - 1, -1)])

        idxs = forward if len(forward) <= len(backward) else backward
        path = self.centerline_world[idxs][:, :2]

        if len(path) >= 2 and num_points > 1:
            t_old = np.linspace(0.0, 1.0, len(path))
            t_new = np.linspace(0.0, 1.0, num_points)
            x_new = np.interp(t_new, t_old, path[:, 0])
            y_new = np.interp(t_new, t_old, path[:, 1])
            path = np.stack([x_new, y_new], axis=1)

        return path

    def plot_plan(self, path_world: np.ndarray):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.top_view, origin="lower")

        cx, cy = self.centerline_world[:, 0], self.centerline_world[:, 1]
        cpx, cpy = self.world_to_pixel(cx, cy)
        ax.plot(cpx, cpy, "y-", lw=1.0, alpha=0.5, label="centerline")

        ppx, ppy = self.world_to_pixel(path_world[:, 0], path_world[:, 1])
        ax.plot(ppx, ppy, "c-", lw=2.0, label="planned path")

        pos = get_position(self.env)
        spx, spy = self.world_to_pixel(pos.x, pos.y)
        ax.scatter([spx], [spy], c="lime", s=60, label="start")

        if self.goal is not None:
            ax.scatter([self.goal.pixel_x], [self.goal.pixel_y], c="red", marker="x", s=80, label="goal")

        ax.set_title("Visual Goal Path Plan")
        ax.legend(loc="upper right")
        plt.show()


def run_path_with_callback(path_world: np.ndarray, step_callback: Callable[[np.ndarray, int], None]):
    for i, target in enumerate(path_world):
        step_callback(target, i)
