# Duckfollow

Visual-goal path planning scaffold.

## Quick usage
```python
from visual_goal_nav import VisualGoalNavigator, run_path_with_callback

nav = VisualGoalNavigator(env)
nav.show_and_pick_goal()          # click goal in bird-eye view
path = nav.plan_to_goal(80)
nav.plot_plan(path)

# connect to your controller loop
def step_callback(target_xy, i):
    pass

run_path_with_callback(path, step_callback)
```

## Note
`visual_goal_nav.py` expects `utils.py` functions (`get_top_view`, `get_trajectory`, `get_position`) available in your environment.
