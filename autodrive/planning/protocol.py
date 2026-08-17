from typing import List, Optional, Protocol, runtime_checkable


@runtime_checkable
class PathPlanner(Protocol):
    def planning(self, start_x: float, start_y: float, goal_x: float, goal_y: float):
        ...

    def smooth_path(self, path, smoothness: float = 0.30):
        ...

    def save_and_show_results(self, path, *args, **kwargs):
        ...
