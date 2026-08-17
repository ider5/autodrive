from typing import Protocol, runtime_checkable


@runtime_checkable
class Controller(Protocol):
    target_speed: float
    env: object

    def set_path(self, path) -> None: ...

    def set_target_speed(self, speed) -> None: ...

    def calculate_steering(self, vehicle, path, road_width=None): ...
