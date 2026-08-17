"""Smoke tests for the public CLI and compatibility imports."""

import importlib

import pytest


def test_main_import_is_safe_and_run_session_is_public():
    main_module = importlib.import_module("main")
    session_module = importlib.import_module("autodrive.simulation.session")

    assert callable(main_module.main)
    assert callable(session_module.run_session)


@pytest.mark.parametrize(
    "module_name",
    [
        "environment",
        "vehicle_model",
        "font_support",
        "rrt_path_planning",
        "astar_path_planning",
        "rrt_star_path_planning",
        "pure_pursuit_controller",
        "mpc_controller",
        "stanley_controller",
    ],
)
def test_root_compatibility_shim_imports(module_name):
    assert importlib.import_module(module_name) is not None
