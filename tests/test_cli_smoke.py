"""Smoke tests for the public CLI and package imports."""

import importlib


def test_main_import_is_safe_and_run_session_is_public():
    main_module = importlib.import_module("main")
    session_module = importlib.import_module("autodrive.simulation.session")

    assert callable(main_module.main)
    assert callable(session_module.run_session)
