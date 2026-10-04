"""Smoke test: the src-layout package installs and imports cleanly."""


def test_package_imports():
    import data_sourcing_agent

    assert data_sourcing_agent.__version__
