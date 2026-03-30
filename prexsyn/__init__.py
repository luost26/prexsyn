__version__ = "1.0.0"

try:
    import prexsyn_engine  # noqa: F401
except ImportError as e:
    raise ImportError(
        "PrexSyn Engine is not installed."
        "Please refer to the documentation for installation instructions: "
        "https://prexsyn.readthedocs.io/"
    ) from e
