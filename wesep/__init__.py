def load_model(*args, **kwargs):
    """Load the CLI extractor only when the public helper is called."""
    from wesep.cli.extractor import load_model as _load_model
    return _load_model(*args, **kwargs)


def load_model_local(*args, **kwargs):
    """Load a local CLI extractor without importing CLI dependencies early."""
    from wesep.cli.extractor import load_model_local as _load_model_local
    return _load_model_local(*args, **kwargs)
