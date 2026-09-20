from importlib import import_module

MODEL_REGISTRY = {
    "TSE_BSRNN_SPK":
    "wesep.models.tse_bsrnn_spk:TSE_BSRNN_SPK",
    "TSE_BSRNN_SPK_SPATIAL":
    ("wesep.models.tse_bsrnn_spk_spatial:TSE_BSRNN_SPK_SPATIAL"),
    "TSE_BSRNN_VISUAL":
    "wesep.models.tse_bsrnn_visual:TSE_BSRNN_VISUAL",
    "TSE_BSRNN_TEXTUAL": ("wesep.models.tse_bsrnn_textual:TSE_BSRNN_TEXTUAL"),
    "TSE_BSRNN_SPATIAL":
    "wesep.models.tse_bsrnn_spatial:TSE_BSRNN_SPATIAL",
    "TSE_NBC2_SPATIAL":
    "wesep.models.tse_nbc2_spatial:TSE_NBC2_SPATIAL",
    "TSE_DPCCN_SPK":
    "wesep.models.tse_dpccn_spk:TSE_DPCCN_SPK",
    "TSE_TFGRIDNET_SPK": ("wesep.models.tse_tfgridnet_spk:TSE_TFGRIDNET_SPK"),
    "TSE_TFGRIDNET_VISUAL":
    ("wesep.models.tse_tfgridnet_visual:TSE_TFGRIDNET_VISUAL"),
    "TSE_SPEX_PLUS":
    "wesep.models.tse_spex_plus:TSE_SPEX_PLUS",
}


def get_model(model_name: str):
    """Load and return one model class without importing other models."""
    target = MODEL_REGISTRY.get(model_name, model_name)
    if ":" not in target:
        raise KeyError(f"Unknown model: {model_name}")

    module_name, class_name = target.rsplit(":", 1)
    module = import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Model class '{class_name}' not found in '{module_name}'"
        ) from exc
