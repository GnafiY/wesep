import json
import logging
from pathlib import Path

import soundfile
import torch
import yaml

from wesep.cli.input_reader import DirectInputReader
from wesep.cli.utils import get_args
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.utils import set_seed

CUE_KEYS = {
    "audio": "audio_aux",
    "spatial": "spatial_aux",
    "visual": "visual_aux",
    "textual": "textual_aux",
}

CLI_CUE_DEFAULTS = {
    "audio_waveform": {
        "shape": [1, 1],
        "fill_value": 0.0
    },  # [C, T]
    "audio_embedding": {
        "shape": [192],
        "fill_value": 0.0
    },  # [D]
    "spatial": {
        "shape": [2],
        "fill_value": 0.0
    },  # [F] or [F, T]
    "visual_raw_video": {
        "shape": [112, 112, 3, 1],
        "fill_value": 0.0,
    },  # [H, W, C, T]
    "visual_feature": {
        "shape": [512, 1],
        "fill_value": 0.0,
    },  # [D, T]
    "textual": {
        "shape": [1],
        "fill_value": 0
    },  # [T]
}


class Extractor:
    """Run one target-level extraction request from tensors or local files."""

    def __init__(self, model_dir, config_path=None, checkpoint_path=None):
        """Load one local model package and its resolved inference config."""
        set_seed()
        model_dir = Path(model_dir)
        config_path = Path(config_path or model_dir / "config.yaml")
        checkpoint_path = Path(checkpoint_path or model_dir / "avg_model.pt")
        with config_path.open("r", encoding="utf-8") as fin:
            self.configs = yaml.safe_load(fin)
        model_name = self.configs["model"]["tse_model"]
        model_config = self.configs["model_args"]["tse_model"]
        self.model = get_model(model_name)(
            model_config,
            defer_pretrained=True,
        )
        load_pretrained_model(self.model, str(checkpoint_path))
        self.model.eval()

        self.cues = (self.configs.get("dataset_args", {}).get("cues", {})
                     or {})
        self.resample_rate = self.configs.get("dataset_args",
                                              {}).get("resample_rate", 16000)
        self.input_reader = DirectInputReader(
            self.resample_rate,
            model_config,
            config_path,
            model_dir,
        )
        self.device = torch.device("cpu")
        self.output_norm = True

    def set_device(self, device):
        """Move the model to the selected inference device."""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

    def set_output_norm(self, output_norm):
        """Enable or disable peak normalization for returned waveforms."""
        self.output_norm = bool(output_norm)

    def predict(self, batch):
        """Run a model-ready batch and return its named output dictionary."""
        model_batch = {
            key:
            value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            return self.model(model_batch)

    def _cue_input_type(self, cue_name):
        """Resolve the raw or precomputed input expected by one frontend."""
        frontend_name = {
            "audio": "spk_ft",
            "visual": "visual_ft",
        }.get(cue_name)
        if frontend_name and hasattr(self.model, frontend_name):
            return getattr(self.model, frontend_name).input_type

        model_config = self.configs["model_args"]["tse_model"]
        if cue_name == "audio":
            features = model_config.get("speaker", {}).get("features", {})
            inputs = {
                conf.get("input", "waveform")
                for conf in features.values() if conf.get("enabled", False)
            }
            return next(iter(inputs), "waveform")
        if cue_name == "visual":
            features = model_config.get("visual", {}).get("features", {})
            inputs = {
                conf.get("input", "raw_video")
                for conf in features.values() if conf.get("enabled", False)
            }
            return next(iter(inputs), "raw_video")
        return cue_name

    def _default_cue(self, cue_name, cue_conf):
        """Create a structural tensor for one missing optional cue."""
        input_type = self._cue_input_type(cue_name)
        if cue_name == "audio":
            default_key = f"audio_{input_type}"
        elif cue_name == "visual":
            default_key = ("visual_raw_video"
                           if input_type == "raw_video" else "visual_feature")
        else:
            default_key = cue_name

        fallback = CLI_CUE_DEFAULTS[default_key]
        shape = fallback["shape"]
        fill_value = fallback["fill_value"]
        if "default_shape" in cue_conf:
            shape = cue_conf["default_shape"]
        else:
            logging.warning(
                "Optional %s cue is missing; using CLI default shape %s. "
                "Set dataset_args.cues.%s.default_shape to override it.",
                cue_name,
                shape,
                cue_name,
            )
        fill_value = cue_conf.get("fill_value", fill_value)
        return torch.full(tuple(shape), fill_value)

    def build_batch(self, item):
        """Read one direct-input item and create a batch of size one."""
        if "wav_mix" not in item:
            raise ValueError("Direct inference requires 'wav_mix'.")

        batch = {"wav_mix": self.input_reader.read(item["wav_mix"], "wav_mix")}
        if batch["wav_mix"].ndim != 2:
            raise ValueError("Mixture waveform must have shape [C, T], got "
                             f"{tuple(batch['wav_mix'].shape)}")

        # Read supplied cues before materializing any configured fallback.
        for cue_name, aux_key in CUE_KEYS.items():
            if aux_key in item and item[aux_key] is not None:
                batch[aux_key] = self.input_reader.read(
                    item[aux_key],
                    aux_key,
                    input_type=self._cue_input_type(cue_name),
                )
                batch[f"{aux_key}_present"] = torch.tensor(True)

        # Required cues fail clearly; optional cues receive a structural value.
        for cue_name, cue_conf in self.cues.items():
            if not cue_conf.get("use", False):
                continue
            aux_key = CUE_KEYS.get(cue_name)
            if aux_key is None or aux_key in batch:
                continue
            if cue_conf.get("required", True):
                raise ValueError(f"Required {cue_name} cue was not provided.")
            batch[aux_key] = self._default_cue(cue_name, cue_conf)
            batch[f"{aux_key}_present"] = torch.tensor(False)

        # Add the singleton batch axis after every file has been decoded.
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0)
        return batch

    def predict_files(self, item):
        """Read one mixture/cue mapping and return speech shaped [S, T]."""
        speech = self.predict(self.build_batch(item))["speech"]
        if speech.ndim != 3 or speech.shape[0] != 1:
            raise RuntimeError(
                f"Model speech output must have shape [1, S, T], got "
                f"{tuple(speech.shape)}")
        speech = speech[0].detach().cpu()
        if self.output_norm:
            peak = speech.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            speech = speech / peak * 0.9
        return speech


def load_model(language):
    """Load a model package exposed by the optional download hub."""
    from wesep.cli.hub import Hub
    return Extractor(Hub.get_model(language))


def load_model_local(model_dir, config_path=None, checkpoint_path=None):
    """Load a model package from a local directory."""
    return Extractor(model_dir, config_path, checkpoint_path)


def _write_speech(speech, output_path, sample_rate):
    """Write one output file per estimated source."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if speech.shape[0] == 1:
        soundfile.write(output_path, speech[0].numpy(), sample_rate)
        return

    for index, source in enumerate(speech, start=1):
        source_path = output_path.with_name(
            f"{output_path.stem}_s{index}{output_path.suffix}")
        soundfile.write(source_path, source.numpy(), sample_rate)


def main():
    """Run direct single-request or JSONL-manifest inference."""
    args = get_args()
    extractor = load_model_local(
        args.model_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    extractor.set_device(args.device)
    extractor.set_output_norm(args.output_norm)

    if args.manifest:
        output_dir = Path(args.output_dir)
        used_keys = {}
        with Path(args.manifest).open("r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                item = json.loads(line)
                key = item.get("key", Path(item["wav_mix"]).stem)
                count = used_keys.get(key, 0) + 1
                used_keys[key] = count
                if count > 1:
                    key = f"{key}_{count}"
                output_path = item.get("output", output_dir / f"{key}.wav")
                speech = extractor.predict_files(item)
                _write_speech(speech, output_path, extractor.resample_rate)
        return

    item = {
        "wav_mix": args.wav_mix,
        "audio_aux": args.audio_aux,
        "spatial_aux": args.spatial_aux,
        "visual_aux": args.visual_aux,
        "textual_aux": args.textual_aux,
    }
    speech = extractor.predict_files(item)
    _write_speech(speech, args.output_file, extractor.resample_rate)


if __name__ == "__main__":
    main()
