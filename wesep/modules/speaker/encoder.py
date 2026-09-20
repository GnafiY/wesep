import logging

import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi

from wesep.utils.torch_compat import (
    cuda_autocast,
    register_load_state_dict_pre_hook,
)


class Fbank_kaldi(nn.Module):
    """
    A wrapper module that performs:
    1) compute_fbank()
    2) apply_cmvn()

    Keep same with kaldi, not sure about the calculation efficiency.
    Strictly preserves the original arguments of both functions.
    """

    def __init__(
        self,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=1.0,
        sample_rate=16000,
        norm_mean=True,
        norm_var=False,
    ):
        super().__init__()
        # store parameters exactly as original function uses
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.sample_rate = sample_rate
        self.norm_mean = norm_mean
        self.norm_var = norm_var

    def compute_fbank(self, data):
        """Exact wrapper of the old compute_fbank()"""
        fbank_list = []
        for index_ in range(data.shape[0]):
            # Kaldi-style PCM scaling requires float32 to avoid AMP overflow.
            with cuda_autocast(enabled=False):
                waveform = data[index_, :].unsqueeze(0).float()
                waveform = waveform * (1 << 15)
                mat = kaldi.fbank(
                    waveform,
                    num_mel_bins=self.num_mel_bins,
                    frame_length=self.frame_length,
                    frame_shift=self.frame_shift,
                    dither=self.dither,
                    sample_frequency=self.sample_rate,
                    window_type="hamming",
                    use_energy=False,
                )
            fbank_list.append(mat.unsqueeze(0))
        np_fbank = torch.cat(fbank_list, 0)
        return np_fbank

    def apply_cmvn(self, data):
        """Exact wrapper of the old apply_cmvn()"""
        mat_list = []
        for index_ in range(data.shape[0]):
            mat = data[index_, :, :]
            if self.norm_mean:
                mat = mat - torch.mean(mat, dim=0)
            if self.norm_var:
                mat = mat / torch.sqrt(torch.var(mat, dim=0) + 1e-8)
            mat_list.append(mat.unsqueeze(0))
        np_mat = torch.cat(mat_list, 0)
        return np_mat

    def forward(self, data):
        """Compute fbank followed by CMVN."""
        fb = self.compute_fbank(data)
        fb = self.apply_cmvn(fb)
        return fb


class SpeakerEncoder(nn.Module):
    """
    Wraps get_speaker_model + loading pretrained + freezing.

    Args:
        conf:
            model_name (str): name used in get_speaker_model(...)
            spk_args (dict): arguments passed to the speaker model constructor
            pretrained (str or None): path to pretrained model
            freeze (bool): whether to freeze all parameters
    """

    def __init__(self, conf, defer_pretrained=False):
        super().__init__()

        # Import WeSpeaker only when a speaker encoder is actually enabled.
        try:
            from wespeaker.models.speaker_model import get_speaker_model
        except ImportError as exc:
            raise ImportError(
                "SpeakerEncoder requires the optional wespeaker package"
            ) from exc

        model_name = conf["model"]
        spk_args = conf.get("spk_args", {})
        self.pretrained = conf.get("pretrained", None)
        self.pretrained_loaded = False
        self.fallback_reported = False
        self.freeze = conf.get("freeze", self.pretrained is not None)

        # Build the encoder now and optionally defer external weight loading.
        self.spk_model = get_speaker_model(model_name)(**spk_args)
        if self.pretrained is not None and not defer_pretrained:
            self.load_pretrained()
        register_load_state_dict_pre_hook(self, self._prepare_pretrained_state)

        # Keep an external pretrained encoder fixed when requested.
        if self.freeze:
            self.spk_model.requires_grad_(False)
            self.spk_model.eval()

    def load_pretrained(self):
        """Load the external encoder checkpoint at most once."""
        if self.pretrained_loaded:
            return
        if self.pretrained is None:
            raise RuntimeError(
                "The TSE checkpoint has no speaker encoder parameters and "
                "speaker_encoder.pretrained is not configured.")

        pretrained_model = torch.load(self.pretrained, map_location="cpu")
        state = self.spk_model.state_dict()
        for key in state:
            if key in pretrained_model:
                state[key] = pretrained_model[key]
            else:
                logging.warning("Speaker encoder parameter not loaded: %s",
                                key)
        self.spk_model.load_state_dict(state)
        self.pretrained_loaded = True

    def _prepare_pretrained_state(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Use external weights when the TSE checkpoint omits this encoder."""
        local_state = self.state_dict()
        if any(prefix + key in state_dict for key in local_state):
            return

        self.load_pretrained()
        for key, value in self.state_dict().items():
            state_dict[prefix + key] = value

        if not self.fallback_reported:
            logging.warning(
                "Speaker encoder is absent from the TSE checkpoint; "
                "loaded external pretrained parameters from %s.",
                self.pretrained,
            )
            self.fallback_reported = True

    def train(self, mode=True):
        """Set training mode while keeping a frozen speaker model in eval mode."""
        super().train(mode)

        # Frozen batch normalization and dropout must remain deterministic.
        if self.freeze:
            self.spk_model.eval()

        return self

    def forward(self, x):
        """Run the configured speaker encoder."""
        return self.spk_model(x)
