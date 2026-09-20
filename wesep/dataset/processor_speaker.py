# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

import io
import logging
import random
import re

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy import signal

from wesep.dataset.FRAM_RIR import single_channel as RIR_sim
from wesep.utils.file_utils import load_json


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _restore_like(x, ref):
    if isinstance(ref, torch.Tensor):
        return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
    return np.asarray(x, dtype=ref.dtype if hasattr(ref, "dtype") else None)


def _read_waveform_item(item, target_sr=None):
    """Load an enrollment waveform as float32 [C, T]."""
    cue, sample_rate = torchaudio.load(item["path"])
    if target_sr is not None and sample_rate != target_sr:
        cue = torchaudio.functional.resample(cue, sample_rate, target_sr)
    return cue.float()


def _read_spk_embedding_item(item):
    """Load a speaker embedding and normalize it to float32 [D]."""
    cue = np.load(item["path"])
    cue = torch.from_numpy(np.asarray(cue)).float()
    if cue.ndim == 2 and cue.shape[0] == 1:
        cue = cue.squeeze(0)
    if cue.ndim != 1:
        raise ValueError(
            f"speaker embedding must be [D] or [1, D], got {tuple(cue.shape)}")
    return cue


# module-level cache (per worker process)
_SPK_RESOURCE_CACHE = {}


def sample_speaker_cue(
    data,
    resource_path,
    key_field,
    policy_type,
    cue_format,
    scope="speaker",
    required=True,
    target_sr=None,
):
    """Attach `audio_spk{i}` waveform or embedding tensors.

    Args:
        data: samples containing `key`, `spk{i}`, and optional
            `source_key_spk{i}` for online mixtures.
        resource_path: JSON mapping lookup keys to cue item lists.
        key_field: `spk_id` or `mix_spk_id`.
        policy_type: `random` or `fixed`.
        cue_format: (`wav`, `waveform`) or (`npy`, `spk_embedding`).

    Returns:
        Samples with available `audio_spk{i}` tensors attached.
    """
    if scope != "speaker":
        raise ValueError(
            "speaker audio cue currently supports scope='speaker'")

    # Select the representation reader and load its index once per worker.
    if cue_format == ("wav", "waveform"):
        item_reader = _read_waveform_item
        reader_kwargs = {"target_sr": target_sr}
    elif cue_format == ("npy", "spk_embedding"):
        item_reader = _read_spk_embedding_item
        reader_kwargs = {}
    else:
        raise ValueError(f"Unsupported speaker cue format: {cue_format}")
    if resource_path not in _SPK_RESOURCE_CACHE:
        _SPK_RESOURCE_CACHE[resource_path] = load_json(resource_path)
    spk_resource = _SPK_RESOURCE_CACHE[resource_path]

    for sample in data:
        # Resolve the ordered target speaker slots in the current sample.
        spk_slots = [key for key in sample if re.fullmatch(r"spk\d+", key)]
        spk_slots.sort(key=lambda key: int(key[3:]))
        if not spk_slots:
            if required:
                raise KeyError("sample has no speaker slots (spk1, spk2, ...)")
            yield sample
            continue

        for slot in spk_slots:
            # Build the dataset-level cue lookup key for this speaker.
            if key_field == "spk_id":
                lookup_key = sample[slot]
            elif key_field == "mix_spk_id":
                mix_key = sample.get(f"source_key_{slot}",
                                     sample.get("key", None))
                if mix_key is None:
                    raise KeyError("sample missing 'key' for mix_spk_id cue")
                lookup_key = f"{mix_key}::{sample[slot]}"
            else:
                raise ValueError(
                    f"Unsupported key_field for speaker cue: {key_field}")

            if lookup_key not in spk_resource:
                if required:
                    raise KeyError(f"speaker cue not found: {lookup_key}")
                continue

            items = spk_resource[lookup_key]
            if not items:
                if required:
                    raise RuntimeError(f"empty speaker cue set: {lookup_key}")
                continue

            if policy_type == "random":
                cue_item = random.choice(items)
            elif policy_type == "fixed":
                cue_item = items[0]
            else:
                raise ValueError(
                    f"Unsupported speaker cue policy: {policy_type}")

            # Read and normalize the declared waveform or embedding format.
            try:
                cue = item_reader(cue_item, **reader_kwargs)
            except Exception as e:
                logging.warning(
                    f"Failed to read speaker cue: {cue_item}, err={e}")
                if required:
                    raise
                continue

            sample[f"audio_{slot}"] = cue

        yield sample


simu_config = {
    "min_max_room": [[3, 3, 2.5], [10, 6, 4]],
    "rt60": [0.1, 0.7],
    "sr": 16000,
    "mic_dist": [0.2, 5.0],
    "num_src": 1,
}


def add_reverb_on_enroll(data, reverb_enroll_prob=0):
    """
    Args:
        data: Iterable[{key, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]

    Returns:
        Iterable[{key, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]

    """
    for sample in data:
        assert "num_speaker" in sample.keys()
        assert "sample_rate" in sample.keys()
        for i in range(sample["num_speaker"]):
            simu_config["sr"] = sample["sample_rate"]
            simu_config["num_src"] = 1
            rirs, _ = RIR_sim(simu_config)  # [n_mic, nsource, nsamples]
            rirs = rirs[0]  # [nsource, nsamples]
            if reverb_enroll_prob > random.random():
                # [1, audio_len], currently only support single-channel audio
                audio_key = f"audio_spk{i+1}"
                if audio_key not in sample:
                    continue
                audio = sample[audio_key]
                audio_np = _to_numpy(audio)
                # rir = rirs[i : i + 1, :]  # [1, nsamples]
                rir = rirs
                rir_audio = signal.convolve(
                    audio_np, rir,
                    mode="full")[:, :audio_np.shape[1]]  # [1, audio_len]

                max_scale = np.max(np.abs(rir_audio))
                out_audio = rir_audio / max_scale * 0.9
                # Note: Here, we do not replace the dry audio with the reverberant audio,  # noqa
                # which means we hope the model to perform dereverberation and
                # TSE simultaneously.
                sample[audio_key] = _restore_like(out_audio, audio)

        yield sample


def add_noise_on_enroll(
    data,
    noise_lmdb_file,
    noise_enroll_prob: float = 0.0,
    noise_db_low: int = 0,
    noise_db_high: int = 25,
    single_channel: bool = True,
):
    """Add noise to mixture

    Args:
        data: Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]
        noise_lmdb_file: noise LMDB data source.
        noise_db_low (int, optional): SNR lower bound. Defaults to 0.
        noise_db_high (int, optional): SNR upper bound. Defaults to 25.
        single_channel (bool, optional): Whether to force the noise file to be single channel.  # noqa
                                         Defaults to True.

    Returns:
        Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ..., noise, snr}]  # noqa
    """

    import librosa
    from wesep.dataset.lmdb_data import LmdbData

    noise_source = LmdbData(noise_lmdb_file)
    for sample in data:
        assert "sample_rate" in sample.keys()
        tgt_fs = sample["sample_rate"]
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith("spk") and "label" not in key:
                if noise_enroll_prob > random.random():
                    audio_key = "audio_" + key
                    if audio_key not in sample:
                        continue
                    speech = sample[audio_key]
                    speech_np = _to_numpy(speech)
                    nsamples = speech_np.shape[1]
                    power = (speech_np**2).mean()
                    noise_key, noise_data = noise_source.random_one()
                    if noise_key.startswith(
                            "speech"
                    ):  # using interference speech as additive noise
                        snr_range = [10, 30]
                    else:
                        snr_range = [noise_db_low, noise_db_high]
                    noise_db = np.random.uniform(snr_range[0], snr_range[1])
                    with sf.SoundFile(io.BytesIO(noise_data)) as f:
                        fs = f.samplerate
                        if tgt_fs and fs != tgt_fs:
                            nsamples_ = int(nsamples / tgt_fs * fs) + 1
                        else:
                            nsamples_ = nsamples
                        if f.frames == nsamples_:
                            noise = f.read(dtype=np.float64, always_2d=True)
                        elif f.frames < nsamples_:
                            offset = np.random.randint(0, nsamples_ - f.frames)
                            # noise: (Time, Nmic)
                            noise = f.read(dtype=np.float64, always_2d=True)
                            # Repeat noise
                            noise = np.pad(
                                noise,
                                [
                                    (offset, nsamples_ - f.frames - offset),
                                    (0, 0),
                                ],
                                mode="wrap",
                            )
                        else:
                            offset = np.random.randint(0, f.frames - nsamples_)
                            f.seek(offset)
                            # noise: (Time, Nmic)
                            noise = f.read(nsamples_,
                                           dtype=np.float64,
                                           always_2d=True)
                            if len(noise) != nsamples_:
                                raise RuntimeError(
                                    f"Something wrong: {noise_lmdb_file}")

                    if single_channel:
                        num_ch = noise.shape[1]
                        chs = [np.random.randint(num_ch)]
                        noise = noise[:, chs]
                    # noise: (Nmic, Time)
                    noise = noise.T
                    if tgt_fs and fs != tgt_fs:
                        logging.warning(
                            f"Resampling noise to match the sampling rate ({fs} -> {tgt_fs} Hz)"  # noqa
                        )
                        noise = librosa.resample(
                            noise,
                            orig_sr=fs,
                            target_sr=tgt_fs,
                            res_type="kaiser_fast",
                        )
                        if noise.shape[1] < nsamples:
                            noise = np.pad(
                                noise,
                                [(0, 0), (0, nsamples - noise.shape[1])],
                                mode="wrap",
                            )
                        else:
                            noise = noise[:, :nsamples]
                    noise_power = (noise**2).mean()
                    scale = (10**(-noise_db / 20) * np.sqrt(power) /
                             np.sqrt(max(noise_power, 1e-10)))
                    scaled_noise = scale * noise
                    speech_np = speech_np + scaled_noise
                    sample[audio_key] = _restore_like(speech_np, speech)
        yield sample
