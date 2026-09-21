# Copyright (c) 2026 Haoyu Li (haoyu.li.cs@sjtu.edu.cn)
#
# SPDX-License-Identifier: Apache-2.0

import json
import random
import zipfile

import numpy as np
import soundfile as sf
import torch

from wesep.dataset.collate import BASE_COLLECT_KEYS, tse_collate_fn
from wesep.dataset.processor import get_random_chunk, parse_raw, random_chunk
from wesep.dataset.processor_textual import sample_textual_cue
from wesep.dataset.textual_alignment import (
    AlignmentNotFoundError,
    LibriSpeechAlignmentReader,
)
from wesep.utils.losses import LossManager


def _write_alignment_archive(path, chapter_id, records):
    member = (
        f"LibriSpeech/train-clean-100/{chapter_id.replace('-', '/')}/"
        f"{chapter_id}.alignment.txt")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, "\n".join(records) + "\n")


def _loss_value(loss_fn, estimate, target):
    value = loss_fn(estimate, target)
    return value.mean() if value.ndim else value


def test_random_chunk_has_shared_bounds_and_silence_padding():
    mix = torch.arange(10, dtype=torch.float32).unsqueeze(0)
    target = (100 + torch.arange(10, dtype=torch.float32)).unsqueeze(0)
    random.seed(7)
    chunks, meta = get_random_chunk(
        [mix, target], chunk_len=4, random_start=True, mix_index=0)
    start = meta["chunk_start"]
    assert torch.equal(chunks[0], mix[..., start:start + 4])
    assert torch.equal(chunks[1], target[..., start:start + 4])
    assert meta["chunk_end"] == start + 4
    assert meta["valid_length"] == 4

    short = torch.tensor([[1.0, 2.0, 3.0]])
    padded, padded_meta = get_random_chunk(
        [short], chunk_len=5, random_start=False, short_policy="pad")
    assert torch.equal(padded[0], torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0]]))
    assert padded_meta["chunk_start"] == 0
    assert padded_meta["chunk_end"] == 3
    assert padded_meta["valid_length"] == 3

    sample = {
        "key": "example",
        "wav_mix": short.clone(),
        "wav_spk1": short.clone(),
    }
    chunked = next(random_chunk(iter([sample]), 5, random_start=False))
    assert chunked["chunk_ratio"] == {
        "start_ratio": 0.0,
        "end_ratio": 5 / 3,
        "orig_len": 3,
        "chunk_len": 5,
    }
    assert chunked["valid_length"] == 3


def test_collate_repeats_valid_length_for_each_target():
    collect_keys = {
        name: dict(BASE_COLLECT_KEYS[name])
        for name in ("wav_mix", "wav_target", "spk", "key", "num_speaker")
    }
    sample = {
        "key": "mix",
        "num_speaker": 2,
        "spk1": "a",
        "spk2": "b",
        "wav_mix": torch.ones(1, 6),
        "wav_spk1": torch.ones(1, 6),
        "wav_spk2": torch.ones(1, 6),
        "valid_length": 4,
    }
    batch = tse_collate_fn([sample], collect_keys)
    assert batch["valid_lengths"].tolist() == [4, 4]


def test_collate_uses_full_length_for_whole_utterance_samples():
    collect_keys = {
        name: dict(BASE_COLLECT_KEYS[name])
        for name in ("wav_mix", "wav_target", "spk", "key", "num_speaker")
    }
    sample = {
        "key": "whole-utterance-mix",
        "num_speaker": 2,
        "spk1": "a",
        "spk2": "b",
        "wav_mix": torch.ones(1, 7),
        "wav_spk1": torch.ones(1, 7),
        "wav_spk2": torch.ones(1, 7),
    }
    batch = tse_collate_fn([sample], collect_keys)
    assert batch["valid_lengths"].tolist() == [7, 7]


def test_valid_length_sisdr_matches_manual_trim_and_l1_is_unchanged():
    estimate = torch.tensor([
        [[1.0, 0.5, -0.2, 0.4, 10.0, -10.0]],
        [[0.2, -0.1, 0.7, 0.4, -0.3, 0.9]],
    ])
    target = torch.tensor([
        [[0.8, 0.2, -0.1, 0.3, 0.0, 0.0]],
        [[0.3, -0.4, 0.5, 0.2, -0.2, 0.7]],
    ])
    batch = {"wav_target": target, "valid_lengths": torch.tensor([4, 6])}
    outputs = {"speech": estimate}
    config = {
        "type": "SISDR",
        "output": "speech",
        "target": "wav_target",
        "stages": ["train", "val"],
    }
    enabled = LossManager(config, use_valid_length_for_sisnr_loss=True)
    enabled_value = enabled(outputs, batch, "train")
    manual = (
        _loss_value(enabled.losses[0], estimate[:1, ..., :4], target[:1, ..., :4])
        + _loss_value(enabled.losses[0], estimate[1:, ..., :6], target[1:, ..., :6])
    ) / 2
    assert torch.allclose(enabled_value, manual)

    disabled = LossManager(config, use_valid_length_for_sisnr_loss=False)
    disabled_value = disabled(outputs, batch, "train")
    assert torch.allclose(
        disabled_value, _loss_value(disabled.losses[0], estimate, target))

    l1_config = dict(config, type="L1")
    l1_enabled = LossManager(l1_config, use_valid_length_for_sisnr_loss=True)
    assert torch.allclose(
        l1_enabled(outputs, batch, "train"),
        _loss_value(l1_enabled.losses[0], estimate, target),
    )


def test_whole_utterance_optional_malformed_cue_is_skipped(tmp_path):
    resource = tmp_path / "textual.json"
    resource.write_text(json.dumps({
        "mix::speaker": {"phn_label": []},
    }))
    sample = {"key": "mix", "spk1": "speaker"}
    output = list(sample_textual_cue(
        iter([sample]),
        str(resource),
        key_field="mix_spk_id",
        cue_format=("json", "dae_keyword_phoneme"),
        format_args={"selection": {"type": "random", "word_range": [2, 6]}},
        required=False,
    ))
    assert len(output) == 1
    assert "textual_spk1" not in output[0]


def test_alignment_reader_and_chunk_textual_selection(tmp_path):
    primary = tmp_path / "primary.zip"
    _write_alignment_archive(
        primary,
        "1-2",
        ['1-2-3 "hello,,world" "0.10,0.20,0.40"'],
    )
    fallback_root = tmp_path / "fallback"
    fallback_file = fallback_root / "train-clean-100/3/4/3-4.alignment.txt"
    fallback_file.parent.mkdir(parents=True)
    fallback_file.write_text('3-4-5 ",again,world," "0.25,0.50"\n')

    with LibriSpeechAlignmentReader(primary, fallback_root) as reader:
        primary_lookup = reader.get_words("1-2-3")
        assert [(word.text, word.start, word.end) for word in primary_lookup.words] == [
            ("hello", 0.0, 0.10),
            ("world", 0.20, 0.40),
        ]
        fallback_lookup = reader.get_words("3-4-5")
        assert fallback_lookup.source == "fallback"
        assert [(word.text, word.start, word.end) for word in fallback_lookup.words] == [
            ("again", 0.0, 0.25),
            ("world", 0.25, 0.50),
        ]
        try:
            reader.get_words("9-9-9")
        except AlignmentNotFoundError:
            pass
        else:
            raise AssertionError("Expected missing alignment to raise")

    lexicon = tmp_path / "lexicon.txt"
    lexicon.write_text("hello 1 2\nworld 3\n")
    resource = tmp_path / "textual.json"
    resource.write_text(json.dumps({
        "1-2-3::speaker": {"phn_label": [[99]]},
    }))
    sample = {
        "key": "1-2-3",
        "spk1": "speaker",
        "sample_rate": 1000,
        "chunk_start": 0,
        "chunk_end": 100,
    }
    selection = {
        "type": "random",
        "word_range": [2, 2],
        "chunk_aware": {
            "enabled": True,
            "alignment_path": str(primary),
            "word2lexicon_path": str(lexicon),
            "no_word_policy": "discard",
        },
    }
    output = list(sample_textual_cue(
        iter([sample]),
        str(resource),
        key_field="mix_spk_id",
        cue_format=("json", "dae_keyword_phoneme"),
        format_args={"selection": selection},
    ))
    assert len(output) == 1
    assert output[0]["textual_spk1"].tolist() == [1, 2]

    no_word = dict(sample, chunk_start=100, chunk_end=200)
    assert list(sample_textual_cue(
        iter([no_word]),
        str(resource),
        key_field="mix_spk_id",
        cue_format=("json", "dae_keyword_phoneme"),
        format_args={"selection": selection},
    )) == []


def test_parse_raw_preserves_distributed_metadata(tmp_path):
    waveform_path = tmp_path / "source.wav"
    sf.write(waveform_path, np.ones(320, dtype=np.float32), 16000)
    raw_record = {
        "key": "source",
        "spk": ["speaker"],
        "mix": {"default": [str(waveform_path)]},
        "src": {"speaker": [str(waveform_path)]},
    }
    source_sample = {
        "src": json.dumps(raw_record),
        "rank": 1,
        "world_size": 2,
        "worker_id": 3,
        "num_workers": 4,
    }
    parsed = next(parse_raw(iter([source_sample])))
    assert {
        key: parsed[key]
        for key in ("rank", "world_size", "worker_id", "num_workers")
    } == {
        "rank": 1,
        "world_size": 2,
        "worker_id": 3,
        "num_workers": 4,
    }
