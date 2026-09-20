import argparse
from functools import partial

from torch.utils.data import DataLoader

from wesep.dataset.collate import (BASE_COLLECT_KEYS, build_collect_keys,
                                   tse_collate_fn)
from wesep.dataset.dataset import Dataset
from wesep.utils.file_utils import load_yaml


def main():
    parser = argparse.ArgumentParser(
        description="Build the configured data pipeline and inspect one batch."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--state",
                        choices=("train", "val", "test"),
                        default="train")
    parser.add_argument("--data-list")
    parser.add_argument("--data-type", choices=("raw", "shard"))
    parser.add_argument("--cues-yaml")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    configs = load_yaml(args.config)
    dataset_args = configs["dataset_args"]
    data_list = args.data_list or configs.get(f"{args.state}_data")
    data_type = args.data_type or configs.get("data_type")
    cues_yaml = args.cues_yaml or configs.get(f"{args.state}_cues")
    if data_list is None or data_type is None:
        raise ValueError(
            "data list and data type must be provided by the config or CLI.")

    # Build the same dataset and collate registration used by training.
    dataset = Dataset(
        data_type,
        data_list,
        dataset_args,
        state=args.state,
        repeat_dataset=False,
        cues_yaml=cues_yaml,
    )
    cues_conf = load_yaml(cues_yaml) if cues_yaml is not None else {}
    collect_keys = build_collect_keys(
        cues_conf,
        dataset_args,
        BASE_COLLECT_KEYS,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=partial(tse_collate_fn, collect_keys=collect_keys),
    )

    # Pull one batch and verify the core TSE waveform contract.
    batch = next(iter(dataloader))
    wav_mix = batch["wav_mix"]
    wav_target = batch["wav_target"]
    if wav_mix.shape[0] != wav_target.shape[0]:
        raise RuntimeError("wav_mix and wav_target batch sizes do not match.")
    if wav_mix.shape[-1] != wav_target.shape[-1]:
        raise RuntimeError("wav_mix and wav_target time lengths do not match.")

    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: list[{len(value)}]")


if __name__ == "__main__":
    main()
