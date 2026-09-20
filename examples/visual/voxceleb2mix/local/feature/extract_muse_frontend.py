#!/usr/bin/env python3
"""Extract MuSE frame-level features and build visual cue indexes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.io import read_video

from wesep.modules.visual.muse import (
    Muse_LipROIProcessor,
    Muse_VisualFrontend,
    compute_muse_frontend,
    load_muse_frontend,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress every N visual cue entries; set <=0 to disable.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    device = torch.device(args.device)

    # Use exactly the same frame frontend as raw-video model training.
    roi = Muse_LipROIProcessor().to(device).eval()
    frontend = Muse_VisualFrontend().to(device).eval()
    load_muse_frontend(frontend, args.checkpoint)

    for split in ("train", "val", "test"):
        cue_dir = dataset_root / split / "cues"
        with open(cue_dir / "visual.json", encoding="utf-8-sig") as stream:
            raw_index = json.load(stream)

        feature_paths = {}
        muse_index = {}
        total = len(raw_index)
        print(f"Extracting {split} MuSE features: {total} cue entries",
              flush=True)
        for index, (cue_key, items) in enumerate(raw_index.items(), 1):
            item = items[0]
            video_path = item["path"]
            if video_path not in feature_paths:
                video, _, info = read_video(video_path, pts_unit="sec")
                if video.numel() == 0:
                    raise RuntimeError(f"Empty video: {video_path}")
                fps = float(info["video_fps"])
                if fps <= 0:
                    raise ValueError(f"Invalid video FPS: {video_path}")

                # Keep the final dimension as the cue time axis.
                video = video.permute(1, 2, 3, 0).unsqueeze(0).to(device)
                with torch.inference_mode():
                    feature = compute_muse_frontend(video, roi,
                                                    frontend).squeeze(0)

                utt_id = item["utt_id"]
                feature_path = output_root / split / f"{utt_id}.npy"
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(
                    feature_path,
                    feature.cpu().numpy().astype(np.float32, copy=False),
                )
                feature_paths[video_path] = (str(feature_path),
                                             video.shape[-1] / fps)

            feature_path, duration_sec = feature_paths[video_path]
            muse_index[cue_key] = [{
                "utt_id": item["utt_id"],
                "path": feature_path,
                "duration_sec": duration_sec,
            }]
            if (args.progress_interval > 0 and
                (index % args.progress_interval == 0 or index == total)):
                print(
                    f"Extracted {index}/{total} {split} cue entries "
                    f"({len(feature_paths)} unique videos)",
                    flush=True,
                )

        with open(cue_dir / "visual_muse.json", "w",
                  encoding="utf-8") as stream:
            json.dump(muse_index, stream, indent=2)
        print(f"Saved {len(feature_paths)} {split} features and "
              f"{len(muse_index)} cue entries")

    # Make the generated dataset advertise its precomputed cue view.
    with open(dataset_root / "visual_cue.json", "w",
              encoding="utf-8") as stream:
        json.dump(
            {
                "type": "npy",
                "format": "muse_frontend",
                "index": "visual_muse.json",
            },
            stream,
            indent=2)


if __name__ == "__main__":
    main()
