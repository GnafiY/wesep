import argparse


def get_args():
    """Parse direct-file and JSONL-manifest CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Extract target speech from direct local inputs.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing config.yaml and avg_model.pt.")
    parser.add_argument(
        "--config",
        help="Optional alternate config, for example a raw-cue config.")
    parser.add_argument("--checkpoint", help="Optional checkpoint override.")
    parser.add_argument("--device", default="cpu")

    # A manifest contains one direct target-level request per JSON line.
    parser.add_argument("--manifest", help="Direct-input JSONL manifest.")
    parser.add_argument("--output-dir", default="extracted_speech")

    # Single-request inputs use the same model-facing names as training.
    parser.add_argument("--wav-mix")
    parser.add_argument("--audio-aux")
    parser.add_argument("--spatial-aux")
    parser.add_argument("--visual-aux")
    parser.add_argument("--textual-aux")
    parser.add_argument("--output-file", default="extracted_speech.wav")
    parser.add_argument(
        "--output-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if not args.manifest and not args.wav_mix:
        parser.error("one of --manifest or --wav-mix is required")
    return args
