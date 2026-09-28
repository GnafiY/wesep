"""Summarize target-speaker SI-SNR results from WeSep inference logs."""

import argparse
import math
from pathlib import Path
from statistics import mean, pstdev


ALPHA = 10.0
BETA = 1.0
CASE_CLASSES = (
    "successful cases with high SI-SNRi",
    "successful cases with low SI-SNRi",
    "unsuccessful cases",
)


def classify_result(sisnri: float, alpha: float = ALPHA, beta: float = BETA) -> int:
    if sisnri >= alpha:
        return 0
    if sisnri >= beta:
        return 1
    return 2


def get_mean_std(values: list[float]) -> str:
    if not values:
        return "N/A"
    return f"{mean(values):.2f}±{pstdev(values):.2f}"


def parse_line(line: str) -> tuple[float, float] | None:
    """Read one target-speaker score; ignore other inference log messages."""
    prefix, separator, scores = line.partition(" | SI-SNR=")
    if not separator or " | SI-SNRi=" not in scores:
        return None
    if "Num=" not in prefix or " | Utt=" not in prefix or " | Target speaker=" not in prefix:
        return None

    sisnr_text, _, sisnri_text = scores.partition(" | SI-SNRi=")
    sisnr = float(sisnr_text.strip())
    sisnri = float(sisnri_text.partition(" | ")[0].strip())
    if not math.isfinite(sisnr) or not math.isfinite(sisnri):
        raise ValueError("SI-SNR and SI-SNRi must be finite")
    return sisnr, sisnri


def log_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix != ".log":
            raise ValueError(f"Expected a .log file: {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.log"))
        if files:
            return files
        raise ValueError(f"No .log files found in: {path}")
    raise ValueError(f"Log file or directory does not exist: {path}")


def merge_log(path: Path, alpha: float = ALPHA, beta: float = BETA) -> None:
    if not math.isfinite(alpha) or not math.isfinite(beta) or alpha <= beta:
        raise ValueError("alpha and beta must be finite, with alpha greater than beta")

    all_sisnr = []
    grouped_sisnri = [[] for _ in CASE_CLASSES]
    for file in log_files(path):
        with file.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    result = parse_line(line)
                except ValueError as exc:
                    raise ValueError(f"{file}:{line_number}: {exc}") from exc
                if result is None:
                    continue
                sisnr, sisnri = result
                all_sisnr.append(sisnr)
                grouped_sisnri[classify_result(sisnri, alpha, beta)].append(sisnri)

    total = len(all_sisnr)
    if not total:
        raise ValueError(f"No target-speaker SI-SNR results found in: {path}")

    print(f"Alpha: {alpha:g} dB, beta: {beta:g} dB; target-speaker results: {total}")
    print("1. Case distribution:")
    for label, values in zip(CASE_CLASSES, grouped_sisnri):
        print(f"case [{label}]: {len(values)} ({len(values) / total * 100:.2f}%)")
    successful = grouped_sisnri[0] + grouped_sisnri[1]
    unsuccessful = grouped_sisnri[2]
    print(f"Successful case rate: {len(successful) / total * 100:.2f}%")

    print("2. SI-SNRi statistics:")
    for label, values in zip(CASE_CLASSES, grouped_sisnri):
        print(f"case [{label}]: {get_mean_std(values)}")
    print(f"Overall: {get_mean_std(successful + unsuccessful)}")

    print("3. SI-SNRi statistics on successful/unsuccessful cases:")
    print(f"case [successful cases]: {get_mean_std(successful)}")
    print(f"case [unsuccessful cases]: {get_mean_std(unsuccessful)}")
    print(f"Overall SI-SNR: {get_mean_std(all_sisnr)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path,
                        help="An inference .log file or a directory containing .log files")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    args = parser.parse_args()
    try:
        merge_log(args.log, args.alpha, args.beta)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
