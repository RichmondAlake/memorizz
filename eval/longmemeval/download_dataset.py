#!/usr/bin/env python3
"""
Download script for LongMemEval dataset

This script downloads the LongMemEval dataset from the official Hugging Face
repository and saves it to the correct location for the evaluation script.

Source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""

import argparse
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

# Hugging Face direct download URLs (official cleaned release)
_HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

# Maps the expected local filename to the remote filename on Hugging Face.
# The "s" and "m" variants were renamed to *_cleaned.json upstream.
DATASET_FILES = {
    "longmemeval_oracle.json": "longmemeval_oracle.json",
    "longmemeval_s.json": "longmemeval_s_cleaned.json",
    "longmemeval_m.json": "longmemeval_m_cleaned.json",
}


def _download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a simple progress indicator."""
    print(f"  {url}")
    print(f"  -> {dest}")
    urlretrieve(url, str(dest))


def main():
    """Download LongMemEval dataset."""
    parser = argparse.ArgumentParser(description="Download LongMemEval datasets")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional output directory for dataset files.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (script_dir / "data")
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    print("LongMemEval Dataset Downloader")
    print("=" * 40)
    print(f"Source: Hugging Face (xiaowu0162/longmemeval-cleaned)")
    print()

    all_found = True
    total_size = 0

    for local_name, remote_name in DATASET_FILES.items():
        dest = data_dir / local_name
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"✅ {local_name} already exists ({size_mb:.1f} MB), skipping.")
            continue

        url = f"{_HF_BASE}/{remote_name}"
        print(f"📥 Downloading {local_name} ...")
        try:
            _download_file(url, dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"✅ {local_name} downloaded ({size_mb:.1f} MB)")
        except (URLError, OSError) as exc:
            print(f"❌ Failed to download {local_name}: {exc}")
            all_found = False

    print()
    print("📋 VERIFYING FILES:")
    for local_name in DATASET_FILES:
        filepath = data_dir / local_name
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✅ {local_name} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {local_name} — missing")
            all_found = False

    print()
    if all_found:
        print(f"🎉 SUCCESS! All dataset files are ready ({total_size:.1f} MB total)")
        print()
        print("📊 DATASET VARIANTS:")
        print("  • longmemeval_oracle.json — Oracle retrieval (easiest, for testing)")
        print(
            "  • longmemeval_s.json      — Short version (~115k tokens, ~40 sessions)"
        )
        print("  • longmemeval_m.json      — Medium version (~500 sessions)")
        print()
        print("🚀 READY TO RUN EVALUATION:")
        print("  cd eval/longmemeval")
        print("  python evaluate_memorizz.py --dataset_variant oracle")
        print("  python evaluate_memorizz.py --dataset_variant s")
        print("  python evaluate_memorizz.py --dataset_variant m")
    else:
        print("⚠️  Some dataset files are missing.")
        print("Check your network connection and try again.")

    print(f"\n📂 Data directory: {data_dir}")
    print("📄 Dataset paper: https://arxiv.org/abs/2410.10813")


if __name__ == "__main__":
    main()
