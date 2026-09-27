#!/usr/bin/env python
"""Download the Ultralytics carparts-seg dataset, verify it, and print what is inside.

    python tools/download_data.py            # -> data/raw/carparts-seg/
    python tools/download_data.py --force    # delete and download again

The dataset is AGPL-3.0 (see the header of its YAML file). It is never committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip"
# Filled in from the first verified download. None means: print the hash, do not check it.
EXPECTED_SHA256: str | None = None
SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    def progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            print(f"\r  {min(blocks * block_size, total) / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    socket.setdefaulttimeout(60)
    urllib.request.urlretrieve(url, part, reporthook=progress)
    print()
    part.rename(dest)  # only a complete download ever gets the final name


def extract(archive: Path, out: Path) -> None:
    staging = out.with_name(out.name + ".extracting")
    shutil.rmtree(staging, ignore_errors=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(staging)
    entries = [p for p in staging.iterdir() if not p.name.startswith(("__MACOSX", "."))]
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
    shutil.rmtree(out, ignore_errors=True)
    shutil.move(str(root), str(out))
    shutil.rmtree(staging, ignore_errors=True)


def summarize(root: Path) -> dict:
    report: dict = {"root": str(root), "splits": {}}
    yamls = sorted(root.glob("*.yaml"))
    if yamls:
        text = yamls[0].read_text()
        report["yaml"] = yamls[0].name
        report["license_lines"] = [
            ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("#") and "licen" in ln.lower()
        ]
        report["num_classes"] = len((yaml.safe_load(text) or {}).get("names", {}))
    for split in SPLITS:
        images = {p.stem for p in (root / "images" / split).glob("*") if p.suffix.lower() in IMG_EXT}
        labels = {p.stem: p for p in (root / "labels" / split).glob("*.txt")}
        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "empty_labels": sum(1 for p in labels.values() if not p.read_text().strip()),
            "images_without_label": len(images - labels.keys()),
            "labels_without_image": len(labels.keys() - images),
        }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/raw/carparts-seg"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    archive = args.out.parent / "carparts-seg.zip"
    if args.force:
        archive.unlink(missing_ok=True)
        shutil.rmtree(args.out, ignore_errors=True)
    if not archive.exists():
        print(f"downloading {URL}")
        download(URL, archive)

    digest = sha256sum(archive)
    print(f"sha256 {digest}")
    if EXPECTED_SHA256 is None:
        print("EXPECTED_SHA256 is not set: copy the hash above into tools/download_data.py")
    elif digest != EXPECTED_SHA256:
        sys.exit("checksum mismatch: rerun with --force; if the upstream asset really changed, update EXPECTED_SHA256 on purpose")
    else:
        print("checksum OK")

    if not (args.out / "images").exists():
        print(f"extracting to {args.out}")
        extract(archive, args.out)

    report = summarize(args.out)
    print(json.dumps(report, indent=2))
    if any(s["images"] == 0 for s in report["splits"].values()):
        sys.exit("a split has no images; the archive layout is not what this script expects")


if __name__ == "__main__":
    main()