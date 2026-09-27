#!/usr/bin/env python
"""Print the environment and prove the selected device computes a matmul correctly.

    python tools/check_env.py
    python tools/check_env.py --device cpu
"""

from __future__ import annotations

import argparse
import json

import torch

from carparts.utils import env_report, pick_device, seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    print(json.dumps(env_report(), indent=2))
    seed_everything(0)
    device = pick_device(args.device)
    a, b = torch.randn(1024, 1024), torch.randn(1024, 1024)
    ref = a @ b
    out = (a.to(device) @ b.to(device)).cpu()
    err = (out - ref).abs().max().item()
    print(f"device {device}: max abs diff vs CPU matmul = {err:.2e}")
    if err > 1e-2:
        raise SystemExit("device result disagrees with the CPU; do not train on this device until this is fixed")
    print("device OK")


if __name__ == "__main__":
    main()