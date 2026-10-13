#!/usr/bin/env python3
"""Run reproducible PixMix still-image evaluation against RORD-50 ground truth."""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageStat


QUICK_IDS = [
    "I-210907_O03015_W03",
    "I-211025_O11048_W10",
    "I-210618_I01001_W01",
    "I-211016_I09028_T02",
    "I-211117_O07035_T19",
]


def masked_mae(left: Image.Image, right: Image.Image, mask: Image.Image) -> float:
    return sum(ImageStat.Stat(ImageChops.difference(left, right), mask).mean) / 3.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./build/pixmix")
    parser.add_argument("--data", default="data/rord50")
    parser.add_argument("--output", default="/tmp/pixmix-rord-eval")
    parser.add_argument("--json", default="")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--baseline", default="tools/rord_quick_baseline.json")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dilate", type=int, default=8)
    parser.add_argument("--tau-s", type=float)
    parser.add_argument("--tau-a", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--locality", type=float)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--no-structure", action="store_true")
    args = parser.parse_args()

    data = Path(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ids = sorted(path.stem for path in (data / "gt").glob("*.png")) if args.all else QUICK_IDS
    rows = []

    for index, image_id in enumerate(ids, 1):
        image_path = data / "stills" / f"{image_id}.png"
        mask_path = data / "stills" / f"{image_id}_mask.png"
        output_path = output / f"{image_id}.png"
        command = [
            args.binary,
            "--still",
            str(image_path),
            "--mask",
            str(mask_path),
            "--dilate",
            str(args.dilate),
            "--seed",
            str(args.seed),
            "--output",
            str(output_path),
        ]
        if args.no_structure:
            command.append("--no-structure")
        if args.tau_s is not None:
            command.extend(["--tau-s", str(args.tau_s)])
        if args.tau_a is not None:
            command.extend(["--tau-a", str(args.tau_a)])
        if args.alpha is not None:
            command.extend(["--alpha", str(args.alpha)])
        if args.locality is not None:
            command.extend(["--locality", str(args.locality)])
        import os

        environment = dict(os.environ)
        environment["PIXMIX_PROFILE"] = "1"
        started = time.perf_counter()
        result = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
        wall_ms = (time.perf_counter() - started) * 1000.0
        match = re.search(r"hole=(\d+)\s+inpaint_ms=([\d.]+)", result.stdout)
        if not match:
            raise RuntimeError(f"could not parse PixMix output: {result.stdout!r}")

        source = Image.open(image_path).convert("RGB")
        truth = Image.open(data / "gt" / f"{image_id}.png").convert("RGB")
        rendered = Image.open(output_path).convert("RGB")
        mask = Image.open(mask_path).convert("L").point(lambda value: 255 if value > 127 else 0)
        if args.dilate > 0:
            mask = mask.filter(ImageFilter.MaxFilter(args.dilate * 2 + 1))
        row = {
            "id": image_id,
            "seed": args.seed,
            "dilate": args.dilate,
            "hole_pixels": int(match.group(1)),
            "inpaint_ms": float(match.group(2)),
            "wall_ms": wall_ms,
            "mae": masked_mae(rendered, truth, mask),
            "input_mae": masked_mae(source, truth, mask),
        }
        choices = re.findall(r"selected=(regular|structured)", result.stderr)
        row["candidate"] = choices[-1] if choices else "regular"
        if args.profile:
            row["profile"] = result.stderr.strip().splitlines()
        rows.append(row)
        print(
            f"{index:02d}/{len(ids)} {image_id} "
            f"{row['inpaint_ms']:.1f} ms MAE={row['mae']:.2f}",
            flush=True,
        )

    report = {
        "seed": args.seed,
        "dilate": args.dilate,
        "structural_candidate": not args.no_structure,
        "tau_s": args.tau_s,
        "tau_a": args.tau_a,
        "alpha": args.alpha,
        "locality": args.locality,
        "count": len(rows),
        "mean_mae": sum(row["mae"] for row in rows) / len(rows),
        "median_inpaint_ms": sorted(row["inpaint_ms"] for row in rows)[len(rows) // 2],
        "rows": rows,
    }
    serialized = json.dumps(report, indent=2)
    if args.json:
        Path(args.json).write_text(serialized + "\n")
    print(serialized)
    if args.check:
        if args.all:
            raise SystemExit("--check is only defined for the fixed quick subset")
        baseline = json.loads(Path(args.baseline).read_text())
        quality_limit = baseline["mean_mae"] + baseline["allowed_mae_increase"]
        runtime_limit = baseline["median_inpaint_ms"] * baseline["runtime_multiplier"]
        failures = []
        if report["mean_mae"] > quality_limit:
            failures.append(f"mean MAE {report['mean_mae']:.3f} > {quality_limit:.3f}")
        if report["median_inpaint_ms"] > runtime_limit:
            failures.append(
                f"median inpaint {report['median_inpaint_ms']:.1f} ms > {runtime_limit:.1f} ms"
            )
        if failures:
            raise SystemExit("RORD quick regression: " + "; ".join(failures))


if __name__ == "__main__":
    main()
