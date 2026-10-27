"""Measure the whole system against trays whose damage is known exactly.

This is the project's central claim rehearsed end to end: photograph a sample,
recover the percentage of it that is mechanically damaged, and order lots by
how fast they are degrading. Synthetic scenes make it testable before any maize
is bought, because a composited tray's damage is known by construction rather
than by annotation.

What this does and does not establish. It measures whether the parts compose --
whether a localiser at 0.7% counting error and a classifier at 0.90 macro-F1
combine into a damage figure close to the truth, or whether their errors
multiply. It says nothing about real photographs: synthetic kernels cast no
shadows on each other and every cutout came from the same laboratory rig.
Read it as an integration test with numbers, not as a performance claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.manifest import build, load_grainset
from src.pipeline import Pipeline, rank_lots
from src.vision.synth import build_cutout_pool, compose, pool_for_damage

#: Damage levels the calibration mixtures use, so the synthetic rehearsal
#: matches the physical protocol it stands in for.
LEVELS = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grainset", default="data/raw/grainset/maize")
    p.add_argument("--localiser", default="models/localiser-real.pt")
    p.add_argument("--classifier", default="models/classifier.session.pt")
    p.add_argument("--out", default="data/interim/validation")
    p.add_argument("--kernels", type=int, default=200)
    p.add_argument("--pool", type=int, default=400)
    p.add_argument("--results", default="models/pipeline-validation.json")
    p.add_argument(
        "--calibration",
        default="",
        help=(
            "Bias correction to divide out. Off by default: the correction "
            "does not transfer between acquisition sessions -- see the "
            "docstring of src.calibrate_pipeline."
        ),
    )
    p.add_argument("--temperature", type=float, default=20.0)
    p.add_argument("--moisture", type=float, default=14.0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Validation kernels come from the held-out split, so nothing the models
    # trained on reappears here.
    manifest = build(load_grainset(Path(args.grainset)))
    pool = build_cutout_pool(manifest.split("test"), limit=args.pool, seed=7)
    print(f"cutout pool: {len(pool)} (from the test split)", flush=True)

    truths, paths = [], []
    for level in LEVELS:
        selection = pool_for_damage(pool, level, args.kernels, seed=int(level) + 100)
        scene = compose(selection, seed=int(level) + 100)
        path = out / f"lot_{int(level):02d}.png"
        scene.save(path)
        truths.append(scene.truth)
        paths.append(path)
        print(
            f"lot {int(level):>2}% -> placed {scene.placed:>3} | "
            f"true mass damage {scene.truth.mechanical_damage_mass_pct:5.2f}%",
            flush=True,
        )

    pipeline = Pipeline(args.localiser, args.classifier, calibration=args.calibration or None)
    if pipeline.calibration is not None:
        c = pipeline.calibration
        print(
            f"\ncalibration: FPR {c.false_positive_rate:.4f} "
            f"FNR {c.false_negative_rate:.4f} over {c.kernels} held-out kernels",
            flush=True,
        )

    readings, rows = [], []
    print(
        f"\n{'lot':>6}{'true%':>8}{'raw%':>8}{'meas%':>8}{'err':>8}"
        f"{'kernels':>9}{'truekern':>10}{'gap±':>7}",
        flush=True,
    )
    for path, truth in zip(paths, truths):
        reading = pipeline.read(
            path,
            lot_id=path.stem,
            temperature_c=args.temperature,
            moisture_pct_wb=args.moisture,
        )
        readings.append(reading)
        raw = reading.raw_damage_mass_pct
        error = reading.damage_mass_pct - truth.mechanical_damage_mass_pct
        rows.append(
            {
                "lot": path.stem,
                "true_damage_mass_pct": truth.mechanical_damage_mass_pct,
                "measured_damage_mass_pct": reading.damage_mass_pct,
                "raw_damage_mass_pct": raw,
                "error_pct_points": error,
                "raw_error_pct_points": raw - truth.mechanical_damage_mass_pct,
                "kernels_counted": reading.kernels_counted,
                "true_kernels": truth.kernels_counted,
                "resolvable_gap_pct": reading.resolvable_gap_pct,
            }
        )
        print(
            f"{path.stem[-2:]:>6}{truth.mechanical_damage_mass_pct:>8.2f}"
            f"{raw:>8.2f}{reading.damage_mass_pct:>8.2f}{error:>+8.2f}"
            f"{reading.kernels_counted:>9}{truth.kernels_counted:>10}"
            f"{reading.resolvable_gap_pct:>7.2f}",
            flush=True,
        )

    errors = np.array([r["error_pct_points"] for r in rows])
    raw_errors = np.array([r["raw_error_pct_points"] for r in rows])
    true_order = [
        r["lot"]
        for r in sorted(rows, key=lambda r: r["true_damage_mass_pct"], reverse=True)
    ]
    ranked = rank_lots(readings)
    predicted_order = [r.reading.lot_id for r in ranked]

    print("\nranking")
    for entry in ranked:
        tied = f"  (tied with {', '.join(entry.tied_with)})" if entry.tied_with else ""
        print(
            f"  {entry.rank}. {entry.reading.lot_id} "
            f"damage {entry.reading.damage_mass_pct:5.2f}% "
            f"rate {entry.reading.assessment.degradation_rate:.4f}{tied}",
            flush=True,
        )

    summary = {
        "mean_abs_error_pct_points": float(np.mean(np.abs(errors))),
        "max_abs_error_pct_points": float(np.max(np.abs(errors))),
        "mean_signed_error_pct_points": float(np.mean(errors)),
        "raw_mean_abs_error_pct_points": float(np.mean(np.abs(raw_errors))),
        "raw_max_abs_error_pct_points": float(np.max(np.abs(raw_errors))),
        "raw_mean_signed_error_pct_points": float(np.mean(raw_errors)),
        "calibrated": pipeline.calibration is not None,
        "ranking_correct": predicted_order == true_order,
        "true_order": true_order,
        "predicted_order": predicted_order,
        "mode": ranked[0].reading.assessment.mode,
    }
    print("\n" + json.dumps(summary, indent=2), flush=True)
    Path(args.results).write_text(json.dumps({"summary": summary, "lots": rows}, indent=2))


if __name__ == "__main__":
    main()
