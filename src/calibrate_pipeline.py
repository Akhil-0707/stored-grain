"""Fit the pipeline's bias on trays of known damage, held out from the test set.

The classifier's per-kernel error rates, measured on clean dataset images,
describe a situation the pipeline is never in. Downstream the classifier sees
crops the localiser cut out of a crowded tray, at whatever scale the kernels
happened to occupy, sometimes clipped by a neighbour. Correcting with those
rates over-corrects, because they overstate the errors the pipeline actually
makes.

So fit the whole thing at once. Compose trays of known damage from the
**validation** cutouts, read each one with the uncalibrated pipeline, and fit

    measured = intercept + slope x true

by least squares. The intercept is the damage a sound tray appears to have; the
slope is how much of real damage survives to the reading. Inverting that line
is the correction, and it absorbs every stage -- localiser recall, the debris
threshold, crop quality and the classifier -- rather than the classifier alone.

Split discipline, which is the whole point. The line is fitted on validation
cutouts and applied to test cutouts in ``src.validate_pipeline``. Kernels used
here never appear there. A correction fitted on the trays it is then scored
against would report its own residuals and prove nothing.

**The correction does not transfer between sessions, and that is the result.**
Measured 18 August 2026, three estimates of the same line:

    per-kernel confusion, val images   4.75 + 0.774 x true
    trays composed from val cutouts    1.71 + 0.913 x true   (R2 0.976)
    trays composed from test cutouts   0.96 + 0.842 x true

Each is a good fit to its own data -- the tray fit explains 98% of the spread --
and applying either val line to the test trays makes the reading worse, not
better: mean absolute error goes from 1.11 points uncorrected to 1.33 with the
val tray fit and 0.85 with the per-kernel rates, the latter only by accident of
where the two errors cancel. The intercept is the unstable term. A sound val
tray reads 2.77% damaged and a sound test tray 1.44%, because the false
positives come from how a particular acquisition session lit and staged its
kernels, not from the classifier alone.

So the pipeline ships uncorrected. That is why ``--calibration`` is off by
default in ``src.validate_pipeline``: an uncorrected reading is merely biased,
while a reading corrected by someone else's session is biased *and* claims not
to be.

What would make this legitimate. Correction needs calibration trays from the
same session as the lots being read -- which the physical protocol already
produces, since the weighed 0/2/5/10/20/40% mixtures are photographed on the
same rig on the same day as everything else. Fit the line on those, apply it to
that session, refit next session. The machinery here is what does that; only
the cross-session shortcut is unsound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.manifest import build, load_grainset
from src.measurement.sampling import DamageCalibration
from src.pipeline import Pipeline
from src.vision.synth import build_cutout_pool, compose, pool_for_damage

#: More levels than the calibration mixtures use, because a line fitted to six
#: points is mostly fitted to their noise. These are synthetic and free.
LEVELS = (0.0, 1.0, 2.0, 5.0, 8.0, 12.0, 20.0, 30.0, 40.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grainset", default="data/raw/grainset/maize")
    p.add_argument("--localiser", default="models/localiser-real.pt")
    p.add_argument("--classifier", default="models/classifier.session.pt")
    p.add_argument("--out", default="data/interim/calibration")
    p.add_argument("--results", default="models/calibration-trays.json")
    p.add_argument("--kernels", type=int, default=200)
    p.add_argument("--pool", type=int, default=400)
    p.add_argument("--temperature", type=float, default=20.0)
    p.add_argument("--moisture", type=float, default=14.0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = build(load_grainset(Path(args.grainset)))
    pool = build_cutout_pool(manifest.split("val"), limit=args.pool, seed=11)
    print(f"cutout pool: {len(pool)} (from the val split)", flush=True)

    # Seeds differ from the validation run's so that no tray is composed twice
    # from the same draw, even where the pools happen to overlap in class mix.
    pipeline = Pipeline(args.localiser, args.classifier, calibration=None)

    truths, measured, kernels = [], [], 0
    print(f"\n{'level':>7}{'true%':>8}{'meas%':>8}{'kernels':>9}", flush=True)
    for level in LEVELS:
        selection = pool_for_damage(pool, level, args.kernels, seed=int(level) + 500)
        scene = compose(selection, seed=int(level) + 500)
        path = out / f"cal_{int(level):02d}.png"
        scene.save(path)

        reading = pipeline.read(
            path,
            lot_id=path.stem,
            temperature_c=args.temperature,
            moisture_pct_wb=args.moisture,
        )
        truths.append(scene.truth.mechanical_damage_mass_pct)
        measured.append(reading.damage_mass_pct)
        kernels += reading.kernels_counted
        print(
            f"{level:>7.1f}{truths[-1]:>8.2f}{measured[-1]:>8.2f}"
            f"{reading.kernels_counted:>9}",
            flush=True,
        )

    true = np.array(truths)
    seen = np.array(measured)
    slope, intercept = np.polyfit(true, seen, 1)
    residuals = seen - (intercept + slope * true)
    # Share of the spread in the readings the line accounts for. A low value
    # means the bias is not affine and inverting the line will not fix it.
    r2 = 1.0 - float(residuals.var() / seen.var())

    calibration = DamageCalibration.from_affine(
        intercept_pct=float(intercept), slope=float(slope), kernels=kernels, basis="mass"
    )

    print(
        f"\nmeasured = {intercept:.3f} + {slope:.4f} x true   (R2 {r2:.4f})\n"
        f"  a sound tray reads {intercept:.2f}% damaged\n"
        f"  {(1 - slope) * 100:.1f}% of real damage is lost on the way to the reading",
        flush=True,
    )
    print("\n  measured -> corrected")
    for m in (1.0, 3.0, 5.0, 10.0, 20.0, 40.0):
        print(f"  {m:>8.1f} -> {calibration.correct(m):8.2f}")

    payload = {
        **calibration.to_dict(),
        "fit": {
            "intercept_pct": float(intercept),
            "slope": float(slope),
            "r_squared": r2,
            "levels": list(LEVELS),
            "true_damage_mass_pct": truths,
            "measured_damage_mass_pct": measured,
        },
        "classifier": args.classifier,
        "localiser": args.localiser,
        "split": "val",
    }
    Path(args.results).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.results}", flush=True)


if __name__ == "__main__":
    main()
