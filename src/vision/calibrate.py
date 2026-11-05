"""Measure how far the classifier's damage percentage sits from the truth.

A classifier that errs in both directions does not merely add noise to a damage
percentage -- it compresses it. On a clean sample every mistake can only invent
damage, and on a heavily damaged one every mistake can only remove it, so
readings are pulled toward the middle of the range. The effect is systematic,
it has a closed form, and it is therefore correctable:

    d' = FPR + d(1 - FNR - FPR)

This script measures ``FPR`` and ``FNR`` on the **validation** split and writes
them to a calibration file the pipeline can load.

Which split, and why it matters. The correction must be measured on kernels the
correction is not later judged on, or it flatters itself. The validation split
is the honest choice here: the test split supplies the cutouts for the
synthetic trays that the whole pipeline is scored against, so calibrating on it
would be fitting the correction to the evaluation. Validation was used to pick
the training epoch, which is a weaker form of contact than gradient descent but
not zero -- expect the rates to be a little optimistic, and read the tray
results as the real test.

The correction changes absolute damage only. It is monotone, so ranking -- the
project's primary output -- is bit-for-bit identical with or without it.

These rates are *not* the ones to correct with. They describe the classifier on
clean dataset images, while the pipeline feeds it crops cut out of a crowded
tray, so they overstate the errors it actually makes: FPR 4.75% here against an
intercept of 1.71 points fitted on whole trays. They are worth measuring as a
decomposition -- which stage contributes what -- and ``src.calibrate_pipeline``
is the fit that would be applied, if any were. Read its docstring first: on
this corpus neither one transfers between acquisition sessions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.manifest import build, by_session, load_cornseeds, load_grainset
from src.mapping.classes import damage_confusion
from src.measurement.sampling import DamageCalibration
from src.vision.dataset import CLASS_NAMES, KernelDataset, eval_transform
from src.vision.train import build_model, device


@torch.no_grad()
def collect_pairs(model, loader, dev) -> list[tuple[str, str]]:
    """Every validation kernel as a ``(true class, predicted class)`` pair."""
    model.eval()
    pairs: list[tuple[str, str]] = []
    for images, labels, _ in loader:
        predictions = model(images.to(dev)).argmax(1).cpu()
        pairs.extend(
            (CLASS_NAMES[int(t)], CLASS_NAMES[int(p)])
            for t, p in zip(labels, predictions)
        )
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grainset", default="data/raw/grainset/maize")
    p.add_argument("--cornseeds", default="data/raw/kaggle/cornseeds/corn")
    p.add_argument("--classifier", default="models/classifier-real2.final.pt")
    p.add_argument("--out", default="models/calibration-perkernel.json")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--single-view",
        action="store_true",
        help="Match a checkpoint trained on split kernel views.",
    )
    args = p.parse_args()

    dev = device()
    manifest = build(
        load_grainset(Path(args.grainset)) + load_cornseeds(Path(args.cornseeds)),
        key=by_session,
    )
    samples = manifest.split("val")
    print(f"calibrating on {len(samples)} held-out kernels", flush=True)

    loader = DataLoader(
        KernelDataset(samples, eval_transform(args.size), single_view=args.single_view),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model = build_model().to(dev)
    state = torch.load(args.classifier, map_location=dev)
    model.load_state_dict(state["model"])

    counts = damage_confusion(collect_pairs(model, loader, dev))
    calibration = DamageCalibration.from_counts(**counts)

    print(
        f"  damaged  {counts['damaged_total']:>5}  missed  {counts['damaged_missed']:>5}"
        f"  -> FNR {calibration.false_negative_rate:.4f}\n"
        f"  sound    {counts['sound_total']:>5}  flagged {counts['sound_flagged']:>5}"
        f"  -> FPR {calibration.false_positive_rate:.4f}",
        flush=True,
    )

    # What the rates do to a reading, across the calibration mixtures. Printed
    # because a correction that moves a number by 0.1 points is not worth the
    # extra moving part, and this is where that shows.
    print("\n  measured -> corrected")
    for measured in (1.0, 3.0, 5.0, 10.0, 20.0, 40.0):
        print(f"  {measured:>8.1f} -> {calibration.correct(measured):8.2f}")

    payload = {
        **calibration.to_dict(),
        "classifier": args.classifier,
        "split": "val",
        "confusion": counts,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
