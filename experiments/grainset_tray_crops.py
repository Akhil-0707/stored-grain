"""Labelled GrainSet kernels cut out of synthetic trays, the way the app cuts them.

Fine-tuning on the team's popcorn crops alone taught the classifier that a pale
kernel at crop resolution is cracked: the cracked popcorn is paler than the
sound popcorn, and the only GrainSet images it saw were full-resolution pairs.
Dent-corn kernels cut from a tray then read as cracked, and a synthetic tray of
sound grain measured 25% damage.

These crops close that gap. Trays are composed from **train**-split cutouts, the
app's localiser finds the kernels, each detection takes the class of the
composited kernel it overlaps (IoU at least 0.5), and the crop is cut exactly as
``src.pipeline`` cuts it. The validation trays use the test split and the
calibration trays the val split, so neither shares a kernel with these.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.manifest import build, load_grainset
from src.pipeline import Pipeline, _crop
from src.vision.evaluate_localiser import predict_scene
from src.vision.synth import build_cutout_pool, compose, pool_for_damage
from src.vision.train_localiser import app_instances

LEVELS = (0.0, 5.0, 10.0, 20.0, 40.0, 60.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grainset", default="data/raw/grainset/maize")
    p.add_argument("--out", default="data/interim/grainset_tray_crops")
    p.add_argument("--scenes", type=int, default=48)
    p.add_argument("--kernels", type=int, default=200)
    p.add_argument("--pool", type=int, default=3000)
    args = p.parse_args()

    manifest = build(load_grainset(Path(args.grainset)))
    pool = build_cutout_pool(manifest.split("train"), limit=args.pool, seed=11)
    print(f"cutout pool: {len(pool)} (train split)", flush=True)
    pipeline = Pipeline()
    out = Path(args.out)
    counts: dict[str, int] = {}
    for i in range(args.scenes):
        level = LEVELS[i % len(LEVELS)]
        scene = compose(pool_for_damage(pool, level, args.kernels, seed=1000 + i), seed=1000 + i)
        with torch.no_grad():
            predicted = app_instances(predict_scene(pipeline.localiser, scene.image, pipeline.device))
        for label in [l for l in np.unique(predicted) if l]:
            mask = predicted == label
            truth_ids, overlap = np.unique(scene.instances[mask], return_counts=True)
            best = int(np.argmax(np.where(truth_ids > 0, overlap, 0)))
            truth = int(truth_ids[best])
            if truth == 0:
                continue
            union = mask.sum() + (scene.instances == truth).sum() - overlap[best]
            if overlap[best] / union < 0.5:
                continue
            name = scene.class_names[truth - 1]
            crop = _crop(scene.image, predicted, int(label))
            if crop is None:
                continue
            (out / name).mkdir(parents=True, exist_ok=True)
            crop.save(out / name / f"s{i:03d}_{int(label):03d}.png")
            counts[name] = counts.get(name, 0) + 1
        print(f"scene {i + 1}/{args.scenes} at {level:.0f}%: {sum(counts.values())} crops", flush=True)
    print(dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
