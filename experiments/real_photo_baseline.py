"""Cut the team's labelled phone photos into kernel crops and score the current classifier on them.

Photos live in ``Classifier Photos/<class>/<batch>/``. Sound batches are labelled
by bowl: every kernel the localiser finds counts, minus merged blobs and kernels
cut by the frame edge. Cracked and broken batches carry a ``kernel_labels.json``
of hand-checked boxes; only those kernels are used. A batch whose labels say
``"independent": false`` is re-sampled from other batches and is never scored.

Crops are cut exactly as the pipeline cuts them (working scale, bounding box plus
4 px) and saved for later training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.app.images import to_working_scale
from src.mapping.classes import CLASSES, Admissibility
from src.pipeline import Pipeline
from src.vision.evaluate_localiser import predict_scene
from src.vision.train_localiser import app_instances

FOLDER_TO_CLASS = {"sound": "sound", "cracked": "seed_coat_cracked", "broken": "fragment"}
PAD = 4


def crop_box(image: np.ndarray, box: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = box
    h, w = image.shape[:2]
    return Image.fromarray(image[max(0, y0 - PAD):min(h, y1 + PAD), max(0, x0 - PAD):min(w, x1 + PAD)])


def sound_boxes(pipeline: Pipeline, image: np.ndarray) -> list[tuple[int, int, int, int]]:
    with torch.no_grad():
        inst = app_instances(predict_scene(pipeline.localiser, image, pipeline.device))
    labels, areas = np.unique(inst[inst > 0], return_counts=True)
    median = np.median(areas)
    boxes = []
    for label, area in zip(labels, areas):
        if area > 1.7 * median:
            continue
        ys, xs = np.nonzero(inst == label)
        if ys.min() <= 1 or xs.min() <= 1 or ys.max() >= inst.shape[0] - 2 or xs.max() >= inst.shape[1] - 2:
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--photos", default=str(Path.home() / "Desktop/Classifier Photos"))
    p.add_argument("--crops", default="data/interim/real_crops")
    p.add_argument("--results", default="models/real-photo-baseline.json")
    args = p.parse_args()

    pipeline = Pipeline()
    root, out = Path(args.photos), Path(args.crops)
    rows = []
    for folder, true_class in FOLDER_TO_CLASS.items():
        for batch in sorted((root / folder).glob("batch*")):
            photos = sorted(batch.glob("*.jpg"))
            if not photos:
                continue
            labels_path = batch / "kernel_labels.json"
            labels = json.loads(labels_path.read_text()) if labels_path.exists() else None
            independent = labels is None or labels.get("independent", True)
            for photo in photos:
                raw = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
                image = np.array(to_working_scale(raw))
                scale = image.shape[1] / raw.width
                if labels is None:
                    boxes = sound_boxes(pipeline, image)
                else:
                    entry = labels["photos"].get(photo.name)
                    if entry is None:
                        continue
                    boxes = [tuple(round(v * scale) for v in b) for b in entry[folder]]
                target = out / folder / batch.name
                target.mkdir(parents=True, exist_ok=True)
                crops = []
                for i, box in enumerate(boxes):
                    crop = crop_box(image, box)
                    crop.save(target / f"{photo.stem}_{i:03d}.png")
                    crops.append(pipeline.transform(crop))
                if not independent or not crops:
                    continue
                with torch.no_grad():
                    for start in range(0, len(crops), 128):
                        batch_x = torch.stack(crops[start:start + 128]).to(pipeline.device)
                        for idx in pipeline.classifier(batch_x).argmax(1).cpu().tolist():
                            rows.append({"true": true_class, "pred": pipeline.classes[idx],
                                         "batch": f"{folder}/{batch.name}", "photo": photo.name})
            print(folder, batch.name, "independent" if independent else "training only", flush=True)

    classes = list(FOLDER_TO_CLASS.values())
    preds = sorted({r["pred"] for r in rows})
    matrix = {t: {p_: sum(1 for r in rows if r["true"] == t and r["pred"] == p_) for p_ in preds} for t in classes}
    mechanical = {c for c, k in CLASSES.items() if k.admissibility == Admissibility.MECHANICAL}
    damaged_true = [r for r in rows if r["true"] != "sound"]
    sound_true = [r for r in rows if r["true"] == "sound"]
    summary = {
        "kernels_scored": len(rows),
        "per_class_recall": {t: matrix[t].get(t, 0) / max(1, sum(matrix[t].values())) for t in classes},
        "damaged_detected": sum(r["pred"] in mechanical for r in damaged_true) / max(1, len(damaged_true)),
        "sound_called_damaged": sum(r["pred"] in mechanical for r in sound_true) / max(1, len(sound_true)),
        "confusion": matrix,
    }
    Path(args.results).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
