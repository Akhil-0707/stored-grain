"""Damage reported by the whole pipeline on whole real photographs of one-class batches.

Uses batch 1 of each class, which the fold-B classifier never saw, and every kernel the
localiser finds -- not only the hand-checked ones -- so the number is what the app shows.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

from src.app.images import to_working_scale
from src.pipeline import Pipeline


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--photos", default=str(Path.home() / "Desktop/Classifier Photos"))
    p.add_argument("--classifiers", nargs="+", default=["models/classifier.session.pt", "models/classifier-real2.B.pt"])
    p.add_argument("--results", default="models/real-photo-e2e.json")
    args = p.parse_args()
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "w.png"
        for ck in args.classifiers:
            pipeline = Pipeline(classifier_checkpoint=ck)
            for folder in ("sound", "cracked", "broken"):
                rows = []
                for photo in sorted((Path(args.photos) / folder / "batch1").glob("*.jpg")):
                    to_working_scale(ImageOps.exif_transpose(Image.open(photo)).convert("RGB")).save(work)
                    r = pipeline.read(work, lot_id=photo.stem, temperature_c=20, moisture_pct_wb=14)
                    rows.append({"photo": photo.name, "kernels": r.kernels_counted,
                                 "damage_count_pct": r.damage_count_pct, "damage_mass_pct": r.damage_mass_pct})
                out[f"{Path(ck).stem}|{folder}"] = rows
                print(Path(ck).stem, folder, [round(x["damage_count_pct"], 1) for x in rows], flush=True)
    Path(args.results).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
