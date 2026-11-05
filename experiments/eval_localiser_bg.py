"""Score localisers on the held-out real trays, unseen surfaces, damaged kernels and GrainSet scenes.

Kernel-only scoring: impurities are dropped from the truth by the pipeline's
debris rule, as the app drops them from its count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.vision.localise import UNet
from src.vision.train_localiser import device, evaluate_scenes, real_scene_paths

MIXED = Path("data/interim/mixed")
SETS = {
    "held_out_real_trays": lambda: real_scene_paths(MIXED, groups={"tray03", "tray13", "tray32", "tray47"}),
    "unseen_surfaces": lambda: real_scene_paths(MIXED, groups={"bgval000", "bgval001", "bgval002", "bgval003"}),
    "damaged_kernels_unseen_surfaces": lambda: real_scene_paths(MIXED, groups={"synval000", "synval001"}),
    "grainset_val_scenes": lambda: sorted(p for p in Path("data/interim/scenes/val").glob("*.png")
                                          if not p.stem.endswith("_inst")),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", default=["models/localiser-real.pt", "models/localiser-bg2.pt"])
    p.add_argument("--results", default="models/localiser-bg2-eval.json")
    args = p.parse_args()
    dev = device()
    out = {}
    for ck in args.checkpoints:
        model = UNet()
        model.load_state_dict(torch.load(ck, map_location="cpu")["model"])
        model.to(dev)
        for name, paths in SETS.items():
            scenes = paths()
            r = evaluate_scenes(model, scenes, dev)
            out[f"{Path(ck).stem}|{name}"] = {"n": len(scenes), **r}
            print(f"{Path(ck).stem:16s} {name:32s} n={len(scenes):3d} count err "
                  f"{100 * r['mean_abs_relative_count_error']:5.1f}% F1 {r['instance_f1']:.3f}", flush=True)
    Path(args.results).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
