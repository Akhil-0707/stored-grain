"""Score a classifier checkpoint once on the public session-split test set (GrainSet + Corn Seeds)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.manifest import build, by_session, load_cornseeds, load_grainset
from src.vision.dataset import CLASS_NAMES, KernelDataset, eval_transform
from src.vision.train import build_model, device, macro_f1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="models/classifier-real2.final.pt")
    p.add_argument("--results", default="models/compare/app_model_test_real2.json")
    args = p.parse_args()
    dev = device()
    model = build_model().to(dev)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev)["model"])
    model.eval()
    test = build(load_grainset(Path("data/raw/grainset/maize")) + load_cornseeds(Path("data/raw/kaggle/cornseeds/corn")),
                 key=by_session).split("test")
    cm = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.long)
    started = time.time()
    with torch.no_grad():
        for x, y, _ in DataLoader(KernelDataset(test, eval_transform()), batch_size=64, num_workers=8):
            for t, pr in zip(y, model(x.to(dev)).argmax(1).cpu()):
                cm[t, pr] += 1
    f1, per = macro_f1(cm)
    out = {"checkpoint": Path(args.checkpoint).name, "macro_f1": f1, "accuracy": cm.diag().sum().item() / cm.sum().item(),
           "per_class_f1": per,
           "per_class_recall": {c: (cm[i, i] / cm[i].sum()).item() for i, c in enumerate(CLASS_NAMES)},
           "confusion": cm.tolist(), "n": cm.sum().item(),
           "ms_per_kernel": 1000 * (time.time() - started) / cm.sum().item()}
    Path(args.results).write_text(json.dumps(out, indent=1))
    print(f"macro-F1 {f1:.4f} accuracy {out['accuracy']:.4f}")


if __name__ == "__main__":
    main()
