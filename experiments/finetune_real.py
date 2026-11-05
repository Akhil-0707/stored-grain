"""Fine-tune the classifier on the team's real phone crops, with a two-fold batch swap.

Real crops come from ``experiments/real_photo_baseline.py`` and are grouped by
physical batch of kernels. Each fold trains on some batches and tests on batches
of different kernels, so the test score is on kernels the model never saw.
``cracked/batch3`` re-samples kernels of batches 1 and 2 -- the same crops, not
new kernels -- so it is used nowhere: it overlaps every test batch, and in the
final model it only tripled the weight of cracked examples.

GrainSet kernels cut out of synthetic trays (``experiments/grainset_tray_crops``)
join every epoch. Without them the only crop-resolution images were popcorn, the
model learned "pale crop means cracked", and dent-corn trays read 25% damaged
with no damage in them.

Every epoch draws a fresh random subset of the public training set alongside all
real crops repeated, so the phone photos are not drowned out by 23k laboratory
images, and the model is still checked on the public validation split at the end
so a gain on phones is not bought by forgetting GrainSet.

The epoch count is fixed in advance and the last epoch is kept: choosing the
epoch on the real test batches would leak them into model selection.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.data.manifest import build, by_session, load_cornseeds, load_grainset
from src.mapping.classes import CLASSES, Admissibility
from src.vision.dataset import CLASS_INDEX, CLASS_NAMES, KernelDataset, eval_transform, train_transform
from src.vision.train import build_model, device, evaluate

FOLDER_TO_CLASS = {"sound": "sound", "cracked": "seed_coat_cracked", "broken": "fragment"}
FOLDS = {
    "A": {"train": ["sound/batch1", "sound/batch3", "cracked/batch1", "broken/batch1"],
          "test": ["sound/batch2", "cracked/batch2", "broken/batch2"]},
    "B": {"train": ["sound/batch2", "sound/batch3", "cracked/batch2", "broken/batch2"],
          "test": ["sound/batch1", "cracked/batch1", "broken/batch1"]},
    "final": {"train": ["sound/batch1", "sound/batch2", "sound/batch3", "cracked/batch1", "cracked/batch2",
                        "broken/batch1", "broken/batch2"], "test": []},
}
MECHANICAL = {c for c, k in CLASSES.items() if k.admissibility == Admissibility.MECHANICAL}


class TrayCrops(Dataset):
    def __init__(self, items: list[tuple[Path, int]], transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label, 0


class RealCrops(Dataset):
    def __init__(self, root: Path, batches: list[str], transform, repeats: int = 1):
        self.items = [(p, CLASS_INDEX[FOLDER_TO_CLASS[b.split("/")[0]]])
                      for b in batches for p in sorted((root / b).glob("*.png"))] * repeats
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label, 2


@torch.no_grad()
def score_real(model, loader, dev) -> dict:
    model.eval()
    rows = []
    for x, y, _ in loader:
        for t, p in zip(y.tolist(), model(x.to(dev)).argmax(1).cpu().tolist()):
            rows.append((CLASS_NAMES[t], CLASS_NAMES[p]))
    truths = sorted({t for t, _ in rows})
    preds = sorted({p for _, p in rows} | set(truths))
    confusion = {t: {p: sum(1 for a, b in rows if a == t and b == p) for p in preds} for t in truths}
    damaged = [p for t, p in rows if t != "sound"]
    sound = [p for t, p in rows if t == "sound"]
    return {
        "n": len(rows),
        "accuracy": sum(t == p for t, p in rows) / len(rows),
        "recall": {t: confusion[t][t] / sum(confusion[t].values()) for t in truths},
        "damaged_detected": sum(p in MECHANICAL for p in damaged) / len(damaged),
        "sound_called_damaged": sum(p in MECHANICAL for p in sound) / len(sound),
        "confusion": confusion,
    }


def run_fold(name: str, args, public_train, public_val) -> dict:
    fold = FOLDS[name]
    root = Path(args.crops)
    dev = device()
    model = build_model().to(dev)
    model.load_state_dict(torch.load(args.init, map_location=dev)["model"])

    real_train = RealCrops(root, fold["train"], train_transform(), repeats=args.real_repeats)
    test_loader = (DataLoader(RealCrops(root, fold["test"], eval_transform()), batch_size=64, num_workers=args.workers)
                   if fold["test"] else None)
    counts = torch.zeros(len(CLASS_NAMES))
    for _, label in real_train.items:
        counts[label] += 1
    for s in public_train:
        counts[CLASS_INDEX[s.class_name]] += 1
    tray_items = [(p_, CLASS_INDEX[p_.parent.name]) for p_ in sorted(Path(args.tray_crops).glob("*/*.png"))]
    for _, label in tray_items:
        counts[label] += 1
    weights = torch.where(counts > 0, counts.sum() / (counts * len(CLASS_NAMES)), 0.0)
    criterion = nn.CrossEntropyLoss(weight=weights.to(dev))
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    history = []
    if test_loader:
        before = score_real(model, test_loader, dev)
        history.append({"epoch": 0, **{k: before[k] for k in ("accuracy", "recall", "damaged_detected",
                                                                "sound_called_damaged")}})
        print(f"[{name}] before: acc {before['accuracy']:.3f} damaged {before['damaged_detected']:.3f} "
              f"recall {json.dumps({k: round(v, 3) for k, v in before['recall'].items()})}", flush=True)
    rng = random.Random(0)
    for epoch in range(1, args.epochs + 1):
        subset = rng.sample(public_train, args.public_per_epoch)
        trays = TrayCrops(rng.sample(tray_items, min(args.tray_per_epoch, len(tray_items))), train_transform())
        loader = DataLoader(ConcatDataset([KernelDataset(subset, train_transform(), train=True), real_train, trays]),
                            batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        model.train()
        started, running, seen = time.time(), 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(dev), y.to(dev)
            optimiser.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimiser.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        schedule.step()
        line = f"[{name}] epoch {epoch}/{args.epochs} loss {running / seen:.4f}"
        if test_loader:
            m = score_real(model, test_loader, dev)
            history.append({"epoch": epoch, "loss": running / seen, **{k: m[k] for k in (
                "accuracy", "recall", "damaged_detected", "sound_called_damaged")}})
            line += (f" | real test acc {m['accuracy']:.3f} damaged {m['damaged_detected']:.3f} "
                     f"sound->damaged {m['sound_called_damaged']:.3f}")
        print(line + f" ({time.time() - started:.0f}s)", flush=True)

    public = evaluate(model, DataLoader(KernelDataset(public_val, eval_transform()), batch_size=64,
                                        num_workers=args.workers), dev)
    print(f"[{name}] public val macro-F1 {public['macro_f1']:.4f} acc {public['accuracy']:.4f}", flush=True)
    result = {"fold": name, "train": fold["train"], "test": fold["test"], "history": history,
              "public_val": public, "before": before if test_loader else None,
              "after": score_real(model, test_loader, dev) if test_loader else None}
    torch.save({"model": model.state_dict(), "classes": CLASS_NAMES, "single_view": False,
                "fine_tuned_on_real": fold["train"], "macro_f1": public["macro_f1"]},
               f"{args.checkpoint}.{name}.pt")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crops", default="data/interim/real_crops")
    p.add_argument("--init", default="models/classifier.session.pt")
    p.add_argument("--checkpoint", default="models/classifier-real")
    p.add_argument("--results", default="models/classifier-real-results.json")
    p.add_argument("--folds", default="A,B,final")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--public-per-epoch", type=int, default=6000)
    p.add_argument("--real-repeats", type=int, default=3)
    p.add_argument("--tray-crops", default="data/interim/grainset_tray_crops")
    p.add_argument("--tray-per-epoch", type=int, default=4000)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    manifest = build(load_grainset(Path("data/raw/grainset/maize")) + load_cornseeds(Path("data/raw/kaggle/cornseeds/corn")),
                     key=by_session)
    public_train, public_val = manifest.split("train"), manifest.split("val")
    results = {}
    for name in args.folds.split(","):
        results[name] = run_fold(name, args, public_train, public_val)
        Path(args.results).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
