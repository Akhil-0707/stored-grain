"""Compare kernel classifiers under one recipe, and keep the confusion matrices.

Every model sees the same corpus, the same session split, the same single-view
framing the pipeline uses, the same augmentation and the same class weights.
The checkpoint is chosen on the validation split; the number reported is the
held-out test split, which nothing here ever trains or selects on. Under those
rules a difference between two rows is a difference between the models, not
between their setups.

Two classical baselines (hand-made colour and shape features fed to an SVM and
a random forest) sit alongside the networks. They answer the question a
reviewer asks first: does this task need deep learning at all?

    python -m experiments.compare_models            # train what is missing
    python -m experiments.compare_models --plot     # redraw figures only
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms

from src.data.manifest import build, by_session, load_cornseeds, load_grainset
from src.vision.dataset import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    KernelDataset,
    class_weights,
    eval_transform,
    train_transform,
)
from src.vision.train import device, macro_f1

OUT = Path("models/compare")

#: Display names, in the order the figures list them.
LABELS = {
    "svm": "SVM (HOG + colour)",
    "random_forest": "Random forest (HOG + colour)",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet-50 (current)",
    "yolo11m_cls": "YOLO11m-cls",
    "densenet121": "DenseNet-121",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_b_16": "ViT-B/16",
}
CLASSICAL = ("svm", "random_forest")

#: Transformer-style models are conventionally fine-tuned at a lower rate;
#: 3e-4 makes ViT unstable, which would be measuring the recipe, not the model.
LR = {"vit_b_16": 1e-4, "convnext_tiny": 1e-4}


class YoloClassifier(nn.Module):
    """YOLO11 classification backbone behind the same interface as the rest.

    YOLO expects pixels in 0..1, not ImageNet-normalised, so the shared
    normalisation is undone on the way in. In eval mode its head returns
    (probabilities, logits); the logits are taken so every model is scored on
    the same kind of output.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        from ultralytics import YOLO

        self.net = YOLO(str(OUT / "yolo11m-cls.pt")).model
        head = self.net.model[-1]
        head.linear = nn.Linear(head.linear.in_features, n_classes)
        for p in self.net.parameters():
            p.requires_grad = True
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        out = self.net(x * self.std + self.mean)
        return out[1] if isinstance(out, tuple) else out


def build_network(name: str, n: int = len(CLASS_NAMES)) -> nn.Module:
    if name == "yolo11m_cls":
        return YoloClassifier(n)
    m = getattr(models, name)(weights="DEFAULT")
    if name == "resnet50":
        m.fc = nn.Linear(m.fc.in_features, n)
    elif name == "densenet121":
        m.classifier = nn.Linear(m.classifier.in_features, n)
    elif name == "vit_b_16":
        m.heads.head = nn.Linear(m.heads.head.in_features, n)
    else:   # efficientnet, mobilenet, convnext: last layer of .classifier
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
    return m


def score(confusion: torch.Tensor) -> dict:
    f1, per_class = macro_f1(confusion)
    total = confusion.sum().item()
    recall = {
        c: (confusion[i, i] / confusion[i].sum()).item() if confusion[i].sum() else 0.0
        for i, c in enumerate(CLASS_NAMES)
    }
    return {
        "macro_f1": f1,
        "accuracy": confusion.diag().sum().item() / total,
        "per_class_f1": per_class,
        "per_class_recall": recall,
        "confusion": confusion.tolist(),
        "n": total,
    }


@torch.no_grad()
def evaluate(model, loader, dev) -> tuple[dict, float]:
    model.eval()
    n = len(CLASS_NAMES)
    confusion = torch.zeros(n, n, dtype=torch.long)
    seconds = 0.0
    for images, labels, _ in loader:
        images = images.to(dev)
        started = time.perf_counter()
        preds = model(images).argmax(1).cpu()   # .cpu() waits for the device
        seconds += time.perf_counter() - started
        for t, p in zip(labels, preds):
            confusion[t, p] += 1
    return score(confusion), 1000 * seconds / confusion.sum().item()


def train_network(name: str, splits: dict, args) -> dict:
    torch.manual_seed(0)
    loaders = {
        s: DataLoader(
            KernelDataset(
                splits[s],
                train_transform(args.size) if s == "train" else eval_transform(args.size),
                single_view=True,
                train=s == "train",
            ),
            batch_size=args.batch_size,
            shuffle=s == "train",
            num_workers=args.workers,
            persistent_workers=True,
        )
        for s in ("train", "val", "test")
    }
    dev = device()
    model = build_network(name).to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights(splits["train"]).to(dev))
    lr = LR.get(name, args.lr)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    best, best_state, history, started_all = {"macro_f1": -1.0}, None, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        for images, labels, _ in loaders["train"]:
            images, labels = images.to(dev), labels.to(dev)
            optimiser.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimiser.step()
            running += loss.item() * images.size(0)
        schedule.step()
        val, _ = evaluate(model, loaders["val"], dev)
        history.append({"epoch": epoch, "loss": running / len(splits["train"]),
                        "val_macro_f1": val["macro_f1"]})
        print(f"[{name}] epoch {epoch}/{args.epochs} loss {history[-1]['loss']:.4f} "
              f"val macro-F1 {val['macro_f1']:.4f} ({time.time() - started:.0f}s)", flush=True)
        if val["macro_f1"] > best["macro_f1"]:
            best = {**val, "epoch": epoch}
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best_state)
    test, ms = evaluate(model, loaders["test"], dev)
    torch.save({"model": best_state, "classes": CLASS_NAMES, "single_view": True,
                "arch": name}, OUT / f"{name}.pt")
    return {
        "test": test,
        "val_macro_f1": best["macro_f1"],
        "best_epoch": best["epoch"],
        "history": history,
        "lr": lr,
        "params_millions": sum(p.numel() for p in model.parameters()) / 1e6,
        "ms_per_kernel": ms,
        "train_minutes": (time.time() - started_all) / 60,
    }


class Features:
    """Colour histogram plus HOG, the classic hand-made description of a kernel.

    Colour carries mould, heat damage and discolouration; HOG (histogram of
    oriented gradients, i.e. which way the edges point) carries shape, breaks
    and cracks.
    """

    def __init__(self, size: int = 128):
        self.resize = transforms.Resize((size, size))

    def __call__(self, image: Image.Image) -> torch.Tensor:
        from skimage.color import rgb2gray, rgb2hsv
        from skimage.feature import hog

        rgb = np.asarray(self.resize(image), dtype=np.float32) / 255.0
        hsv = rgb2hsv(rgb)
        colour = np.concatenate(
            [np.histogram(hsv[..., c], bins=16, range=(0, 1), density=True)[0] for c in range(3)]
        )
        shape = hog(rgb2gray(rgb), orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2))
        return torch.from_numpy(np.concatenate([colour, shape]).astype(np.float32))


def features_of(samples, args) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(KernelDataset(samples, Features(), single_view=True),
                        batch_size=256, num_workers=args.workers)
    xs, ys = [], []
    for x, y, _ in loader:
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def train_classical(name: str, splits: dict, args) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    cache = OUT / "features.npz"
    if cache.exists():
        d = np.load(cache)
        data = {s: (d[f"{s}_x"], d[f"{s}_y"]) for s in ("train", "test")}
    else:
        data = {s: features_of(splits[s], args) for s in ("train", "test")}
        np.savez(cache, **{f"{s}_{k}": v for s, (x, y) in data.items()
                           for k, v in (("x", x), ("y", y))})

    if name == "svm":
        model = make_pipeline(StandardScaler(), PCA(256, random_state=0),
                              SVC(C=10, class_weight="balanced", random_state=0))
    else:
        model = RandomForestClassifier(500, class_weight="balanced", n_jobs=-1, random_state=0)

    started = time.time()
    model.fit(*data["train"])
    x, y = data["test"]
    t = time.perf_counter()
    pred = model.predict(x)
    ms = 1000 * (time.perf_counter() - t) / len(y)
    n = len(CLASS_NAMES)
    confusion = torch.zeros(n, n, dtype=torch.long)
    for a, b in zip(y, pred):
        confusion[a, b] += 1
    return {"test": score(confusion), "ms_per_kernel": ms,
            "train_minutes": (time.time() - started) / 60}


def plot(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [k for k in LABELS if k in results]
    short = [c.replace("_", " ") for c in CLASS_NAMES]

    for name in names:
        cm = np.array(results[name]["test"]["confusion"], dtype=float)
        pct = 100 * cm / cm.sum(1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                if cm[i, j]:
                    ax.text(j, i, f"{int(cm[i, j])}\n{pct[i, j]:.0f}%", ha="center",
                            va="center", fontsize=7.5,
                            color="white" if pct[i, j] > 55 else "black")
        ax.set_xticks(range(len(short)), short, rotation=40, ha="right")
        ax.set_yticks(range(len(short)), short)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        f1 = results[name]["test"]["macro_f1"]
        ax.set_title(f"{LABELS[name]} — test set, macro-F1 {f1:.3f}")
        fig.tight_layout()
        fig.savefig(OUT / f"confusion_{name}.png", dpi=160)
        plt.close(fig)

    ranked = sorted(names, key=lambda k: results[k]["test"]["macro_f1"])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(ranked) + 1.2))
    vals = [results[k]["test"]["macro_f1"] for k in ranked]
    bars = ax.barh([LABELS[k] for k in ranked], vals,
                   color=["#c0392b" if k in CLASSICAL else "#2e86c1" for k in ranked])
    for b, v in zip(bars, vals):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Macro-F1 on held-out test set (higher is better)")
    fig.tight_layout()
    fig.savefig(OUT / "comparison_macro_f1.png", dpi=160)
    plt.close(fig)

    rows = ["| Model | Test macro-F1 | Accuracy | Cracked F1 | Fragment F1 | "
            "Params (M) | ms/kernel |", "|---|---|---|---|---|---|---|"]
    for k in reversed(ranked):
        r, t = results[k], results[k]["test"]
        rows.append(
            f"| {LABELS[k]} | {t['macro_f1']:.3f} | {t['accuracy']:.3f} | "
            f"{t['per_class_f1']['seed_coat_cracked']:.3f} | "
            f"{t['per_class_f1']['fragment']:.3f} | "
            f"{r.get('params_millions', float('nan')):.1f} | {r['ms_per_kernel']:.2f} |"
        )
    (OUT / "comparison.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grainset", default="data/raw/grainset/maize")
    p.add_argument("--cornseeds", default="data/raw/kaggle/cornseeds/corn")
    p.add_argument("--models", default=",".join(LABELS))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--plot", action="store_true", help="Redraw figures from results only.")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}

    if not args.plot:
        samples = load_grainset(Path(args.grainset)) + load_cornseeds(Path(args.cornseeds))
        manifest = build(samples, key=by_session)
        splits = {s: manifest.split(s) for s in ("train", "val", "test")}
        for name in args.models.split(","):
            if name in results:
                print(f"[{name}] already done, skipping", flush=True)
                continue
            fit = train_classical if name in CLASSICAL else train_network
            results[name] = fit(name, splits, args)
            t = results[name]["test"]
            print(f"[{name}] TEST macro-F1 {t['macro_f1']:.4f} acc {t['accuracy']:.4f}",
                  flush=True)
            path.write_text(json.dumps(results, indent=2))   # survive a crash mid-run

    plot(results)


if __name__ == "__main__":
    main()
