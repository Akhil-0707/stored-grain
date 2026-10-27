"""Train the kernel localiser on synthetic bulk scenes.

Scenes are pre-generated rather than composed on the fly: composition costs
about a third of a second, which would starve the GPU, and a fixed scene set
makes runs comparable to each other.

Split discipline carries over from the classifier. A scene is built only from
kernels belonging to one split, so a kernel whose image trained the localiser
never reappears inside a validation scene.

With ``--real`` it instead fine-tunes on auto-labelled phone photographs
(``src.vision.real_labels``), optionally mixed with synthetic scenes resized to
the real kernel size. Real photos are split by tray, never by photo: the shots
of one tray are near-identical bursts, so a photo-level split would leak.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.data.manifest import build, load_grainset
from src.vision.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.vision.localise import (
    UNet,
    count_error,
    instance_f1,
    instances_from_logits,
    instance_targets,
)
from src.vision.synth import build_cutout_pool, compose, pool_for_damage


def generate(pool, n_scenes: int, n_kernels: int, seed: int, out: Path) -> list[Path]:
    """Write scenes and their instance maps to disk, skipping any already there."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    rng = random.Random(seed)
    for i in range(n_scenes):
        image_path = out / f"scene_{i:04d}.png"
        inst_path = out / f"scene_{i:04d}_inst.npy"
        if not (image_path.exists() and inst_path.exists()):
            damage = rng.choice([0, 2, 5, 10, 20, 40])
            selection = pool_for_damage(pool, damage, n_kernels, seed=seed * 1000 + i)
            scene = compose(selection, seed=seed * 1000 + i)
            scene.save(image_path)
            np.save(inst_path, scene.instances)
        paths.append(image_path)
    return paths


def load_instances(image_path: Path) -> np.ndarray:
    """The instance map beside a scene: ``_inst.npy`` (synthetic) or 16-bit ``_inst.png`` (real)."""
    from PIL import Image

    npy = image_path.with_name(image_path.stem + "_inst.npy")
    if npy.exists():
        return np.load(npy)
    return np.array(Image.open(image_path.with_name(image_path.stem + "_inst.png"))).astype(np.int32)


class SceneDataset(Dataset):
    """Random crops from pre-generated scenes.

    ``scale`` resizes each scene before cropping, so synthetic scenes composed
    at one magnification can be mixed with real photographs at another.
    ``repeats`` draws several crops per scene per epoch, for large photographs.
    """

    def __init__(self, paths: list[Path], crop: int = 384, augment: bool = True,
                 scale: float = 1.0, repeats: int = 1):
        self.paths = paths
        self.crop = crop
        self.augment = augment
        self.scale = scale
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.paths) * self.repeats

    def __getitem__(self, i: int):
        from PIL import Image

        path = self.paths[i % len(self.paths)]
        image = Image.open(path).convert("RGB")
        instances = load_instances(path)
        if self.scale != 1.0:
            size = (round(image.width * self.scale), round(image.height * self.scale))
            image = image.resize(size, Image.LANCZOS)
            instances = np.array(
                Image.fromarray(instances.astype(np.int32)).resize(size, Image.NEAREST)
            )
        image = np.array(image)

        h, w = instances.shape
        c = min(self.crop, h, w)
        top = random.randint(0, h - c)
        left = random.randint(0, w - c)
        image = image[top : top + c, left : left + c]
        instances = instances[top : top + c, left : left + c]

        if self.augment:
            k = random.randint(0, 3)
            image = np.rot90(image, k, axes=(0, 1)).copy()
            instances = np.rot90(instances, k).copy()
            if random.random() < 0.5:
                image = image[:, ::-1].copy()
                instances = instances[:, ::-1].copy()

        target = instance_targets(instances)

        x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        x = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(
            IMAGENET_STD
        )[:, None, None]
        return x, torch.from_numpy(target), torch.from_numpy(instances.astype(np.int32))


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, dataset, dev, limit: int = 24) -> dict:
    model.eval()
    counts, f1s, relative = [], [], []
    for i in range(min(limit, len(dataset))):
        x, _, instances = dataset[i]
        logits = model(x.unsqueeze(0).to(dev))[0]
        predicted = instances_from_logits(logits)
        truth = instances.numpy()
        ce = count_error(predicted, truth)
        counts.append(abs(ce["error"]))
        relative.append(abs(ce["relative_error"]))
        f1s.append(instance_f1(predicted, truth)["f1"])
    return {
        "mean_abs_count_error": float(np.mean(counts)),
        "mean_abs_relative_count_error": float(np.mean(relative)),
        "instance_f1": float(np.mean(f1s)),
    }


def app_instances(logits: torch.Tensor) -> np.ndarray:
    """Instances exactly as the app counts them: watershed, then the pipeline's
    rule dropping blobs below a fraction of the median kernel area."""
    from src.pipeline import MIN_RELATIVE_AREA

    instances = instances_from_logits(logits)
    labels, areas = np.unique(instances[instances > 0], return_counts=True)
    if len(labels):
        instances[np.isin(instances, labels[areas < np.median(areas) * MIN_RELATIVE_AREA])] = 0
    return instances


@torch.no_grad()
def evaluate_scenes(model, paths: list[Path], dev) -> dict:
    """Score whole scenes against their instance maps, as the app sees them.

    For real photographs the instance maps are auto-labels, so these numbers
    measure agreement with the labels, not accuracy against a hand count.
    """
    from PIL import Image

    from src.vision.evaluate_localiser import predict_scene

    model.eval()
    counts, relative, f1s = [], [], []
    for path in paths:
        image = np.array(Image.open(path).convert("RGB"))
        truth = load_instances(path)
        predicted = app_instances(predict_scene(model, image, dev))
        ce = count_error(predicted, truth)
        counts.append(abs(ce["error"]))
        relative.append(abs(ce["relative_error"]))
        f1s.append(instance_f1(predicted, truth)["f1"])
    return {
        "mean_abs_count_error": float(np.mean(counts)),
        "mean_abs_relative_count_error": float(np.mean(relative)),
        "max_abs_relative_count_error": float(np.max(relative)),
        "instance_f1": float(np.mean(f1s)),
    }


def real_scene_paths(root: Path, groups: set[str] | None = None,
                     exclude: set[str] = frozenset()) -> list[Path]:
    """Labelled real photos under ``root/<group>/``, filtered by group (tray) name."""
    paths = []
    for inst in sorted(root.rglob("*_inst.png")):
        group = inst.parent.relative_to(root).as_posix()
        if group in exclude or (groups is not None and group not in groups):
            continue
        paths.append(inst.with_name(inst.name[: -len("_inst.png")] + ".png"))
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grainset", default="data/raw/grainset/maize")
    p.add_argument("--scenes", default="data/interim/scenes")
    p.add_argument("--train-scenes", type=int, default=300)
    p.add_argument("--val-scenes", type=int, default=40)
    p.add_argument("--kernels", type=int, default=200)
    p.add_argument("--pool", type=int, default=600)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop", type=int, default=384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--checkpoint", default="models/localiser.pt")
    p.add_argument("--results", default="models/localiser-results.json")
    # Fine-tuning on real photographs (all optional; omit to train on synthetic scenes only).
    p.add_argument("--real", help="root of auto-labelled real photos, one folder per tray")
    p.add_argument("--val-groups", default="", help="comma-separated trays used for validation")
    p.add_argument("--test-groups", default="", help="comma-separated trays never used here")
    p.add_argument("--real-repeats", type=int, default=8, help="crops per real photo per epoch")
    p.add_argument("--synthetic-scale", type=float, default=1.0,
                   help="resize synthetic scenes to the real kernel size (e.g. 0.5)")
    p.add_argument("--synthetic-limit", type=int, default=0, help="use only this many synthetic scenes")
    p.add_argument("--init", help="checkpoint to fine-tune from")
    p.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    p.add_argument("--history", help="write per-epoch metrics here (JSON)")
    args = p.parse_args()

    dev = device()
    print(f"device: {dev}", flush=True)
    val_set = None
    real_val: list[Path] = []
    datasets = []

    if args.real:
        real_root = Path(args.real)
        val_groups = {g for g in args.val_groups.split(",") if g}
        test_groups = {g for g in args.test_groups.split(",") if g}
        real_train = real_scene_paths(real_root, exclude=val_groups | test_groups)
        real_val = real_scene_paths(real_root, groups=val_groups)
        print(f"real photos: train {len(real_train)} | val {len(real_val)} "
              f"| held out for test: {sorted(test_groups)}", flush=True)
        datasets.append(SceneDataset(real_train, args.crop, augment=True, repeats=args.real_repeats))
        synthetic = sorted(Path(args.scenes).glob("*.png"))
        synthetic = [s for s in synthetic if not s.stem.endswith("_inst")]
        if args.synthetic_limit:
            synthetic = synthetic[: args.synthetic_limit]
        if synthetic:
            datasets.append(SceneDataset(synthetic, args.crop, augment=True, scale=args.synthetic_scale))
        print(f"synthetic scenes mixed in: {len(synthetic)} (scale {args.synthetic_scale})", flush=True)
    else:
        manifest = build(load_grainset(Path(args.grainset)))
        root = Path(args.scenes)

        print("building cutout pools", flush=True)
        train_pool = build_cutout_pool(manifest.split("train"), limit=args.pool, seed=1)
        val_pool = build_cutout_pool(manifest.split("val"), limit=args.pool // 3, seed=2)
        print(f"pools: train {len(train_pool)} | val {len(val_pool)}", flush=True)

        started = time.time()
        train_paths = generate(train_pool, args.train_scenes, args.kernels, 1, root / "train")
        val_paths = generate(val_pool, args.val_scenes, args.kernels, 2, root / "val")
        print(
            f"scenes: train {len(train_paths)} | val {len(val_paths)} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )
        datasets.append(SceneDataset(train_paths, args.crop, augment=True))
        val_set = SceneDataset(val_paths, args.crop, augment=False)

    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    model = UNet().to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev)["model"])
        print(f"fine-tuning from {args.init}", flush=True)
    # The border class is a thin rim and therefore rare in pixel terms, but it
    # is the entire mechanism for separating touching kernels. Left unweighted,
    # the network learns to skip it and merges neighbours.
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 3.0]).to(dev))
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    use_amp = args.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    if real_val:
        start = evaluate_scenes(model, real_val, dev)
        history.append({"epoch": 0, "loss": None, **start})
        print(f"before training: instance-F1 {start['instance_f1']:.4f} "
              f"count error {start['mean_abs_relative_count_error'] * 100:.1f}%", flush=True)

    best = {"instance_f1": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, target, _ in loader:
            x, target = x.to(dev), target.to(dev)
            optimiser.zero_grad()
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x), target)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        schedule.step()

        metrics = evaluate_scenes(model, real_val, dev) if real_val else evaluate(model, val_set, dev)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "loss": running / seen, "seconds": elapsed, **metrics})
        print(
            f"epoch {epoch}/{args.epochs} loss {running / seen:.4f} "
            f"instance-F1 {metrics['instance_f1']:.4f} "
            f"count error {metrics['mean_abs_count_error']:.1f} kernels "
            f"({metrics['mean_abs_relative_count_error'] * 100:.1f}%) "
            f"({elapsed:.0f}s)",
            flush=True,
        )
        if metrics["instance_f1"] > best["instance_f1"]:
            best = {**metrics, "epoch": epoch}
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), **best}, args.checkpoint)
        if args.history:
            Path(args.history).write_text(json.dumps(history, indent=2))

    Path(args.results).write_text(json.dumps(best, indent=2))
    print(f"best: {best}", flush=True)


if __name__ == "__main__":
    main()
