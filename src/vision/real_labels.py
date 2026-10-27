"""Auto-label real tray photographs for localiser training.

There are no hand-drawn masks for the phone photographs, so the labels are
built from two weak sources that fail in different places:

* The **foreground** comes from the tray itself. Kernels on the matte black
  card are several times brighter than the card, so an Otsu threshold on
  brightness gives a clean kernel/background split with no learning involved.
* The **split between touching kernels** comes from the synthetic-trained
  localiser. On phone photographs it invents kernels on the empty card, but
  inside the foreground its interior predictions separate neighbours well.

Restricting the localiser's interiors to the threshold foreground removes the
invented kernels; a second pass splits blobs still much larger than a typical
kernel by distance-transform watershed. These are pseudo-labels: a model
trained on them is scored against them only as *agreement*, never as accuracy.

Each photo is written at the app's working scale as ``<name>.png`` with its
instance map as a 16-bit ``<name>_inst.png``; ``train_localiser.SceneDataset``
reads that alongside the synthetic scenes' ``_inst.npy``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

#: A blob this many times the median kernel area is treated as merged kernels.
MERGED_AREA_FACTOR = 1.7
#: Blobs below this fraction of the median kernel area are dropped as debris.
DEBRIS_AREA_FACTOR = 0.15
#: Holes in the foreground below this fraction of a kernel's area are shading
#: inside a kernel and are filled. Larger holes are card showing between
#: touching kernels and stay background.
MAX_HOLE_FACTOR = 0.05


def typical_kernel_area(mask: np.ndarray) -> float:
    """Lower quartile of foreground island areas, ignoring specks.

    Islands of touching kernels only push areas up, so the lower quartile stays
    close to a single kernel even on dense trays.
    """
    islands, n = ndimage.label(mask)
    if not n:
        return 0.0
    areas = ndimage.sum(mask, islands, np.arange(1, n + 1))
    areas = areas[areas >= 50]
    return float(np.percentile(areas, 25)) if len(areas) else 0.0


def foreground(image: np.ndarray) -> np.ndarray:
    """Kernel pixels against the dark card, by Otsu on smoothed brightness.

    Only small holes are filled: filling every hole would also fill the card
    enclosed by a ring of touching kernels, which then gets labelled as kernel.
    """
    from skimage.filters import threshold_otsu

    value = np.asarray(Image.fromarray(image).convert("HSV"), dtype=np.float32)[..., 2]
    value = ndimage.gaussian_filter(value, 1.0)
    mask = value > threshold_otsu(value)
    mask = ndimage.binary_opening(mask, iterations=1)
    holes, n = ndimage.label(ndimage.binary_fill_holes(mask) & ~mask)
    if n:
        areas = ndimage.sum(holes > 0, holes, np.arange(1, n + 1))
        small = 1 + np.nonzero(areas < MAX_HOLE_FACTOR * typical_kernel_area(mask))[0]
        mask = mask | np.isin(holes, small)
    return mask


def interiors_to_instances(interior_prob: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Seed one instance per predicted interior inside the foreground, then grow."""
    from skimage.segmentation import watershed

    seeds = (interior_prob > 0.5) & ndimage.binary_erosion(fg, iterations=2)
    markers, n = ndimage.label(seeds)
    if n:
        sizes = ndimage.sum(seeds, markers, np.arange(1, n + 1))
        markers[np.isin(markers, 1 + np.nonzero(sizes < 25)[0])] = 0
        markers, _ = ndimage.label(markers > 0)
    # Foreground islands with no seed at all (isolated kernels the network
    # missed) become their own instance rather than being lost.
    islands, _ = ndimage.label(fg)
    seeded = np.unique(islands[markers > 0])
    orphan = fg & ~np.isin(islands, seeded)
    if orphan.any():
        extra, _ = ndimage.label(orphan)
        markers = np.where(extra > 0, extra + markers.max(), markers)
    return watershed(-interior_prob, markers, mask=fg)


def split_merged(instances: np.ndarray) -> np.ndarray:
    """Split blobs far larger than a typical kernel by distance-transform watershed."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    labels, areas = np.unique(instances[instances > 0], return_counts=True)
    if not len(labels):
        return instances
    median = float(np.median(areas))
    radius = np.sqrt(median / np.pi)
    out = instances.copy()
    next_id = int(out.max()) + 1
    for label, area in zip(labels, areas):
        if area < MERGED_AREA_FACTOR * median:
            continue
        blob = out == label
        dist = ndimage.distance_transform_edt(blob)
        peaks = peak_local_max(dist, min_distance=max(3, int(radius * 0.6)),
                               threshold_abs=radius * 0.45, labels=blob.astype(int))
        if len(peaks) < 2:
            continue
        markers = np.zeros(blob.shape, np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        parts = watershed(-dist, markers, mask=blob)
        for k in range(2, len(peaks) + 1):
            out[parts == k] = next_id
            next_id += 1
    return out


def drop_debris(instances: np.ndarray) -> np.ndarray:
    labels, areas = np.unique(instances[instances > 0], return_counts=True)
    if not len(labels):
        return instances
    small = labels[areas < DEBRIS_AREA_FACTOR * np.median(areas)]
    out = instances.copy()
    out[np.isin(out, small)] = 0
    # Renumber densely so instance ids are 1..n.
    ids = np.unique(out[out > 0])
    lut = np.zeros(int(out.max()) + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[out]


def save_instances(path: Path, instances: np.ndarray) -> None:
    """Store an instance map as a lossless 16-bit PNG (~50 KB instead of ~6 MB as .npy)."""
    if instances.max() > np.iinfo(np.uint16).max:
        raise ValueError("more instances than a 16-bit map can hold")
    Image.fromarray(instances.astype(np.uint16)).save(path)


def label_image(image: np.ndarray, interior_prob: np.ndarray) -> np.ndarray:
    fg = foreground(image)
    instances = interiors_to_instances(interior_prob, fg)
    return drop_debris(split_merged(instances))


def main() -> None:
    import torch

    from src.app.images import decode, to_working_scale
    from src.pipeline import predict_scene
    from src.vision.localise import UNet
    from src.vision.train_localiser import device

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--photos", required=True, help="root with one folder per tray")
    p.add_argument("--out", required=True, help="where labelled scenes are written")
    p.add_argument("--checkpoint", default="models/localiser.pt")
    args = p.parse_args()

    dev = device()
    model = UNet().to(dev)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev)["model"])
    model.eval()

    root, out = Path(args.photos), Path(args.out)
    summary = []
    for photo in sorted(root.rglob("*")):
        if photo.suffix.lower() not in {".jpg", ".jpeg", ".png"} or photo.name.startswith("._"):
            continue
        group = photo.parent.relative_to(root).as_posix()
        image = np.asarray(to_working_scale(decode(photo.read_bytes())).convert("RGB"))
        with torch.no_grad():
            probs = torch.softmax(predict_scene(model, image, dev), 0).cpu().numpy()
        instances = label_image(image, probs[1])
        dest = out / group
        dest.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(dest / f"{photo.stem}.png")
        save_instances(dest / f"{photo.stem}_inst.png", instances)
        areas = np.unique(instances[instances > 0], return_counts=True)[1]
        summary.append({"group": group, "photo": photo.name, "kernels": int(len(areas)),
                        "median_area_px": int(np.median(areas)) if len(areas) else 0})
        print(f"{group}/{photo.name}: {summary[-1]['kernels']} kernels", flush=True)
    (out / "labels.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
