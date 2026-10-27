"""Auto-labels for real tray photos, and the training plumbing that reads them."""

import numpy as np
import pytest
import torch

from src.vision import train_localiser
from src.vision.real_labels import (
    drop_debris,
    foreground,
    interiors_to_instances,
    save_instances,
    split_merged,
)
from src.vision.train_localiser import SceneDataset, load_instances, real_scene_paths

CARD = (18, 18, 20)
KERNEL = (215, 170, 70)


def disc(shape, cy, cx, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def tray(shape=(120, 160), centres=((30, 30), (30, 90), (90, 60)), r=12):
    image = np.empty(shape + (3,), np.uint8)
    image[:] = CARD
    for cy, cx in centres:
        image[disc(shape, cy, cx, r)] = KERNEL
    return image


def test_foreground_is_the_kernels_not_the_card():
    image = tray()
    fg = foreground(image)
    kernel_px = np.all(image == KERNEL, axis=2)
    agreement = (fg == kernel_px).mean()
    assert agreement > 0.98


def test_card_enclosed_by_touching_kernels_stays_background():
    """A ring of touching kernels encloses card; only a speck inside a kernel is filled."""
    shape = (140, 200)
    ring = ((40, 40), (40, 66), (66, 40), (66, 66))  # overlap pairwise, enclose a gap
    loose = ((30, 120), (30, 165), (90, 120), (90, 165), (120, 30))
    image = tray(shape, ring + loose, r=14)
    image[89:92, 164:167] = CARD  # dark speck inside an isolated kernel
    fg = foreground(image)
    assert not fg[53, 53]  # centre of the ring is card
    assert fg[90, 165]  # the speck is filled


def test_each_seeded_interior_becomes_one_kernel():
    image = tray()
    fg = foreground(image)
    interior = np.zeros(fg.shape, np.float32)
    for cy, cx in ((30, 30), (30, 90), (90, 60)):
        interior[disc(fg.shape, cy, cx, 7)] = 0.9
    instances = interiors_to_instances(interior, fg)
    assert len(np.unique(instances[instances > 0])) == 3


def test_a_kernel_the_network_missed_is_kept_not_lost():
    """An isolated kernel with no predicted interior must still be labelled."""
    image = tray()
    fg = foreground(image)
    interior = np.zeros(fg.shape, np.float32)
    interior[disc(fg.shape, 30, 30, 7)] = 0.9  # only one of three seeded
    instances = interiors_to_instances(interior, fg)
    assert len(np.unique(instances[instances > 0])) == 3


def test_invented_kernels_on_the_card_are_removed():
    """Interiors predicted on empty card fall outside the foreground and vanish."""
    image = tray()
    fg = foreground(image)
    interior = np.zeros(fg.shape, np.float32)
    for cy, cx in ((30, 30), (30, 90), (90, 60), (100, 140)):  # last one is empty card
        interior[disc(fg.shape, cy, cx, 7)] = 0.9
    instances = interiors_to_instances(interior, fg)
    assert len(np.unique(instances[instances > 0])) == 3
    assert instances[100, 140] == 0


def test_two_touching_kernels_labelled_as_one_are_split():
    shape = (120, 200)
    blob = disc(shape, 60, 80, 15) | disc(shape, 60, 106, 15)
    singles = [disc(shape, 20, 20 + 40 * k, 15) for k in range(4)]
    instances = np.zeros(shape, np.int32)
    for k, s in enumerate(singles, start=1):
        instances[s] = k
    instances[blob] = 9
    out = split_merged(instances)
    assert len(np.unique(out[blob])) == 2


def test_debris_is_dropped_and_ids_are_dense():
    shape = (80, 80)
    instances = np.zeros(shape, np.int32)
    instances[disc(shape, 20, 20, 10)] = 5
    instances[disc(shape, 20, 60, 10)] = 9
    instances[disc(shape, 60, 40, 10)] = 12
    instances[60:62, 70:72] = 30  # four pixels of dust
    out = drop_debris(instances)
    assert sorted(np.unique(out[out > 0])) == [1, 2, 3]


def test_instance_maps_round_trip_through_16_bit_png(tmp_path):
    instances = np.zeros((50, 70), np.int32)
    instances[5:15, 5:15] = 1
    instances[20:40, 30:60] = 700
    image_path = tmp_path / "photo.png"
    save_instances(tmp_path / "photo_inst.png", instances)
    assert np.array_equal(load_instances(image_path), instances)


def test_real_photos_are_split_by_tray_never_by_photo(tmp_path):
    for tray_name in ("tray01", "tray02", "tray03"):
        for shot in range(2):
            d = tmp_path / tray_name
            d.mkdir(exist_ok=True)
            save_instances(d / f"shot{shot}_inst.png", np.ones((4, 4), np.int32))
    train = real_scene_paths(tmp_path, exclude={"tray02", "tray03"})
    val = real_scene_paths(tmp_path, groups={"tray02"})
    assert {p.parent.name for p in train} == {"tray01"}
    assert {p.parent.name for p in val} == {"tray02"}
    assert len(train) == len(val) == 2


def test_scene_dataset_rescales_and_repeats(tmp_path):
    from PIL import Image

    Image.fromarray(tray((200, 200), ((50, 50), (150, 150)), 20)).save(tmp_path / "s.png")
    instances = np.zeros((200, 200), np.int32)
    instances[disc((200, 200), 50, 50, 20)] = 1
    instances[disc((200, 200), 150, 150, 20)] = 2
    save_instances(tmp_path / "s_inst.png", instances)
    ds = SceneDataset([tmp_path / "s.png"], crop=96, augment=False, scale=0.5, repeats=3)
    assert len(ds) == 3
    x, target, inst = ds[2]
    assert x.shape == (3, 96, 96)
    assert target.shape == inst.shape == (96, 96)
    assert set(np.unique(inst.numpy())) <= {0, 1, 2}


def test_training_uses_an_nvidia_gpu_when_one_is_present(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert train_localiser.device().type == "cuda"
