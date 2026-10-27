"""Photograph in, ranked lots out.

The whole system in one place: localise every kernel, classify each one, map
the classes onto the mechanical-damage term, run the published deterioration
model, and order the lots. Each stage already exists and is tested on its own;
this is where the seams are, and the seams are where a plausible-looking wrong
number comes from.

Three properties this module is responsible for:

**The damage percentage reaching the model is mass basis.** The classifier
counts kernels. Steele's damage term is percent by weight. The mapping layer
converts, and the pipeline must pass the converted figure -- passing the count
would overstate damage wherever fragments are present.

**Biological deterioration never enters the equation.** Mould, insect damage
and sprouting are measured, reported to the user, and withheld from the model.
The multiplier was fitted on mechanical damage alone.

**The ranking declines to order what it cannot separate.** Counting is a finite
sample, so two lots whose damage differs by less than the sampling noise cannot
honestly be ordered. Ties are reported as ties rather than invented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.mapping.classes import DamageMeasurement, measure
from src.measurement.sampling import DamageCalibration, resolution
from src.physics.deterioration import Assessment, assess
from src.vision.dataset import CLASS_NAMES, eval_transform
from src.vision.evaluate_localiser import predict_scene
from src.vision.localise import UNet, instances_from_logits

#: A predicted kernel smaller than this fraction of the median is debris.
MIN_RELATIVE_AREA = 0.15

# --- Framing limits --------------------------------------------------------
# Measured across the six validation trays, where a kernel occupies 0.277% to
# 0.331% of the frame. The tolerance either side is the scale band from
# docs/scale-experiment-result.md: counting holds between
# 0.6x and 1.5x the trained size, which is 0.36x to 2.25x in area.
#
# This matters because the localiser is fully convolutional and has no scale
# invariance. Handed a close-up it does not degrade, it merges the frame into a
# single instance and reports a confident percentage of one thing -- which is
# indistinguishable, on screen, from a real reading of clean grain.

#: Median kernel area as a fraction of the frame, at the near limit.
MAX_KERNEL_FRAME_FRACTION = 0.00745

#: The same at the far limit.
MIN_KERNEL_FRAME_FRACTION = 0.00100

#: Below this a damage percentage is arithmetic on too small a sample to mean
#: anything, whatever the framing looks like.
MIN_KERNELS_FOR_A_PERCENTAGE = 10


class FramingError(ValueError):
    """The photograph is not framed like the trays the models were trained on.

    A ValueError so that callers already treating an unreadable photograph as a
    422 keep doing so, and get a message that says which way to move the phone.
    """


def check_framing(instances: np.ndarray, labels: list[int]) -> None:
    """Refuse a photograph whose kernels are the wrong size on the sensor.

    Raised before classification rather than after, so a misframed shot costs
    one segmentation instead of a full pass, and so no number is ever computed
    that could be shown.
    """
    if len(labels) < MIN_KERNELS_FOR_A_PERCENTAGE:
        raise FramingError(
            f"only {len(labels)} kernels found. A damage percentage needs at "
            f"least {MIN_KERNELS_FOR_A_PERCENTAGE}; photograph a tray of grain "
            f"rather than a few kernels"
        )

    frame = int(instances.shape[0]) * int(instances.shape[1])
    median_area = float(np.median([int((instances == l).sum()) for l in labels]))
    fraction = median_area / frame

    if fraction > MAX_KERNEL_FRAME_FRACTION:
        raise FramingError(
            f"the kernels fill too much of the frame ({100 * fraction:.2f}% "
            f"each, against {100 * MAX_KERNEL_FRAME_FRACTION:.2f}% allowed). "
            f"Move the camera further away so the whole tray is in view"
        )

    if fraction < MIN_KERNEL_FRAME_FRACTION:
        raise FramingError(
            f"the kernels are too small in the frame ({100 * fraction:.3f}% "
            f"each, against {100 * MIN_KERNEL_FRAME_FRACTION:.3f}% needed). "
            f"Move the camera closer so the tray fills the frame"
        )


def assert_crop_framing(state: dict) -> None:
    """Refuse a classifier trained on framing the pipeline does not produce.

    A GrainSet file holds two views of one kernel, and splitting them into
    single views is the right call for training -- each view fills the input
    instead of sharing it. But it changes what a kernel looks like to the
    model, and the pipeline feeds tight bounding-box crops cut out of a tray,
    which is the other framing.

    The cost is not subtle and does not show up in the metric that selected the
    checkpoint. The split-view model scores 0.8893 macro-F1 on its own
    validation split, then reads 16.91% damage off a tray with none, calling 56
    of 161 sound kernels fragments -- 22.6 points of mean error end to end
    against the paired model's 1.11, with the lot ranking broken.
    """
    if state.get("single_view"):
        raise ValueError(
            "this checkpoint was trained on split views; the pipeline crops "
            "single kernels from a tray, and the framing mismatch reads "
            "undamaged kernels as fragments"
        )


@dataclass(frozen=True)
class CorrectedDamage:
    """Damage after the classifier's own error rates are divided out."""

    mass_pct: float
    count_pct: float


@dataclass(frozen=True)
class Detections:
    """Where the kernels were and what each one was called.

    The aggregate reading cannot be drawn on a photograph, and a second pass to
    recover the instances would be a second measurement that could disagree
    with the first. So the pipeline hands both out of one pass.
    """

    #: Instance labels per pixel; 0 is background.
    instances: np.ndarray

    #: Instance labels that survived the debris filter, in prediction order.
    labels: list[int]

    #: Project class name for each kept instance.
    predictions: list[str]

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.predictions):
            raise ValueError(
                "one prediction per instance: "
                f"{len(self.labels)} labels, {len(self.predictions)} predictions"
            )


def correct_measurement(
    measurement: DamageMeasurement, calibration: DamageCalibration | None
) -> CorrectedDamage:
    """Remove the classifier's systematic bias from a damage measurement.

    The error rates are counted per kernel, so the correction belongs on the
    count basis; the mass figure follows by the same count-to-mass ratio the
    measurement already carries. Correcting the mass percentage directly would
    subtract a count-basis false-positive floor from a mass-basis number, and
    where the damaged population is mostly fragments those two live on
    different scales.

    Without a calibration the measurement passes through untouched. An
    uncorrected reading is honest -- it is simply the raw one -- and a wrong
    correction is not, so the pipeline runs without one rather than assuming.
    """
    if calibration is None:
        return CorrectedDamage(
            mass_pct=measurement.mechanical_damage_mass_pct,
            count_pct=measurement.mechanical_damage_count_pct,
        )
    if calibration.basis == "mass":
        mass_pct = calibration.correct(measurement.mechanical_damage_mass_pct)
        return CorrectedDamage(
            mass_pct=mass_pct,
            count_pct=mass_pct / measurement.count_to_mass_ratio,
        )
    count_pct = calibration.correct(measurement.mechanical_damage_count_pct)
    return CorrectedDamage(
        mass_pct=count_pct * measurement.count_to_mass_ratio,
        count_pct=count_pct,
    )


@dataclass
class LotReading:
    """Everything one photograph says about one lot."""

    lot_id: str
    kernels_counted: int
    damage_mass_pct: float
    damage_count_pct: float
    biological_pct: dict[str, float]
    class_counts: dict[str, int]

    temperature_c: float
    moisture_pct_wb: float

    assessment: Assessment = field(repr=False)

    #: What the classifier reported before its bias was divided out, and the
    #: rates that were divided out. Left unset they fall back to the reading
    #: itself, which is what an uncorrected reading's raw figure is -- so a
    #: consumer never has to test for absence.
    raw_damage_mass_pct: float | None = None
    raw_damage_count_pct: float | None = None
    calibration: DamageCalibration | None = None

    def __post_init__(self) -> None:
        if self.raw_damage_mass_pct is None:
            self.raw_damage_mass_pct = self.damage_mass_pct
        if self.raw_damage_count_pct is None:
            self.raw_damage_count_pct = self.damage_count_pct

    @property
    def resolvable_gap_pct(self) -> float:
        """Smallest damage difference this reading can honestly distinguish."""
        return resolution(
            self.damage_mass_pct, max(self.kernels_counted, 1)
        ).minimum_resolvable_gap_pct


def _crop(image: np.ndarray, instances: np.ndarray, label: int, pad: int = 4):
    ys, xs = np.where(instances == label)
    if len(ys) == 0:
        return None
    y0 = max(0, ys.min() - pad)
    y1 = min(image.shape[0], ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(image.shape[1], xs.max() + 1 + pad)
    return Image.fromarray(image[y0:y1, x0:x1])


class Pipeline:
    def __init__(
        self,
        localiser_checkpoint: str | Path = "models/localiser-real.pt",
        classifier_checkpoint: str | Path = "models/classifier.session.pt",
        device: torch.device | None = None,
        batch_size: int = 128,
        calibration: DamageCalibration | str | Path | None = None,
    ):
        from torchvision import models as tv

        if isinstance(calibration, (str, Path)):
            calibration = DamageCalibration.from_dict(
                json.loads(Path(calibration).read_text())
            )
        self.calibration = calibration

        self.device = device or (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        )
        self.batch_size = batch_size

        self.localiser = UNet().to(self.device)
        self.localiser.load_state_dict(
            torch.load(localiser_checkpoint, map_location=self.device)["model"]
        )
        self.localiser.eval()

        state = torch.load(classifier_checkpoint, map_location=self.device)
        assert_crop_framing(state)
        classifier = tv.resnet50()
        classifier.fc = torch.nn.Linear(classifier.fc.in_features, len(CLASS_NAMES))
        classifier.load_state_dict(state["model"])
        self.classifier = classifier.to(self.device).eval()
        self.classes = state.get("classes", CLASS_NAMES)
        self.transform = eval_transform()

    @torch.no_grad()
    def analyse(
        self,
        image_path: str | Path,
        *,
        lot_id: str,
        temperature_c: float,
        moisture_pct_wb: float,
    ) -> tuple[LotReading, Detections]:
        """The reading and the instances it was computed from, in one pass."""
        image = np.array(Image.open(image_path).convert("RGB"))

        instances = instances_from_logits(
            predict_scene(self.localiser, image, self.device)
        )
        labels = [int(l) for l in np.unique(instances) if l != 0]
        if not labels:
            raise ValueError(f"no kernels found in {image_path}")

        # Drop debris relative to the kernels actually present rather than to a
        # fixed pixel count, so the same threshold holds at any magnification.
        areas = {l: int((instances == l).sum()) for l in labels}
        floor = np.median(list(areas.values())) * MIN_RELATIVE_AREA
        labels = [l for l in labels if areas[l] >= floor]

        check_framing(instances, labels)

        crops, kept = [], []
        for label in labels:
            crop = _crop(image, instances, label)
            if crop is not None:
                crops.append(self.transform(crop))
                kept.append(label)

        predictions: list[int] = []
        for start in range(0, len(crops), self.batch_size):
            batch = torch.stack(crops[start : start + self.batch_size]).to(self.device)
            predictions.extend(self.classifier(batch).argmax(1).cpu().tolist())

        counts: dict[str, int] = {}
        for index in predictions:
            name = self.classes[index]
            counts[name] = counts.get(name, 0) + 1

        measurement = measure(counts)
        corrected = correct_measurement(measurement, self.calibration)
        assessment = assess(corrected.mass_pct, temperature_c, moisture_pct_wb)

        reading = LotReading(
            lot_id=lot_id,
            kernels_counted=measurement.kernels_counted,
            damage_mass_pct=corrected.mass_pct,
            damage_count_pct=corrected.count_pct,
            biological_pct=measurement.biological_count_pct,
            class_counts=counts,
            temperature_c=temperature_c,
            moisture_pct_wb=moisture_pct_wb,
            assessment=assessment,
            raw_damage_mass_pct=measurement.mechanical_damage_mass_pct,
            raw_damage_count_pct=measurement.mechanical_damage_count_pct,
            calibration=self.calibration,
        )

        names = [self.classes[index] for index in predictions]
        return reading, Detections(
            instances=instances, labels=kept, predictions=names
        )

    def read(
        self,
        image_path: str | Path,
        *,
        lot_id: str,
        temperature_c: float,
        moisture_pct_wb: float,
    ) -> LotReading:
        """The reading alone, for callers that do not draw the photograph."""
        reading, _ = self.analyse(
            image_path,
            lot_id=lot_id,
            temperature_c=temperature_c,
            moisture_pct_wb=moisture_pct_wb,
        )
        return reading


@dataclass
class RankedLot:
    reading: LotReading
    rank: int

    #: Lots this one could not be separated from. A tie is a result: it says
    #: the damage difference is inside the sampling noise of the measurement.
    tied_with: tuple[str, ...] = ()


def rank_lots(readings: list[LotReading]) -> list[RankedLot]:
    """Order lots fastest-degrading first, refusing to split unresolvable pairs.

    Ordering is by degradation rate from the deterioration model, not by damage
    directly, because temperature and moisture belong in the comparison too.
    Ties are decided on the damage measurement, since that is the quantity
    carrying sampling error -- the environmental inputs are typed in by the user
    and carry no counting noise.
    """
    ordered = sorted(
        readings, key=lambda r: r.assessment.degradation_rate, reverse=True
    )
    ranked: list[RankedLot] = []
    for i, reading in enumerate(ordered):
        tied = []
        for other in ordered:
            if other is reading:
                continue
            gap = abs(other.damage_mass_pct - reading.damage_mass_pct)
            if gap < max(reading.resolvable_gap_pct, other.resolvable_gap_pct):
                tied.append(other.lot_id)
        ranked.append(RankedLot(reading=reading, rank=i + 1, tied_with=tuple(tied)))
    return ranked
