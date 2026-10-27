"""Turning what a phone sent into what the models expect.

The localiser is fully convolutional and has no scale invariance: it learned
kernels at one size in pixels and a photograph presenting them at another is a
different problem. Every upload is therefore brought to one working scale
before it reaches the pipeline. The size, and the drift it tolerates, were
measured -- see docs/scale-experiment-result.md.
"""

from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from src.app.limits import MAX_IMAGE_PIXELS

#: Long edge every upload is resized to, matching the scale the localiser was
#: trained at. Measured, not guessed.
WORKING_LONG_EDGE_PX = 1391

# Pillow's own ceiling is a warning by default, and it is the only thing
# standing between this process and a file that decodes to more pixels than
# there is memory. Raised deliberately rather than left at the default, which
# is low enough to reject a legitimate 200 MP photograph.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class NotAnImage(Exception):
    """The upload could not be decoded as an image."""


class TooManyPixels(Exception):
    """The upload decodes to more pixels than the endpoint will accept."""


def decode(data: bytes) -> Image.Image:
    """Decode an upload, honouring EXIF rotation and forcing RGB.

    Phones record orientation in EXIF rather than in the pixels, so a tray
    photographed in portrait arrives sideways unless it is transposed. The
    pipeline indexes three channels, so a greyscale or RGBA upload has to be
    converted here rather than failing deep inside inference.
    """
    if not data:
        raise NotAnImage("the upload was empty")
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise TooManyPixels(
                f"the photograph decodes to {image.width}x{image.height} pixels, "
                f"beyond the {MAX_IMAGE_PIXELS // 1_000_000} megapixel limit"
            )
        image.load()
    except Image.DecompressionBombError as exc:
        raise TooManyPixels(f"the photograph decodes too large: {exc}") from exc
    except (UnidentifiedImageError, OSError) as exc:
        # The decoder's own message names internal objects; it stays in the
        # chained exception for the log and out of the reply.
        raise NotAnImage("could not decode the upload as an image") from exc
    return ImageOps.exif_transpose(image).convert("RGB")


def to_working_scale(image: Image.Image) -> Image.Image:
    """Resize so the long edge matches the trained scale, never upscaling.

    Enlarging a small photograph would invent detail that reads as kernel
    texture, so an image already below the working size is left as it is and
    allowed to fail honestly if its kernels are too small.
    """
    long_edge = max(image.size)
    if long_edge <= WORKING_LONG_EDGE_PX:
        return image
    factor = WORKING_LONG_EDGE_PX / long_edge
    size = (round(image.width * factor), round(image.height * factor))
    return image.resize(size, Image.LANCZOS)
