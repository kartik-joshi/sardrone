"""Make clean simulated pictures look more like a real camera took them.

A rendered image is too perfect: no sensor noise, no blur, no compression
blocks, always the same exposure. A detector trained on such images can
come to rely on that cleanliness and then stumble on real footage. Each
training image is therefore degraded a little, in four ways a real drone
camera does it:

    exposure    brighter or darker overall, and a gamma change
    blur        slight defocus or motion softness (half of the images)
    noise       random speckle on every pixel, as from a small sensor
    JPEG        strong compression, which smears fine detail into blocks

Only the pixels change; the boxes stay where they are. The amount of each
effect is drawn from a range, and the draw is seeded by the image's NAME,
so the same image always gets the same treatment on every machine.
"""

import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class RealismConfig:
    gain: tuple[float, float] = (0.8, 1.2)  # exposure multiplier
    gamma: tuple[float, float] = (0.85, 1.15)
    blur_probability: float = 0.5
    blur_radius_px: tuple[float, float] = (0.3, 1.2)
    noise_sigma: tuple[float, float] = (1.0, 6.0)  # in 8-bit grey levels
    jpeg_quality: tuple[int, int] = (35, 85)


@dataclass(frozen=True)
class RealismDraw:
    """What was actually applied to one image."""

    gain: float
    gamma: float
    blur_radius_px: float  # 0 = not blurred
    noise_sigma: float
    jpeg_quality: int


def draw_for(name: str, seed: int, config: RealismConfig = RealismConfig()) -> tuple[RealismDraw, np.random.Generator]:
    """The seeded random choices for the image called `name`."""
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    blurred = rng.random() < config.blur_probability
    draw = RealismDraw(
        gain=float(rng.uniform(*config.gain)),
        gamma=float(rng.uniform(*config.gamma)),
        blur_radius_px=float(rng.uniform(*config.blur_radius_px)) if blurred else 0.0,
        noise_sigma=float(rng.uniform(*config.noise_sigma)),
        jpeg_quality=int(rng.integers(config.jpeg_quality[0], config.jpeg_quality[1] + 1)),
    )
    return draw, rng


def apply(image: Image.Image, draw: RealismDraw, rng: np.random.Generator) -> bytes:
    """Degrade `image` as `draw` says; returns JPEG bytes."""
    if draw.blur_radius_px > 0:
        image = image.filter(ImageFilter.GaussianBlur(draw.blur_radius_px))
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    # Gamma first (a curve), then gain (a multiplier), as a camera's
    # processing would: output = gain * input ** gamma.
    pixels = draw.gain * np.power(pixels, draw.gamma)
    pixels = pixels * 255.0 + rng.normal(0.0, draw.noise_sigma, pixels.shape).astype(np.float32)
    out = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    out.save(buffer, format="JPEG", quality=draw.jpeg_quality)
    return buffer.getvalue()


def degrade_file(source, name: str, seed: int = 51, config: RealismConfig = RealismConfig()) -> tuple[bytes, RealismDraw]:
    draw, rng = draw_for(name, seed, config)
    with Image.open(source) as image:
        return apply(image, draw, rng), draw
