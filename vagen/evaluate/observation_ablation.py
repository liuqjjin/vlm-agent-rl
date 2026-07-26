"""Deterministic visual ablations for anti-cheating evaluations."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
from PIL import Image


VALID_OBSERVATION_ABLATIONS = {"none", "remove", "shuffle_tiles"}


def _rng_seed(seed: int, turn: int, image_index: int) -> int:
    payload = f"{int(seed)}:{int(turn)}:{int(image_index)}".encode()
    return int.from_bytes(
        hashlib.blake2s(payload, digest_size=8).digest(),
        byteorder="little",
        signed=False,
    )


def shuffle_image_tiles(
    image: Image.Image,
    *,
    seed: int,
    turn: int,
    image_index: int,
    grid_size: int = 5,
) -> Image.Image:
    """Shuffle equal-sized spatial tiles while preserving pixels and dimensions."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    width, height = image.size
    tile_width = width // grid_size
    tile_height = height // grid_size
    if tile_width == 0 or tile_height == 0:
        raise ValueError(
            f"image {image.size} is too small for a {grid_size}x{grid_size} tile shuffle"
        )

    tiles = []
    for row in range(grid_size):
        for column in range(grid_size):
            left = column * tile_width
            top = row * tile_height
            tiles.append(
                image.crop((left, top, left + tile_width, top + tile_height))
            )

    permutation = np.random.default_rng(
        _rng_seed(seed, turn, image_index)
    ).permutation(len(tiles))
    shuffled = image.copy()
    for destination, source in enumerate(permutation.tolist()):
        row, column = divmod(destination, grid_size)
        shuffled.paste(
            tiles[source],
            (column * tile_width, row * tile_height),
        )
    return shuffled


def ablate_images(
    images: Iterable[Image.Image],
    *,
    mode: str,
    seed: int,
    turn: int,
) -> list[Image.Image]:
    """Apply one declared visual ablation without mutating environment images."""
    image_list = list(images)
    if mode not in VALID_OBSERVATION_ABLATIONS:
        raise ValueError(
            f"unknown observation ablation {mode!r}; "
            f"expected one of {sorted(VALID_OBSERVATION_ABLATIONS)}"
        )
    if mode == "none":
        return image_list
    if mode == "remove":
        return []
    return [
        shuffle_image_tiles(
            image,
            seed=seed,
            turn=turn,
            image_index=image_index,
        )
        for image_index, image in enumerate(image_list)
    ]
