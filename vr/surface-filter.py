"""Tile-local mip pyramid for opaque city ground surfaces (colour + normal data)."""
import numpy as np
from PIL import Image

BASE = 368
GUTTER = 4
# Base at left, successively smaller levels down the right-hand strip.
LEVELS = [(0, 0, BASE)]
y = 0
size = BASE // 2
while size:
    LEVELS.append((BASE + 2 * GUTTER, y, size))
    y += size + 2 * GUTTER
    size //= 2


def encode_srgb(value):
    return np.where(value <= .0031308, value * 12.92,
                    1.055 * np.maximum(value, 0) ** (1 / 2.4) - .055)


def pyramid(image, normal=False):
    pixels = np.asarray(image.convert('RGBA'), dtype=np.float32) / 255
    rgb = pixels[:, :, :3]
    if normal:
        rgb = rgb * 2 - 1
    else:
        rgb = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    values = np.concatenate((rgb, pixels[:, :, 3:4]), axis=2)
    tile = np.zeros((744, 744, 4), dtype=np.uint8)
    for x, y, size in LEVELS:
        # Area reduction, always from the original: no gamma-space averaging or
        # cumulative quantisation. Average normal vectors without renormalising.
        filtered = values
        for axis in (0, 1):
            moved=np.moveaxis(filtered,axis,0)
            edges=np.linspace(0,len(moved),size+1)
            integral=np.concatenate((np.zeros_like(moved[:1]),np.cumsum(moved,axis=0,dtype=np.float64)),axis=0)
            indices=np.minimum(edges.astype(int),len(moved)-1)
            samples=integral[indices]+(edges-indices).reshape((-1,)+(1,)*(moved.ndim-1))*moved[indices]
            reduced=np.diff(samples,axis=0)/(len(moved)/size)
            filtered=np.moveaxis(reduced,0,axis)
        filtered[:, :, :3] = ((filtered[:, :, :3] + 1) / 2 if normal
                              else encode_srgb(filtered[:, :, :3]))
        encoded = np.clip(np.rint(filtered * 255), 0, 255).astype(np.uint8)
        padded = np.pad(encoded, ((GUTTER, GUTTER), (GUTTER, GUTTER), (0, 0)), mode='wrap')
        tile[y:y + size + 2 * GUTTER, x:x + size + 2 * GUTTER] = padded
    return Image.fromarray(tile)
