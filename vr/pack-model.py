"""Package evaluated static geometry into the bounded DLMESH01 renderer format."""
import json
import struct
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parent
assets = ROOT / 'assets/test-model'
source = np.load(assets / 'geometry.npy', allow_pickle=False)
materials = json.loads((assets / 'materials.json').read_text())
if source.ndim != 2 or source.shape[1] != 9 or len(source) % 3:
    raise ValueError('Invalid geometry array')
size, grid, gutter = 4096, 5, 4
tile = size // grid
atlas = Image.new('RGBA', (size, size), (255, 255, 255, 255))
vertices = np.zeros((len(source), 16), dtype='<f4')
# Blender Z-up to renderer Y-up. Character faces towards the starting viewer.
vertices[:, :3] = source[:, [0, 2, 1]]
vertices[:, 2] += 2.35 # Clear floor in front of table (starts at Z=3.0).
vertices[:, 1] -= vertices[:, 1].min()
vertices[:, 3:6] = source[:, [3, 5, 4]]
lengths = np.linalg.norm(vertices[:, 3:6], axis=1)
if np.any(lengths < 1e-8):
    raise ValueError('Zero normal')
vertices[:, 3:6] /= lengths[:, None]
tile_index = 0
for index, material in enumerate(materials):
    mask = source[:, 8] == index
    if not np.any(mask):
        continue
    if tile_index >= grid * grid:
        raise ValueError('Atlas material capacity exceeded')
    x, y = (tile_index % grid) * tile, (tile_index // grid) * tile
    tile_index += 1
    width = tile - 2 * gutter
    image = Image.open(assets / material['texture']).convert('RGBA') if material['texture'] else Image.new('RGBA', (1, 1), 'white')
    mask = source[:, 8] == index
    uv = source[mask, 6:8]
    lower = np.floor(uv.min(axis=0))
    upper = np.ceil(uv.max(axis=0))
    span = np.maximum(upper - lower, 1)
    # Preserve repeat addressing and UDIM-style integer UV offsets in an atlas.
    pixels = np.asarray(image)
    def address(coordinates, mode):
        if mode == 10497:
            return coordinates % 1
        if mode == 33071:
            return np.clip(coordinates, 0, 1)
        if mode == 33648:
            return 1 - np.abs(coordinates % 2 - 1)
        raise ValueError('Unsupported texture addressing')
    u = address((np.arange(width) + 0.5) / width * span[0] + lower[0], material['wrap_s'])
    v = address((np.arange(width) + 0.5) / width * span[1] + lower[1], material['wrap_t'])
    image = Image.fromarray(pixels[np.minimum((v * pixels.shape[0]).astype(int), pixels.shape[0]-1)[:, None], np.minimum((u * pixels.shape[1]).astype(int), pixels.shape[1]-1)[None, :]])
    atlas.paste(image, (x + gutter, y + gutter))
    atlas.paste(image.crop((0, 0, 1, width)).resize((gutter, width)), (x, y + gutter))
    atlas.paste(image.crop((width - 1, 0, width, width)).resize((gutter, width)), (x + gutter + width, y + gutter))
    atlas.paste(atlas.crop((x, y + gutter, x + tile, y + gutter + 1)).resize((tile, gutter)), (x, y))
    atlas.paste(atlas.crop((x, y + gutter + width - 1, x + tile, y + gutter + width)).resize((tile, gutter)), (x, y + gutter + width))
    vertices[mask, 6:8] = ((uv - lower) / span * (width - 1) + [x + gutter + 0.5, y + gutter + 0.5]) / size
    vertices[mask, 8:11] = [material['roughness'], material['metallic'], material['cutoff']]
    vertices[mask, 12:16] = material['factor']
if not np.isfinite(vertices).all():
    raise ValueError('Non-finite vertex data')
output = ROOT / 'probe/model.bin'
with output.open('wb') as stream:
    stream.write(struct.pack('<8sIIII', b'DLMESH01', len(vertices), size, size, 0))
    stream.write(vertices.tobytes())
    stream.write(atlas.tobytes())
atlas.save(assets / 'atlas.png')
print(f'PACKED {output}: {len(vertices) // 3} triangles; bounds {vertices[:, :3].min(axis=0)}..{vertices[:, :3].max(axis=0)}')
