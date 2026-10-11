"""Package assembled city geometry into the bounded DLMESH01 renderer format."""
import importlib.util
import json
import struct
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parent
if '--automotive' in sys.argv and '--supplied' in sys.argv: raise ValueError('Select one imported scene')
name = 'automotive' if '--automotive' in sys.argv else 'supplied-city' if '--supplied' in sys.argv else 'city'
assets = ROOT / 'assets' / name
spec=importlib.util.spec_from_file_location('surface_filter', ROOT/'surface-filter.py')
filtering=importlib.util.module_from_spec(spec)
spec.loader.exec_module(filtering)
source = np.load(assets / 'geometry.npy', allow_pickle=False)
materials = json.loads((assets / 'materials.json').read_text())
if source.ndim != 2 or source.shape[1] != 9 or len(source) % 3:
    raise ValueError('Invalid geometry array')
size, grid, gutter = 8192, 11, 4
tile = size // grid
atlas = Image.new('RGBA', (size, size), (255, 255, 255, 255))
vertices = np.zeros((len(source), 16), dtype='<f4')
# Blender Z-up to renderer Y-up. Preserve assembled metre-scale positions.
vertices[:, :3] = source[:, [0, 2, 1]]
vertices[:, 0] *= -1  # Proper rotation, not an XZY reflection.
vertices[:, 3:6] = source[:, [3, 5, 4]]
vertices[:, 3] *= -1
lengths = np.linalg.norm(vertices[:, 3:6], axis=1)
if np.any(lengths < 1e-8):
    raise ValueError('Zero normal')
vertices[:, 3:6] /= lengths[:, None]
tile_index = 0
for index, material in enumerate(materials):
    mask = source[:, 8] == index
    if not np.any(mask):
        continue
    if tile_index + len(materials) >= grid * grid:
        raise ValueError('Atlas material capacity exceeded')
    x, y = (tile_index % grid) * tile, (tile_index // grid) * tile
    tile_index += 1
    width = tile - 2 * gutter
    image = Image.open(assets / material['texture']).convert('RGBA') if material['texture'] else Image.new('RGBA', (1, 1), 'white')
    original_image=image.copy()
    mask = source[:, 8] == index
    uv = source[mask, 6:8].copy()
    uv=np.where(np.abs(uv-np.round(uv))<1e-5,np.round(uv),uv)
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
    surface=material['name'] in {'city_asphalt','city_pavement'}
    surface_colour=original_image if surface else None
    if surface and (np.any(span!=1) or material['wrap_s']!=10497 or material['wrap_t']!=10497):
        raise ValueError('Filtered ground requires one repeat in each UV direction')
    atlas.paste(image, (x + gutter, y + gutter))
    atlas.paste(image.crop((0, 0, 1, width)).resize((gutter, width)), (x, y + gutter))
    atlas.paste(image.crop((width - 1, 0, width, width)).resize((gutter, width)), (x + gutter + width, y + gutter))
    atlas.paste(atlas.crop((x, y + gutter, x + tile, y + gutter + 1)).resize((tile, gutter)), (x, y))
    atlas.paste(atlas.crop((x, y + gutter + width - 1, x + tile, y + gutter + width)).resize((tile, gutter)), (x, y + gutter + width))
    vertices[mask, 6:8] = ((uv - lower) / span * (width - 1) + [x + gutter + 0.5, y + gutter + 0.5]) / size
    if material.get('normal_texture'):
        normal_index=tile_index+len(materials)
        nx,ny=(normal_index%grid)*tile,(normal_index//grid)*tile
        normal=Image.open(assets/material['normal_texture']).convert('RGBA')
        original_normal=normal.copy()
        pixels=np.asarray(normal)
        normal=Image.fromarray(pixels[np.minimum((v*pixels.shape[0]).astype(int),pixels.shape[0]-1)[:,None],np.minimum((u*pixels.shape[1]).astype(int),pixels.shape[1]-1)[None,:]])
        padded=np.pad(np.asarray(normal),((gutter,gutter),(gutter,gutter),(0,0)),mode='edge')
        normal=Image.fromarray(padded)
        atlas.paste(filtering.pyramid(original_normal,normal=True) if surface else normal,(nx,ny))
        vertices[mask,11]=normal_index+1+(128 if surface else 0)
    if surface:
        if not material.get('normal_texture'):
            raise ValueError('Filtered ground requires a normal tile')
        atlas.paste(filtering.pyramid(surface_colour),(x,y))
    vertices[mask, 8:11] = [material['roughness'], material['metallic'], material['cutoff']]
    vertices[mask, 12:16] = material['factor']
if len(vertices) > 15500000:
    raise ValueError('City exceeds 15.5-million-vertex build budget (runtime cap is 16 million)')
if not np.isfinite(vertices).all():
    raise ValueError('Non-finite vertex data')
output = ROOT / 'probe' / (name+'.bin')
with output.open('wb') as stream:
    stream.write(struct.pack('<8sIIII', b'DLMESH01', len(vertices), size, size, 0))
    stream.write(vertices.tobytes())
    stream.write(atlas.tobytes())
atlas.save(assets / 'atlas.png')
print(f'PACKED {output}: {len(vertices) // 3} triangles; bounds {vertices[:, :3].min(axis=0)}..{vertices[:, :3].max(axis=0)}')
