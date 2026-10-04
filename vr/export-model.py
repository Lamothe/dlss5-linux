"""Import a GLB in a fresh Blender process; export evaluated static geometry."""
import bpy
import json
import numpy as np
import sys
import struct
from PIL import Image
import io
from pathlib import Path
args = sys.argv[sys.argv.index('--') + 1:]
source, destination = Path(args[0]), Path(args[1])
destination.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(source))
deps = bpy.context.evaluated_depsgraph_get()
materials = list(bpy.data.materials)
with source.open('rb') as stream:
    magic, version, length = struct.unpack('<III', stream.read(12))
    if magic != 0x46546c67 or version != 2:
        raise ValueError('Expected glTF 2 GLB')
    chunks = {}
    while stream.tell() < length:
        chunk_length, chunk_type = struct.unpack('<II', stream.read(8))
        chunks[chunk_type] = stream.read(chunk_length)
    gltf = json.loads(chunks[0x4e4f534a])
    binary = chunks[0x004e4942]
source_names = [item.get('name') for item in gltf['materials']]
# Blender may rename unnamed/duplicate materials. Reject instead of guessing.
if any(not name for name in source_names) or len(set(source_names)) != len(source_names):
    raise ValueError('This static exporter requires unique, non-empty glTF material names')
metadata = []
for index, material in enumerate(materials):
    original = next(item for item in gltf['materials'] if item.get('name') == material.name)
    pbr = original.get('pbrMetallicRoughness', {})
    base = pbr.get('baseColorTexture')
    texture = None
    wrap_s = wrap_t = 10497
    if base is not None:
        if base.get('texCoord', 0) != 0 or base.get('extensions'):
            raise ValueError('Unsupported texture UV mapping: ' + material.name)
        texture_info = gltf['textures'][base['index']]
        sampler = gltf.get('samplers', [])[texture_info['sampler']] if 'sampler' in texture_info else {}
        wrap_s, wrap_t = sampler.get('wrapS', 10497), sampler.get('wrapT', 10497)
        if wrap_s not in (10497, 33071, 33648) or wrap_t not in (10497, 33071, 33648):
            raise ValueError('Unsupported texture address mode')
        image = gltf['images'][texture_info['source']]
        if 'bufferView' not in image:
            raise ValueError('Expected embedded image')
        view = gltf['bufferViews'][image['bufferView']]
        offset = view.get('byteOffset', 0)
        texture = f'material-{index}.png'
        Image.open(io.BytesIO(binary[offset:offset + view['byteLength']])).convert('RGBA').save(destination / texture)
    mode = original.get('alphaMode', 'OPAQUE')
    metadata.append({'name': material.name, 'texture': texture,
                     'factor': pbr.get('baseColorFactor', [1, 1, 1, 1]),
                     'roughness': pbr.get('roughnessFactor', 1),
                     'metallic': pbr.get('metallicFactor', 1),
                     'alpha_mode': mode, 'wrap_s': wrap_s, 'wrap_t': wrap_t,
                     'cutoff': original.get('alphaCutoff', 0.5) if mode == 'MASK' else (0.5 if mode == 'BLEND' else 0)})
vertices = []
for obj in bpy.context.scene.objects:
    if obj.type != 'MESH':
        continue
    if not obj.material_slots:
        print('SKIP unmaterialled helper', obj.name)
        continue
    evaluated = obj.evaluated_get(deps)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        uv = mesh.uv_layers.active
        if uv is None:
            raise ValueError('Missing UVs: ' + obj.name)
        normal_matrix = evaluated.matrix_world.to_3x3().inverted().transposed()
        for triangle in mesh.loop_triangles:
            material = obj.material_slots[triangle.material_index].material
            material_index = next(i for i, source_material in enumerate(materials) if source_material.name == material.name)
            for loop_index in triangle.loops:
                loop = mesh.loops[loop_index]
                position = evaluated.matrix_world @ mesh.vertices[loop.vertex_index].co
                normal = (normal_matrix @ mesh.corner_normals[loop_index].vector).normalized()
                texcoord = uv.data[loop_index].uv
                vertices.append([*position, *normal, texcoord.x, 1 - texcoord.y, material_index])
    finally:
        evaluated.to_mesh_clear()
array = np.asarray(vertices, dtype='<f4')
if not len(array) or not np.isfinite(array).all():
    raise ValueError('Invalid or empty geometry')
np.save(destination / 'geometry.npy', array)
(destination / 'materials.json').write_text(json.dumps(metadata, indent=2))
print(f'EXPORTED vertices={len(array)} materials={len(metadata)} bounds={array[:, :3].min(axis=0)}..{array[:, :3].max(axis=0)}')
