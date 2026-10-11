"""Assemble licensed downloads in a fresh Blender process and export static inputs.
blender -b -t 2 --python vr/build-city.py
World: Blender Z-up; renderer (X,Z,Y). Road runs Blender Y, sidewalks X +/-5.5.
"""
import json
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix, Vector
import math
import struct
import sys

ROOT = Path(__file__).resolve().parent / 'assets/city'
AUTOMOTIVE = '--automotive' in sys.argv
SUPPLIED = '--supplied' in sys.argv or AUTOMOTIVE
if AUTOMOTIVE and '--supplied' in sys.argv: raise ValueError('Select one imported scene')
SCENE_NAME = 'automotive' if AUTOMOTIVE else 'supplied-city'
if SUPPLIED:
    ROOT = ROOT.parent / SCENE_NAME
ROOT.mkdir(parents=True, exist_ok=True)

DOWNLOADS = ROOT / 'downloads'
bpy.ops.wm.read_factory_settings(use_empty=True)
COLLIDERS=[]
def collision_bounds(objects):
    lo,hi=bounds(objects)
    COLLIDERS.append([-hi.x,lo.z,lo.y,-lo.x,hi.z,hi.y])
GLTF_MATERIALS = {}


def import_meshes(path):
    old = set(bpy.context.scene.objects)
    if path.suffix == '.fbx':
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        if path.suffix == '.glb':
            data = path.read_bytes()
            length, kind = struct.unpack_from('<II', data, 12)
            if kind != 0x4e4f534a: raise ValueError('Missing GLB JSON')
            source = json.loads(data[20:20+length])
        else: source = json.loads(path.read_text())
        for i, material in enumerate(source.get('materials', [])):
            material['name'] = (path.stem + '_' + str(i) + '_' + material.get('name', 'Material')) if path.suffix == '.gltf' else material.get('name', 'Material_' + str(i))
            if material['name'] in GLTF_MATERIALS: raise ValueError('Duplicate source material')
            material['_samplers']={}
            for key,info in [('colour',material.get('pbrMetallicRoughness',{}).get('baseColorTexture',{})),('normal',material.get('normalTexture',{}))]:
                if info:
                    texture=source.get('textures',[])[info['index']]
                    sampler=source.get('samplers',[])[texture['sampler']] if 'sampler' in texture else {}
                    material['_samplers'][key]=(sampler.get('wrapS',10497),sampler.get('wrapT',10497))
            GLTF_MATERIALS[material['name']] = material
        if path.suffix == '.gltf':
            temporary = path.with_name('city-import.gltf')
            temporary.write_text(json.dumps(source))
            try: bpy.ops.import_scene.gltf(filepath=str(temporary))
            finally: temporary.unlink()
        else: bpy.ops.import_scene.gltf(filepath=str(path))
    bpy.context.view_layer.update()
    return [o for o in bpy.context.scene.objects if o not in old and o.type == 'MESH']


def bounds(objects):
    points = [o.matrix_world @ Vector(corner) for o in objects for corner in o.bound_box]
    return Vector([min(p[i] for p in points) for i in range(3)]), Vector([max(p[i] for p in points) for i in range(3)])


def duplicate(objects, transform, tag):
    result = []
    for original in objects:
        obj = original.copy()
        obj.parent = None
        bpy.context.collection.objects.link(obj)
        obj.matrix_world = transform @ original.matrix_world
        obj.name = tag + '_' + original.name
        result.append(obj)
    return result


def delete_sources(objects):
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)


if SUPPLIED:
    source_path=Path('/home/michael/Downloads/VR/Vehicles/Modern+Minimalist+Automotive+Scene.glb' if AUTOMOTIVE else '/home/michael/Downloads/VR/Scenes/street_city_7_for_games_free.glb')
    imported=import_meshes(source_path)
    lo,hi=bounds(imported)
    # Preserve authored scale/placement. Tall, compact mesh envelopes are coarse
    # obstacles; broad ground/roof planes must not block the entire neighbourhood.
    for obj in imported:
        low,high=bounds([obj]); extent=high-low
        if extent.z>1.5 and extent.x>.15 and extent.y>.15 and extent.x<18 and extent.y<18:
            collision_bounds([obj])
    x0,x1=-hi.x,-lo.x; z0,z1=lo.y,hi.y
    COLLIDERS.extend([[x0-1,-5,z0-1,x0,30,z1+1],
                      [x1,-5,z0-1,x1+1,30,z1+1],
                      [x0,-5,z0-1,x1,30,z0],
                      [x0,-5,z1,x1,30,z1+1]])
    with (Path(__file__).resolve().parent/('probe/'+SCENE_NAME+'-collision.bin')).open('wb') as file:
        file.write(struct.pack('<8sI',b'DLCOLL01',len(COLLIDERS)))
        for box in COLLIDERS: file.write(struct.pack('<6f',*box))
    (ROOT/'source.json').write_text(json.dumps({'path':str(source_path),'asset':json.loads(source_path.read_bytes()[20:20+struct.unpack_from('<I',source_path.read_bytes(),12)[0]])['asset']},indent=2))
else:
    kit = import_meshes(DOWNLOADS / 'modular_urban_apartments_facade/modular_urban_apartments_facade_1k.gltf')
    named = {o.name: o for o in kit}
    # Kit parts use author-provided origins: Z=0 ground doors, Z=4 upper windows,
    # Z=3.4 cornices, Z=4 crowns. Replace each lateral origin when assembling.
    SIDE_BAYS = 21
    END_BAYS = 6
    for side in (-1, 1):
        rotation = Matrix.Rotation(-side * math.pi / 2, 4, 'Z')
        for bay in range(SIDE_BAYS):
            along = (-18 if side == -1 else -21) + bay * 3
            for floor in range(3):
                parts = ['wall_door_centered_large_01', 'door_centered_large_01'] if floor == 0 and bay % 4 == 0 else ['wall_window_centered_large_01', 'window_centered_large_01']
                for name in parts + ['cornice_standard_standard_01']:
                    obj = named[name]
                    offset = Vector((0, 0, floor * 3.0 + .12 + (2.8 if name == 'cornice_standard_standard_01' else 0))) - obj.location
                    duplicate([obj], Matrix.Translation((side * 8, along, 0)) @ rotation @ Matrix.Translation(offset), f'facade_{side}_{bay}_{floor}')
            obj = named['crown_standard_standard_01']
            duplicate([obj], Matrix.Translation((side * 8, along, 9.12)) @ rotation @ Matrix.Translation(-obj.location), f'crown_{side}_{bay}')
    # Transverse frontages face back into the street; contiguous 3m bays close
    # the road at -21m and +42m rather than leaving either end open to the sky.
    for end, angle in [(-21, 0), (42, math.pi)]:
        rotation = Matrix.Rotation(angle, 4, 'Z')
        for centre_x in [-7.5, -4.5, -1.5, 1.5, 4.5, 7.5]:
            for floor in range(3):
                for name in ['wall_window_centered_large_01', 'window_centered_large_01', 'cornice_standard_standard_01']:
                    obj = named[name]
                    offset = Vector((0, 0, floor * 3.0 + .12 + (2.8 if name == 'cornice_standard_standard_01' else 0))) - obj.location
                    pieces = duplicate([obj], rotation @ Matrix.Translation(offset), f'end_{end}_{centre_x}_{floor}')
                    # Window/cornice retain their relative alignment with the 3m wall.
                    translation = Vector((centre_x + (1.5 if angle == 0 else -1.5), end, 0))
                    for piece in pieces: piece.matrix_world = Matrix.Translation(translation) @ piece.matrix_world
            obj = named['crown_standard_standard_01']
            duplicate([obj], Matrix.Translation((centre_x + (1.5 if angle == 0 else -1.5), end, 9.12)) @ rotation @ Matrix.Translation(-obj.location), f'end_crown_{end}_{centre_x}')
    delete_sources(kit)

    # User-supplied Volvo exported without modifying the original.
    cars = import_meshes(DOWNLOADS / 'user-volvo.glb')
    # Keep the detailed source untouched; reduce repeated display geometry for VR.
    for obj in cars:
        if len(obj.data.polygons)>200:
            bpy.context.view_layer.objects.active=obj
            modifier=obj.modifiers.new('VR display reduction','DECIMATE');modifier.ratio=.20
            bpy.ops.object.modifier_apply(modifier=modifier.name)
    lo, hi = bounds(cars)
    scale = 4.95 / (hi.y-lo.y)
    centre = (lo+hi)*.5
    for i, (x, y, direction) in enumerate([(-2.65,-6,1),(2.65,6,-1),(-2.65,17,1),(2.65,29,-1),(2.65,-12,-1)]):
        transform = Matrix.Translation((x,y,-.02)) @ Matrix.Rotation(0 if direction>0 else math.pi,4,'Z') @ Matrix.Scale(scale,4) @ Matrix.Translation((-centre.x,-centre.y,-lo.z))
        placed=duplicate(cars,transform,'car_'+str(i))
        bottom,top=bounds(placed)
        if abs(bottom.z+.02)>1e-4: raise ValueError('Car contact does not match road height')
        print('CAR CONTACT',i,bottom.z,top.z)
        collision_bounds(placed)
    delete_sources(cars)

    for avatar, prefix, height, positions in [
        ('Male_Adult_01', 'm002', 1.82, [(-5.6, 4, 0), (5.7, 19, math.pi), (-6, 30, .4)]),
        ('Female_Adult_03', 'f003', 1.7, [(5.6, 7, math.pi), (-6, 18, .5), (6, 32, 2.8)])
    ]:
        folder = DOWNLOADS / 'rocketbox' / avatar
        people = import_meshes(folder / 'Export' / (avatar + '.fbx'))
        # Lower arms from bind pose for a natural static pedestrian pose.
        armatures = {o.parent for o in people if o.parent and o.parent.type == 'ARMATURE'}
        for armature in armatures:
            for bone in armature.pose.bones:
                if bone.name.lower().endswith('upperarm') or 'upperarm' in bone.name.lower():
                    bone.rotation_mode = 'XYZ'
                    bone.rotation_euler.y = (-1 if ' l ' in (' ' + bone.name.lower() + ' ') or bone.name.lower().startswith('l') else 1) * .65
        for obj in people:
            for slot in obj.material_slots:
                mat = slot.material
                shader = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
                suffix = 'opacity' if 'opacity' in mat.name else ('head' if 'head' in mat.name else 'body')
                image = bpy.data.images.load(str(folder / 'Textures' / f'{prefix}_{suffix}_color.tga'), check_existing=True)
                node = mat.node_tree.nodes.new('ShaderNodeTexImage'); node.image = image
                for socket in ['Base Color', 'Alpha', 'Normal', 'Roughness', 'Metallic']:
                    for link in list(shader.inputs[socket].links): mat.node_tree.links.remove(link)
                mat.node_tree.links.new(node.outputs['Color'], shader.inputs['Base Color'])
                shader.inputs['Base Color'].default_value = (1, 1, 1, 1)
                shader.inputs['Metallic'].default_value = 0
                shader.inputs['Roughness'].default_value = .7
                shader.inputs['Alpha'].default_value = 1
                if suffix == 'opacity': mat.node_tree.links.new(node.outputs['Alpha'], shader.inputs['Alpha'])
        bpy.context.view_layer.update()
        # Bake pose before copying, so copied meshes don't refer to deleted armature.
        deps = bpy.context.evaluated_depsgraph_get()
        baked = []
        for obj in people:
            mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(deps), depsgraph=deps)
            replacement = bpy.data.objects.new(avatar + '_baked', mesh)
            bpy.context.collection.objects.link(replacement)
            replacement.matrix_world = obj.matrix_world.copy()
            baked.append(replacement)
        lo, hi = bounds(baked); scale = height / (hi.z - lo.z)
        centre = Vector(((lo.x + hi.x)/2, (lo.y + hi.y)/2, lo.z))
        for i, (x, y, angle) in enumerate(positions):
            placed=duplicate(baked, Matrix.Translation((x, y, .12)) @ Matrix.Rotation(angle, 4, 'Z') @ Matrix.Scale(scale, 4) @ Matrix.Translation(-centre), f'person_{avatar}_{i}')
            collision_bounds(placed)
        delete_sources(people + baked)

    # Explicit curb geometry gives depth discontinuities and a raised walking surface.
    def material(name, colour):
        mat = bpy.data.materials.new(name); mat.use_nodes = True
        shader = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        shader.inputs['Base Color'].default_value = (*colour, 1)
        shader.inputs['Roughness'].default_value = .85
        return mat

    def surface_material(name,asset):
        mat=material(name,(1,1,1))
        shader=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        for channel,socket in [('Diffuse','Base Color'),('nor_gl','Normal')]:
            node=mat.node_tree.nodes.new('ShaderNodeTexImage')
            node.image=bpy.data.images.load(str(DOWNLOADS/'surfaces'/f'{asset}-{channel}.jpg'),check_existing=True)
            if channel=='nor_gl':
                node.image.colorspace_settings.name='Non-Color'
                normal=mat.node_tree.nodes.new('ShaderNodeNormalMap')
                mat.node_tree.links.new(node.outputs['Color'],normal.inputs['Color'])
                mat.node_tree.links.new(normal.outputs['Normal'],shader.inputs[socket])
            else: mat.node_tree.links.new(node.outputs['Color'],shader.inputs[socket])
        return mat

    # Individual 2m tiles preserve texel density instead of compressing an entire
    # 63m surface into a single atlas tile. Coplanar edges do not add visible seams.
    pavement=surface_material('city_pavement','concrete_floor_02')
    asphalt=surface_material('city_asphalt','asphalt_02')
    for row in range(32):
        y=-21+row*2
        length=min(2,42-y)
        if length<=0: continue
        for col in range(8):
            x=-8+col*2
            sidewalk=abs(x+1)>4
            z=.12 if sidewalk else -.019
            mesh=bpy.data.meshes.new('surface_tile')
            mesh.from_pydata([(x,y,z),(x+2,y,z),(x+2,y+length,z),(x,y+length,z)],[],[(0,1,2,3)])
            uv=mesh.uv_layers.new()
            for loop,coord in zip(uv.data,[(0,0),(1,0),(1,length/2),(0,length/2)]): loop.uv=coord
            obj=bpy.data.objects.new('sidewalk_tile' if sidewalk else 'road_tile',mesh)
            bpy.context.collection.objects.link(obj);mesh.materials.append(pavement if sidewalk else asphalt)
    # Narrow curb edges and simple white centre markings retain their depth.
    curb=material('city_curb',(.35,.36,.37))
    for side in (-1,1):
        bpy.ops.mesh.primitive_cube_add(size=1,location=(side*4,10.5,.04))
        obj=bpy.context.object;obj.scale=(.1,63,.16);obj.data.materials.append(curb)
    marking=material('city_marking',(.78,.77,.69))
    for y in range(-18,41,5):
        bpy.ops.mesh.primitive_cube_add(size=1,location=(0,y,-.017))
        obj=bpy.context.object;obj.scale=(.11,2.5,.002);obj.data.materials.append(marking)
    # Thick façade envelopes also prevent flying through buildings.
    COLLIDERS.extend([[-9,-.1,-21, -7.7,10,42],[7.7,-.1,-21,9,10,42],[-9,-.1,-21.4,9,10,-20.7],[-9,-.1,41.7,9,10,42.4]])
    with (Path(__file__).resolve().parent/'probe/city-collision.bin').open('wb') as file:
        file.write(struct.pack('<8sI',b'DLCOLL01',len(COLLIDERS)))
        for box in COLLIDERS: file.write(struct.pack('<6f',*box))

# glTF allows primitives without materials: use its conventional neutral default.
default_material = None
for obj in bpy.context.scene.objects:
    if obj.type != 'MESH': continue
    if not obj.material_slots or any(slot.material is None for slot in obj.material_slots):
        if default_material is None:
            default_material=bpy.data.materials.new('import_default_material')
            default_material.use_nodes=True
            shader=default_material.node_tree.nodes.get('Principled BSDF')
            shader.inputs['Base Color'].default_value=(1,1,1,1)
            shader.inputs['Metallic'].default_value=1
            shader.inputs['Roughness'].default_value=1
        if not obj.material_slots: obj.data.materials.append(default_material)
        for slot in obj.material_slots:
            if slot.material is None: slot.material=default_material
# Export base-colour material inputs directly; don't silently claim full PBR fidelity.
bpy.context.view_layer.update()
deps = bpy.context.evaluated_depsgraph_get()
objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
used = sorted({s.material for o in objects for s in o.material_slots if s.material}, key=lambda m: m.name)
metadata = []
for index, mat in enumerate(used):
    print('EXPORT MATERIAL', index, mat.name, flush=True)
    shader = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if shader is None: raise ValueError('Unsupported non-Principled material: '+mat.name)
    # Follow upstream nodes to find colour texture (including glTF colour-factor mixes).
    def find_image(socket, visited):
        for link in socket.links:
            node = link.from_node
            if node in visited: continue
            visited.add(node)
            if node.type == 'TEX_IMAGE' and node.image: return node.image
            for inp in node.inputs:
                found = find_image(inp, visited)
                if found: return found
        return None
    image = find_image(shader.inputs['Base Color'], set())
    filename = None
    if image:
        filename = f'material-{index}.png'
        # Blender image save_render can colour-transform data: save raw image instead.
        original_path, original_format = image.filepath_raw, image.file_format
        image.filepath_raw = str(ROOT / filename); image.file_format = 'PNG'; image.save()
        image.filepath_raw, image.file_format = original_path, original_format
    source = GLTF_MATERIALS.get(mat.name, {})
    pbr = source.get('pbrMetallicRoughness', {})
    factor = pbr.get('baseColorFactor', [1,1,1,1]) if source else list(shader.inputs['Base Color'].default_value)
    if mat.name == 'Glass': factor = list(shader.inputs['Base Color'].default_value)
    texture_info = pbr.get('baseColorTexture', {})
    transform = texture_info.get('extensions', {}).get('KHR_texture_transform', {})
    texcoord = transform.get('texCoord', texture_info.get('texCoord', 0))
    if not isinstance(texcoord, int) or texcoord < 0: raise ValueError('Invalid base-colour UV set')
    normal_info=source.get('normalTexture', {})
    normal_image = find_image(shader.inputs['Normal'], set())
    if normal_info:
        normal_transform=normal_info.get('extensions',{}).get('KHR_texture_transform',{})
        samplers=source.get('_samplers',{})
        if normal_info.get('scale',1)!=1 or normal_transform!=transform or normal_transform.get('texCoord',normal_info.get('texCoord',0))!=texcoord or samplers.get('normal',(10497,10497))!=samplers.get('colour',(10497,10497)):
            print('UNSUPPORTED NORMAL SETTINGS: omit map for',mat.name)
            normal_image=None
    normal_filename = None
    if normal_image:
        normal_filename = f'normal-{index}.png'
        original_path, original_format = normal_image.filepath_raw, normal_image.file_format
        normal_image.filepath_raw = str(ROOT / normal_filename); normal_image.file_format = 'PNG'; normal_image.save()
        normal_image.filepath_raw, normal_image.file_format = original_path, original_format
    glass=('glass' in mat.name.lower() and mat.name != 'Black Glass') or float(shader.inputs['Transmission Weight'].default_value)>0
    if glass: factor[3]=.22 if mat.name=='Glass' else .35
    if mat.name=='Glass': factor=[.24,.32,.38,.22]
    metadata.append({'normal_texture': normal_filename, 'name': mat.name, 'texture': filename, 'uv_transform': transform, 'texcoord': texcoord,
                     'factor': factor,
                     'metallic': pbr.get('metallicFactor', 1) if source and mat.name != 'Glass' else (0 if mat.name == 'Glass' else float(shader.inputs['Metallic'].default_value)),
                     'roughness': pbr.get('roughnessFactor', 1) if source and mat.name != 'Glass' else (.12 if mat.name == 'Glass' else float(shader.inputs['Roughness'].default_value)),
                     'cutoff': -1 if glass else (.5 if shader.inputs['Alpha'].is_linked else 0),
                     'wrap_s': source.get('_samplers',{}).get('colour',(10497,10497))[0], 'wrap_t': source.get('_samplers',{}).get('colour',(10497,10497))[1]})
indices = {m: i for i, m in enumerate(used)}
vertices = []
for number,obj in enumerate(objects):
    evaluated = obj.evaluated_get(deps); mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        loops=np.empty(len(mesh.loops),dtype=np.int32); mesh.loops.foreach_get('vertex_index',loops)
        positions=np.empty(len(mesh.vertices)*3,dtype=np.float32); mesh.vertices.foreach_get('co',positions);positions=positions.reshape(-1,3)
        normals=np.empty(len(mesh.corner_normals)*3,dtype=np.float32); mesh.corner_normals.foreach_get('vector',normals);normals=normals.reshape(-1,3)
        uv=np.zeros((len(loops),2),dtype=np.float32)
        if mesh.uv_layers.active: mesh.uv_layers.active.data.foreach_get('uv',uv.ravel())
        tri=np.empty(len(mesh.loop_triangles)*3,dtype=np.int32);mesh.loop_triangles.foreach_get('loops',tri)
        material_ids=np.empty(len(mesh.loop_triangles),dtype=np.int32);mesh.loop_triangles.foreach_get('material_index',material_ids)
        material_ids=np.repeat(material_ids,3)
        matrix=np.array(evaluated.matrix_world,dtype=np.float32)
        pos=positions[loops[tri]]@matrix[:3,:3].T+matrix[:3,3]
        norm=normals[tri]@np.linalg.inv(matrix[:3,:3])
        lengths=np.linalg.norm(norm,axis=1)
        bad=lengths<1e-8
        if np.any(bad):
            triangles=pos.reshape(-1,3,3);face=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);face=np.repeat(face,3,axis=0)
            norm[bad]=face[bad];lengths=np.linalg.norm(norm,axis=1)
            # Degenerate source triangles have no visible area; retain a finite fallback normal.
            norm[lengths<1e-8]=[0,0,1];lengths=np.linalg.norm(norm,axis=1)
        norm/=lengths[:,None]
        coord=uv[tri].copy();mapped=np.empty(len(tri),dtype=np.float32)
        for slot in np.unique(material_ids):
            mat=obj.material_slots[int(slot)].material;index=indices[mat];mask=material_ids==slot;mapped[mask]=index
            uv_index=metadata[index]['texcoord']
            if uv_index >= len(mesh.uv_layers):
                if metadata[index]['texture']: raise ValueError('Missing required UV layer: '+obj.name)
            else:
                layer_uv=np.empty((len(loops),2),dtype=np.float32)
                mesh.uv_layers[uv_index].data.foreach_get('uv',layer_uv.ravel())
                coord[mask]=layer_uv[tri][mask]
            transform=metadata[index]['uv_transform']
            if transform:
                scale=transform.get('scale',[1,1]);offset=transform.get('offset',[0,0]);angle=transform.get('rotation',0)
                u=coord[mask,0]*scale[0];v=(1-coord[mask,1])*scale[1]
                coord[mask,0]=np.cos(angle)*u-np.sin(angle)*v+offset[0]
                coord[mask,1]=1-(np.sin(angle)*u+np.cos(angle)*v+offset[1])
        coord[:,1]=1-coord[:,1]
        vertices.append(np.column_stack((pos,norm,coord,mapped)).astype('<f4'))
    finally: evaluated.to_mesh_clear()
array=np.concatenate(vertices)
if not len(array) or not np.isfinite(array).all(): raise ValueError('Invalid city geometry')
np.save(ROOT / 'geometry.npy', array)
(ROOT / 'materials.json').write_text(json.dumps(metadata, indent=2) + '\n')
summary = {'objects': len(objects), 'vertices': len(array), 'triangles': len(array)//3, 'materials': len(metadata), 'bounds_blender': [array[:, :3].min(axis=0).tolist(), array[:, :3].max(axis=0).tolist()], 'cars': None if SUPPLIED else 5, 'people': None if SUPPLIED else 6, 'facade_bays': None if SUPPLIED else 2 * SIDE_BAYS + 2 * END_BAYS, 'static': True, 'collision_boxes':len(COLLIDERS), 'car_decimation':.20}
(ROOT / 'assembly.json').write_text(json.dumps(summary, indent=2) + '\n')
print('CITY ASSEMBLED', json.dumps(summary))
