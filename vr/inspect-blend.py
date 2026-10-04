"""Read-only scene inventory; does not save or render the source file."""
import bpy
import json
from collections import Counter
objects = bpy.context.scene.objects
meshes = [obj for obj in objects if obj.type == 'MESH']
summary = {
    'blender': bpy.app.version_string,
    'scene': bpy.context.scene.name,
    'objects_by_type': dict(Counter(obj.type for obj in objects)),
    'mesh_vertices': sum(len(obj.data.vertices) for obj in meshes),
    'mesh_polygons': sum(len(obj.data.polygons) for obj in meshes),
    'materials': len(bpy.data.materials),
    'images': [{'name': img.name, 'size': list(img.size), 'packed': bool(img.packed_file), 'source': img.source} for img in bpy.data.images],
    'units': bpy.context.scene.unit_settings.system,
    'unit_scale': bpy.context.scene.unit_settings.scale_length,
    'meshes': [{'name': obj.name, 'vertices': len(obj.data.vertices), 'polygons': len(obj.data.polygons), 'dimensions': list(obj.dimensions), 'location': list(obj.location), 'modifiers': [mod.type for mod in obj.modifiers]} for obj in meshes][:60],
    'material_nodes': dict(Counter(node.bl_idname for mat in bpy.data.materials if mat.use_nodes and mat.node_tree for node in mat.node_tree.nodes)),
}
print('BLEND_INVENTORY ' + json.dumps(summary))
