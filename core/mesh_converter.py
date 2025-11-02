# core/mesh_converter.py
import torch
import numpy as np
import open3d as o3d
import trimesh
from typing import Dict, Any, List, Tuple
import tempfile
import os

class MeshGenerator:
    def __init__(self):
        self.supported_formats = ['obj', 'gltf', 'glb', 'fbx', 'ply', 'stl', 'usdz']
    
    def export_scene(self, rendered_scene: Dict[str, Any], format: str = 'obj',
                   include_textures: bool = True, include_materials: bool = True) -> Dict[str, Any]:
        
        if 'mesh_data' not in rendered_scene and 'point_cloud' not in rendered_scene:
            raise ValueError("No mesh or point cloud data available for export")
        
        if format.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported formats: {self.supported_formats}")
        
        if 'mesh_data' in rendered_scene:
            mesh = rendered_scene['mesh_data']
        else:
            mesh = self._point_cloud_to_mesh(rendered_scene['point_cloud'])
        
        export_data = self._export_mesh_format(mesh, format, include_textures, include_materials)
        
        preview_image = self._generate_preview_image(mesh)
        
        return {
            'file_data': export_data['data'],
            'filename': export_data['filename'],
            'mime_type': export_data['mime_type'],
            'preview_image': preview_image,
            'format': format,
            'file_size': len(export_data['data'])
        }
    
    def _export_mesh_format(self, mesh: o3d.geometry.TriangleMesh, format: str,
                          include_textures: bool, include_materials: bool) -> Dict[str, Any]:
        
        with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            if format == 'obj':
                o3d.io.write_triangle_mesh(temp_path, mesh, write_ascii=False)
                mime_type = 'model/obj'
            elif format == 'ply':
                o3d.io.write_triangle_mesh(temp_path, mesh, write_ascii=False)
                mime_type = 'application/octet-stream'
            elif format == 'stl':
                o3d.io.write_triangle_mesh(temp_path, mesh, write_ascii=False)
                mime_type = 'application/octet-stream'
            elif format in ['gltf', 'glb']:
                self._export_gltf(mesh, temp_path, format)
                mime_type = 'model/gltf-binary' if format == 'glb' else 'model/gltf+json'
            else:
                o3d.io.write_triangle_mesh(temp_path, mesh, write_ascii=False)
                mime_type = 'application/octet-stream'
            
            with open(temp_path, 'rb') as f:
                file_data = f.read()
            
            filename = f'scene_3d.{format}'
            
            return {
                'data': file_data,
                'filename': filename,
                'mime_type': mime_type
            }
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _export_gltf(self, mesh: o3d.geometry.TriangleMesh, filepath: str, format: str):
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        if mesh.has_vertex_normals():
            normals = np.asarray(mesh.vertex_normals)
        else:
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals)
        
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
        else:
            colors = np.ones_like(vertices)
        
        scene = trimesh.Scene()
        
        mesh_trimesh = trimesh.Trimesh(
            vertices=vertices,
            faces=triangles,
            vertex_normals=normals,
            vertex_colors=colors
        )
        
        scene.add_geometry(mesh_trimesh)
        
        if format == 'glb':
            scene.export(filepath, file_type='glb')
        else:
            scene.export(filepath, file_type='gltf')
    
    def _point_cloud_to_mesh(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        point_cloud.estimate_normals()
        
        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                point_cloud, depth=9
            )
            
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            mesh.compute_vertex_normals()
            
            return mesh
        
        except Exception as e:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, 0.03)
            mesh.compute_vertex_normals()
            return mesh
    
    def _generate_preview_image(self, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=512, height=512, visible=False)
        
        vis.add_geometry(mesh)
        
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().light_on = True
        
        vis.poll_events()
        vis.update_renderer()
        
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        
        image_np = (np.asarray(image) * 255).astype(np.uint8)
        return image_np

class AdvancedMeshConverter(MeshGenerator):
    def __init__(self):
        super().__init__()
    
    def optimize_mesh(self, mesh: o3d.geometry.TriangleMesh, 
                     target_vertices: int = 10000, 
                     preserve_features: bool = True) -> o3d.geometry.TriangleMesh:
        
        current_vertices = len(mesh.vertices)
        
        if current_vertices <= target_vertices:
            return mesh
        
        if preserve_features:
            mesh_simplified = self._feature_preserving_simplification(mesh, target_vertices)
        else:
            mesh_simplified = mesh.simplify_quadric_decimation(target_vertices)
        
        mesh_simplified.compute_vertex_normals()
        
        return mesh_simplified
    
    def _feature_preserving_simplification(self, mesh: o3d.geometry.TriangleMesh,
                                         target_vertices: int) -> o3d.geometry.TriangleMesh:
        
        mesh_curvature = self._compute_mesh_curvature(mesh)
        
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        curvature_weights = np.abs(mesh_curvature)
        curvature_weights = (curvature_weights - curvature_weights.min()) / (curvature_weights.max() - curvature_weights.min())
        
        protection_mask = curvature_weights > 0.7
        
        non_protected_indices = np.where(~protection_mask)[0]
        
        if len(non_protected_indices) > target_vertices * 0.7:
            vertices_to_remove = len(non_protected_indices) - int(target_vertices * 0.7)
            remove_indices = np.random.choice(non_protected_indices, vertices_to_remove, replace=False)
        else:
            remove_indices = np.array([])
        
        keep_mask = np.ones(len(vertices), dtype=bool)
        keep_mask[remove_indices] = False
        
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(keep_mask)[0])}
        
        new_vertices = vertices[keep_mask]
        
        new_triangles = []
        for triangle in triangles:
            if all(keep_mask[triangle]):
                new_triangle = [vertex_map[idx] for idx in triangle]
                new_triangles.append(new_triangle)
        
        new_triangles = np.array(new_triangles)
        
        simplified_mesh = o3d.geometry.TriangleMesh()
        simplified_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
        simplified_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
        
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
            new_colors = colors[keep_mask]
            simplified_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
        
        if mesh.has_vertex_normals():
            normals = np.asarray(mesh.vertex_normals)
            new_normals = normals[keep_mask]
            simplified_mesh.vertex_normals = o3d.utility.Vector3dVector(new_normals)
        
        return simplified_mesh
    
    def _compute_mesh_curvature(self, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        curvature = np.zeros(len(vertices))
        
        for i, vertex in enumerate(vertices):
            connected_triangles = triangles[np.any(triangles == i, axis=1)]
            
            if len(connected_triangles) == 0:
                continue
            
            connected_vertices = set()
            for triangle in connected_triangles:
                connected_vertices.update(triangle)
            connected_vertices.remove(i)
            
            neighbor_vectors = vertices[list(connected_vertices)] - vertex
            neighbor_distances = np.linalg.norm(neighbor_vectors, axis=1)
            
            if len(neighbor_vectors) < 3:
                curvature[i] = 0.0
                continue
            
            neighbor_directions = neighbor_vectors / neighbor_distances[:, np.newaxis]
            
            avg_normal = np.mean(neighbor_directions, axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)
            
            curvature[i] = 1.0 - np.abs(np.dot(neighbor_directions, avg_normal)).mean()
        
        return curvature
    
    def generate_textures(self, mesh: o3d.geometry.TriangleMesh, 
                         base_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
                         material_type: str = "diffuse") -> o3d.geometry.TriangleMesh:
        
        if not mesh.has_vertex_colors():
            base_color_array = np.full((len(mesh.vertices), 3), base_color)
            mesh.vertex_colors = o3d.utility.Vector3dVector(base_color_array)
        
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        
        colors = np.asarray(mesh.vertex_colors)
        
        if material_type == "metallic":
            metallic_factor = 0.8
            roughness_factor = 0.2
            
            fresnel = np.abs(np.dot(normals, [0, 1, 0]))
            metallic_colors = base_color * (1 - metallic_factor) + np.array([0.9, 0.9, 1.0]) * metallic_factor
            colors = colors * (1 - fresnel[:, np.newaxis]) + metallic_colors * fresnel[:, np.newaxis]
        
        elif material_type == "transparent":
            transparency = 0.3
            colors = colors * (1 - transparency) + np.array([1.0, 1.0, 1.0]) * transparency
        
        elif material_type == "emissive":
            emission_strength = 0.5
            colors = colors + np.array([0.5, 0.5, 0.3]) * emission_strength
        
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
        
        return mesh
    
    def create_lod_chain(self, mesh: o3d.geometry.TriangleMesh, 
                        lod_levels: List[int] = [5000, 2000, 500]) -> Dict[int, o3d.geometry.TriangleMesh]:
        
        lod_meshes = {}
        
        for lod_target in lod_levels:
            if len(mesh.vertices) > lod_target:
                lod_mesh = self.optimize_mesh(mesh, lod_target, preserve_features=True)
            else:
                lod_mesh = mesh
            
            lod_meshes[lod_target] = lod_mesh
        
        return lod_meshes
    
    def export_complete_scene(self, rendered_scene: Dict[str, Any], 
                            include_lods: bool = True,
                            include_collision: bool = True) -> Dict[str, Any]:
        
        base_mesh = rendered_scene.get('mesh_data')
        if base_mesh is None and 'point_cloud' in rendered_scene:
            base_mesh = self._point_cloud_to_mesh(rendered_scene['point_cloud'])
        
        export_package = {}
        
        if include_lods:
            lod_meshes = self.create_lod_chain(base_mesh)
            for lod_level, lod_mesh in lod_meshes.items():
                export_data = self.export_scene({'mesh_data': lod_mesh}, 'glb')
                export_package[f'lod_{lod_level}'] = export_data
        
        main_export = self.export_scene({'mesh_data': base_mesh}, 'glb')
        export_package['main'] = main_export
        
        if include_collision:
            collision_mesh = self.optimize_mesh(base_mesh, 1000, preserve_features=False)
            collision_export = self.export_scene({'mesh_data': collision_mesh}, 'obj')
            export_package['collision'] = collision_export
        
        metadata = {
            'num_vertices': len(base_mesh.vertices),
            'num_faces': len(base_mesh.triangles),
            'bounding_box': base_mesh.get_axis_aligned_bounding_box().get_extent().tolist(),
            'lod_levels': list(lod_meshes.keys()) if include_lods else []
        }
        
        export_package['metadata'] = metadata
        
        return export_package