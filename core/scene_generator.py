# core/scene_generator.py
import torch
import torch.nn as nn
import numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import open3d as o3d
from typing import Dict, Any, List, Tuple
import math

class TextTo3DGenerator:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion_model = None
        self.clip_model = None
        self.tokenizer = None
        self.load_models()
    
    def load_models(self):
        with torch.no_grad():
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            
            self.diffusion_model = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            self.diffusion_model.scheduler = DPMSolverMultistepScheduler.from_config(
                self.diffusion_model.scheduler.config
            )
    
    def generate_scene(self, text_prompt: str, style_prompt: str = None, 
                      scene_scale: str = "Room", output_format: str = "NeRF",
                      resolution: int = 256, guidance_scale: float = 7.5,
                      num_inference_steps: int = 50) -> Dict[str, Any]:
        
        full_prompt = text_prompt
        if style_prompt:
            full_prompt = f"{text_prompt}, {style_prompt}"
        
        with torch.no_grad():
            clip_embeddings = self._get_clip_embeddings(full_prompt)
            
            generated_images = self._generate_multiview_images(
                prompt=full_prompt,
                num_views=8,
                resolution=resolution,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
            point_cloud = self._images_to_point_cloud(generated_images)
            
            scene_representation = self._create_scene_representation(
                point_cloud=point_cloud,
                clip_embeddings=clip_embeddings,
                scene_scale=scene_scale,
                output_format=output_format
            )
            
            return scene_representation
    
    def _get_clip_embeddings(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_embeddings = self.clip_model(**inputs).last_hidden_state
        
        return text_embeddings
    
    def _generate_multiview_images(self, prompt: str, num_views: int = 8,
                                 resolution: int = 256, guidance_scale: float = 7.5,
                                 num_inference_steps: int = 50) -> List[torch.Tensor]:
        
        images = []
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        
        for angle in angles:
            azimuth = angle
            elevation = 0.0
            
            view_prompt = self._add_view_description(prompt, azimuth, elevation)
            
            with torch.no_grad():
                image = self.diffusion_model(
                    prompt=view_prompt,
                    height=resolution,
                    width=resolution,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                ).images[0]
            
            image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            images.append(image_tensor)
        
        return images
    
    def _add_view_description(self, prompt: str, azimuth: float, elevation: float) -> str:
        azimuth_deg = math.degrees(azimuth) % 360
        elevation_deg = math.degrees(elevation)
        
        if elevation_deg > 30:
            view_desc = "aerial view"
        elif elevation_deg < -30:
            view_desc = "low angle view"
        else:
            if 45 <= azimuth_deg < 135:
                view_desc = "side view"
            elif 135 <= azimuth_deg < 225:
                view_desc = "back view"
            elif 225 <= azimuth_deg < 315:
                view_desc = "side view"
            else:
                view_desc = "front view"
        
        return f"{prompt}, {view_desc}, photorealistic, high detail, 8k"
    
    def _images_to_point_cloud(self, images: List[torch.Tensor]) -> o3d.geometry.PointCloud:
        depth_maps = self._estimate_depth_maps(images)
        point_cloud = self._depth_maps_to_point_cloud(images, depth_maps)
        
        return point_cloud
    
    def _estimate_depth_maps(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        depth_maps = []
        
        for image in images:
            image_gray = torch.mean(image, dim=-1) if len(image.shape) > 2 else image
            depth_map = self._compute_relative_depth(image_gray)
            depth_maps.append(depth_map)
        
        return depth_maps
    
    def _compute_relative_depth(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        grad_x = nn.functional.conv2d(image.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
        grad_y = nn.functional.conv2d(image.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
        
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        depth_map = 1.0 / (1.0 + gradient_magnitude)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        return depth_map.squeeze()
    
    def _depth_maps_to_point_cloud(self, images: List[torch.Tensor], 
                                 depth_maps: List[torch.Tensor]) -> o3d.geometry.PointCloud:
        
        points = []
        colors = []
        
        num_views = len(images)
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        
        for i, (image, depth_map, angle) in enumerate(zip(images, depth_maps, angles)):
            h, w = depth_map.shape
            
            y_coords, x_coords = torch.meshgrid(
                torch.linspace(-1, 1, h),
                torch.linspace(-1, 1, w),
                indexing='ij'
            )
            
            depth_values = depth_map.flatten()
            x_coords_flat = x_coords.flatten()
            y_coords_flat = y_coords.flatten()
            
            x_3d = x_coords_flat * depth_values
            y_3d = y_coords_flat * depth_values
            z_3d = depth_values
            
            rotation_matrix = self._euler_angles_to_matrix(torch.tensor([0.0, angle, 0.0]))
            
            points_3d = torch.stack([x_3d, y_3d, z_3d], dim=1)
            points_rotated = torch.matmul(points_3d, rotation_matrix.T)
            
            if len(image.shape) == 3:
                image_colors = image.reshape(-1, 3)
            else:
                image_colors = image.unsqueeze(-1).repeat(1, 1, 3).reshape(-1, 3)
            
            points.append(points_rotated)
            colors.append(image_colors)
        
        all_points = torch.cat(points, dim=0)
        all_colors = torch.cat(colors, dim=0)
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(all_points.numpy())
        point_cloud.colors = o3d.utility.Vector3dVector(all_colors.numpy())
        
        point_cloud = point_cloud.voxel_down_sample(voxel_size=0.05)
        
        return point_cloud
    
    def _euler_angles_to_matrix(self, euler_angles: torch.Tensor) -> torch.Tensor:
        x, y, z = euler_angles
        
        cos_x, sin_x = torch.cos(x), torch.sin(x)
        cos_y, sin_y = torch.cos(y), torch.sin(y)
        cos_z, sin_z = torch.cos(z), torch.sin(z)
        
        rx = torch.tensor([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        
        ry = torch.tensor([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        
        rz = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ])
        
        return torch.matmul(rz, torch.matmul(ry, rx))
    
    def _create_scene_representation(self, point_cloud: o3d.geometry.PointCloud,
                                   clip_embeddings: torch.Tensor, scene_scale: str,
                                   output_format: str) -> Dict[str, Any]:
        
        scene_bbox = point_cloud.get_axis_aligned_bounding_box()
        scene_center = scene_bbox.get_center()
        scene_extent = scene_bbox.get_extent()
        
        scale_factors = {
            "Small Object": 0.5,
            "Room": 1.0,
            "Building": 2.0,
            "Landscape": 5.0
        }
        
        scale_factor = scale_factors.get(scene_scale, 1.0)
        scene_extent = scene_extent * scale_factor
        
        mesh = self._point_cloud_to_mesh(point_cloud)
        
        scene_representation = {
            'point_cloud': point_cloud,
            'mesh': mesh,
            'clip_embeddings': clip_embeddings.cpu(),
            'metadata': {
                'bbox': {
                    'center': scene_center,
                    'extent': scene_extent,
                    'min_bound': scene_bbox.min_bound,
                    'max_bound': scene_bbox.max_bound
                },
                'scale': scene_scale,
                'format': output_format,
                'num_points': len(point_cloud.points),
                'num_faces': len(mesh.triangles) if mesh else 0
            }
        }
        
        return scene_representation
    
    def _point_cloud_to_mesh(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        try:
            point_cloud.estimate_normals()
            
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                point_cloud, depth=8
            )[0]
            
            mesh.compute_vertex_normals()
            
            return mesh
        except:
            return o3d.geometry.TriangleMesh()

class AdvancedSceneGenerator(TextTo3DGenerator):
    def __init__(self, device=None):
        super().__init__(device)
        self.depth_estimator = None
        self.normal_estimator = None
    
    def generate_complex_scene(self, text_prompt: str, style_prompt: str = None,
                             scene_composition: Dict[str, Any] = None,
                             num_objects: int = 5, enable_lighting: bool = True) -> Dict[str, Any]:
        
        base_scene = self.generate_scene(text_prompt, style_prompt)
        
        if scene_composition:
            composed_scene = self._compose_scene_with_objects(base_scene, scene_composition)
        else:
            composed_scene = self._generate_scene_with_objects(base_scene, num_objects)
        
        if enable_lighting:
            composed_scene = self._add_dynamic_lighting(composed_scene)
        
        return composed_scene
    
    def _compose_scene_with_objects(self, base_scene: Dict[str, Any], 
                                  composition: Dict[str, Any]) -> Dict[str, Any]:
        
        objects = composition.get('objects', [])
        layout = composition.get('layout', 'organized')
        
        for obj in objects:
            obj_type = obj.get('type', 'generic')
            position = obj.get('position', [0, 0, 0])
            scale = obj.get('scale', 1.0)
            
            object_mesh = self._generate_object_mesh(obj_type, scale)
            transformed_mesh = self._place_object_in_scene(object_mesh, position, base_scene)
            
            if 'composed_meshes' not in base_scene:
                base_scene['composed_meshes'] = []
            
            base_scene['composed_meshes'].append(transformed_mesh)
        
        return base_scene
    
    def _generate_object_mesh(self, obj_type: str, scale: float) -> o3d.geometry.TriangleMesh:
        if obj_type == "chair":
            mesh = o3d.geometry.TriangleMesh.create_box(width=0.4*scale, height=0.8*scale, depth=0.4*scale)
        elif obj_type == "table":
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3*scale, height=0.7*scale)
        elif obj_type == "lamp":
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.2*scale)
        else:
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.3*scale)
        
        mesh.compute_vertex_normals()
        return mesh
    
    def _place_object_in_scene(self, object_mesh: o3d.geometry.TriangleMesh,
                             position: List[float], scene: Dict[str, Any]) -> o3d.geometry.TriangleMesh:
        
        bbox = scene['metadata']['bbox']
        scene_center = bbox['center']
        scene_extent = bbox['extent']
        
        absolute_position = [
            scene_center[0] + position[0] * scene_extent[0] / 2,
            scene_center[1] + position[1] * scene_extent[1] / 2,
            scene_center[2] + position[2] * scene_extent[2] / 2
        ]
        
        object_mesh.translate(absolute_position)
        return object_mesh
    
    def _generate_scene_with_objects(self, base_scene: Dict[str, Any], num_objects: int) -> Dict[str, Any]:
        object_types = ["chair", "table", "lamp", "plant", "decoration"]
        
        for i in range(num_objects):
            obj_type = object_types[i % len(object_types)]
            position = [
                np.random.uniform(-0.8, 0.8),
                np.random.uniform(-0.8, 0.8),
                np.random.uniform(0.0, 0.5)
            ]
            scale = np.random.uniform(0.5, 1.5)
            
            object_mesh = self._generate_object_mesh(obj_type, scale)
            transformed_mesh = self._place_object_in_scene(object_mesh, position, base_scene)
            
            if 'composed_meshes' not in base_scene:
                base_scene['composed_meshes'] = []
            
            base_scene['composed_meshes'].append(transformed_mesh)
        
        return base_scene
    
    def _add_dynamic_lighting(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        lighting_config = {
            'main_light': {
                'position': [2.0, 2.0, 3.0],
                'intensity': 1.0,
                'color': [1.0, 1.0, 0.9]
            },
            'fill_light': {
                'position': [-2.0, -1.0, 2.0],
                'intensity': 0.3,
                'color': [0.8, 0.8, 1.0]
            },
            'rim_light': {
                'position': [0.0, -3.0, 1.0],
                'intensity': 0.2,
                'color': [1.0, 1.0, 1.0]
            }
        }
        
        scene['lighting'] = lighting_config
        return scene