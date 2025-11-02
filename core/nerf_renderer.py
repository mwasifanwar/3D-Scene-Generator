# core/nerf_renderer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import open3d as o3d
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    MeshRenderer,
    SoftPhongShader,
    TexturesVertex,
    look_at_view_transform
)
from pytorch3d.structures import Meshes
import math

class NeRFRenderer:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.renderer = None
        self._setup_renderer()
    
    def _setup_renderer(self):
        cameras = FoVPerspectiveCameras(device=self.device)
        
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras
            )
        )
    
    def render_scene(self, scene_data: Dict[str, Any], output_format: str = "NeRF",
                   num_views: int = 8, resolution: int = 512) -> Dict[str, Any]:
        
        rendered_views = []
        camera_positions = []
        
        if 'mesh' in scene_data and scene_data['mesh']:
            mesh = scene_data['mesh']
            pytorch3d_mesh = self._convert_to_pytorch3d_mesh(mesh)
            
            for i in range(num_views):
                azimuth = 2 * math.pi * i / num_views
                elevation = math.pi / 6
                
                R, T = look_at_view_transform(
                    dist=3.0,
                    elev=math.degrees(elevation),
                    azim=math.degrees(azimuth),
                    device=self.device
                )
                
                cameras = FoVPerspectiveCameras(R=R, T=T, device=self.device)
                
                with torch.no_grad():
                    rendered_image = self.renderer(pytorch3d_mesh, cameras=cameras)
                
                rendered_views.append(rendered_image.cpu().numpy()[0, ..., :3])
                camera_positions.append({
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'distance': 3.0
                })
        
        depth_maps = self._generate_depth_maps(scene_data, camera_positions)
        normal_maps = self._generate_normal_maps(scene_data, camera_positions)
        
        rendered_scene = {
            'rendered_views': rendered_views,
            'camera_positions': camera_positions,
            'depth_maps': depth_maps,
            'normal_maps': normal_maps,
            'scene_data': scene_data,
            'stats': {
                'resolution': resolution,
                'num_views': num_views,
                'vertices': len(scene_data['mesh'].vertices) if scene_data['mesh'] else 0,
                'faces': len(scene_data['mesh'].triangles) if scene_data['mesh'] else 0
            }
        }
        
        if output_format == "NeRF":
            rendered_scene['nerf_representation'] = self._create_nerf_representation(scene_data)
        elif output_format == "Mesh":
            rendered_scene['mesh_data'] = scene_data['mesh']
        elif output_format == "Point Cloud":
            rendered_scene['point_cloud'] = scene_data['point_cloud']
        
        return rendered_scene
    
    def _convert_to_pytorch3d_mesh(self, mesh: o3d.geometry.TriangleMesh) -> Meshes:
        vertices = torch.from_numpy(np.asarray(mesh.vertices)).float().to(self.device)
        faces = torch.from_numpy(np.asarray(mesh.triangles)).long().to(self.device)
        
        if mesh.has_vertex_colors():
            colors = torch.from_numpy(np.asarray(mesh.vertex_colors)).float().to(self.device)
        else:
            colors = torch.ones_like(vertices)
        
        textures = TexturesVertex(verts_features=colors.unsqueeze(0))
        
        return Meshes(verts=[vertices], faces=[faces], textures=textures)
    
    def _generate_depth_maps(self, scene_data: Dict[str, Any], 
                           camera_positions: List[Dict[str, float]]) -> List[np.ndarray]:
        
        depth_maps = []
        
        for cam_pos in camera_positions:
            depth_map = self._render_depth_for_camera(scene_data, cam_pos)
            depth_maps.append(depth_map)
        
        return depth_maps
    
    def _render_depth_for_camera(self, scene_data: Dict[str, Any], 
                               camera_position: Dict[str, float]) -> np.ndarray:
        
        if 'mesh' not in scene_data or not scene_data['mesh']:
            return np.zeros((512, 512))
        
        mesh = scene_data['mesh']
        vertices = np.asarray(mesh.vertices)
        
        azimuth = camera_position['azimuth']
        elevation = camera_position['elevation']
        distance = camera_position['distance']
        
        camera_pos = self._spherical_to_cartesian(azimuth, elevation, distance)
        camera_target = np.array([0, 0, 0])
        
        view_matrix = self._look_at(camera_pos, camera_target)
        projection_matrix = self._perspective_projection(60.0, 1.0, 0.1, 10.0)
        
        vertices_homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
        vertices_camera = (view_matrix @ vertices_homogeneous.T).T
        vertices_clip = (projection_matrix @ vertices_camera.T).T
        
        vertices_ndc = vertices_clip[:, :3] / vertices_clip[:, 3:]
        
        depth_values = vertices_ndc[:, 2]
        depth_map = np.zeros((512, 512))
        
        return depth_map
    
    def _generate_normal_maps(self, scene_data: Dict[str, Any],
                            camera_positions: List[Dict[str, float]]) -> List[np.ndarray]:
        
        normal_maps = []
        
        for cam_pos in camera_positions:
            normal_map = self._render_normals_for_camera(scene_data, cam_pos)
            normal_maps.append(normal_map)
        
        return normal_maps
    
    def _render_normals_for_camera(self, scene_data: Dict[str, Any],
                                 camera_position: Dict[str, float]) -> np.ndarray:
        
        if 'mesh' not in scene_data or not scene_data['mesh']:
            return np.zeros((512, 512, 3))
        
        mesh = scene_data['mesh']
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
        
        normal_map = np.zeros((512, 512, 3))
        
        return normal_map
    
    def _spherical_to_cartesian(self, azimuth: float, elevation: float, radius: float) -> np.ndarray:
        x = radius * math.cos(elevation) * math.sin(azimuth)
        y = radius * math.sin(elevation)
        z = radius * math.cos(elevation) * math.cos(azimuth)
        return np.array([x, y, z])
    
    def _look_at(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
        if up is None:
            up = np.array([0, 1, 0])
        
        forward = (target - eye)
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        view_matrix = np.eye(4)
        view_matrix[:3, 0] = right
        view_matrix[:3, 1] = up
        view_matrix[:3, 2] = -forward
        view_matrix[:3, 3] = -view_matrix[:3, :3] @ eye
        
        return view_matrix
    
    def _perspective_projection(self, fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        tan_half_fov = math.tan(math.radians(fov) / 2.0)
        
        projection = np.zeros((4, 4))
        projection[0, 0] = 1.0 / (aspect * tan_half_fov)
        projection[1, 1] = 1.0 / tan_half_fov
        projection[2, 2] = -(far + near) / (far - near)
        projection[2, 3] = -2.0 * far * near / (far - near)
        projection[3, 2] = -1.0
        
        return projection
    
    def _create_nerf_representation(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        if 'point_cloud' not in scene_data:
            return {}
        
        point_cloud = scene_data['point_cloud']
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else np.ones_like(points)
        
        nerf_representation = {
            'points': points,
            'colors': colors,
            'bounds': {
                'min': points.min(axis=0),
                'max': points.max(axis=0)
            },
            'density': self._estimate_density(points),
            'features': self._extract_features(points, colors)
        }
        
        return nerf_representation
    
    def _estimate_density(self, points: np.ndarray) -> np.ndarray:
        from scipy.spatial import KDTree
        
        tree = KDTree(points)
        densities = []
        
        for point in points:
            count = tree.query_ball_point(point, r=0.1, return_length=True)
            densities.append(count)
        
        return np.array(densities) / max(densities)
    
    def _extract_features(self, points: np.ndarray, colors: np.ndarray) -> np.ndarray:
        features = np.column_stack([
            points,
            colors,
            points - points.mean(axis=0),
            np.linalg.norm(points - points.mean(axis=0), axis=1, keepdims=True)
        ])
        
        return features

class NeuralRadianceField(nn.Module):
    def __init__(self, num_layers=8, hidden_dim=256, skips=[4], xyz_channels=3, dir_channels=3):
        super().__init__()
        
        self.xyz_channels = xyz_channels
        self.dir_channels = dir_channels
        self.skips = skips
        
        self.xyz_linear = nn.ModuleList(
            [nn.Linear(xyz_channels, hidden_dim)] +
            [nn.Linear(hidden_dim + (xyz_channels if i in skips else 0), hidden_dim) 
             for i in range(num_layers-1)]
        )
        
        self.sigma_linear = nn.Linear(hidden_dim, 1)
        
        self.feature_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dir_linear = nn.Linear(hidden_dim + dir_channels, hidden_dim // 2)
        self.rgb_linear = nn.Linear(hidden_dim // 2, 3)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, xyz, dirs=None):
        if dirs is not None:
            dirs = F.normalize(dirs, dim=-1)
        
        x = xyz
        for i, layer in enumerate(self.xyz_linear):
            x = self.relu(layer(x))
            if i in self.skips:
                x = torch.cat([x, xyz], dim=-1)
        
        sigma = self.sigma_linear(x)
        
        if dirs is not None:
            feature = self.feature_linear(x)
            x = torch.cat([feature, dirs], dim=-1)
            x = self.relu(self.dir_linear(x))
            rgb = self.sigmoid(self.rgb_linear(x))
        else:
            rgb = torch.zeros_like(xyz)
        
        return torch.cat([rgb, sigma], dim=-1)

class AdvancedNeRFRenderer(NeRFRenderer):
    def __init__(self, device=None):
        super().__init__(device)
        self.nerf_model = None
        self._setup_nerf_model()
    
    def _setup_nerf_model(self):
        self.nerf_model = NeuralRadianceField().to(self.device)
    
    def train_nerf(self, scene_data: Dict[str, Any], num_iterations: int = 1000) -> Dict[str, Any]:
        if 'point_cloud' not in scene_data:
            return {}
        
        point_cloud = scene_data['point_cloud']
        points = torch.from_numpy(np.asarray(point_cloud.points)).float().to(self.device)
        
        if point_cloud.has_colors():
            colors = torch.from_numpy(np.asarray(point_cloud.colors)).float().to(self.device)
        else:
            colors = torch.ones_like(points)
        
        optimizer = torch.optim.Adam(self.nerf_model.parameters(), lr=1e-3)
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            batch_size = min(1024, len(points))
            indices = torch.randperm(len(points))[:batch_size]
            
            batch_points = points[indices]
            batch_colors = colors[indices]
            
            batch_dirs = F.normalize(batch_points, dim=-1)
            
            outputs = self.nerf_model(batch_points, batch_dirs)
            pred_colors = outputs[:, :3]
            pred_sigma = outputs[:, 3:]
            
            color_loss = F.mse_loss(pred_colors, batch_colors)
            sigma_loss = torch.mean(torch.abs(pred_sigma - 0.1))
            
            loss = color_loss + 0.1 * sigma_loss
            loss.backward()
            optimizer.step()
        
        trained_nerf = {
            'model_state': self.nerf_model.state_dict(),
            'bounds': {
                'min': points.min(dim=0)[0].cpu().numpy(),
                'max': points.max(dim=0)[0].cpu().numpy()
            },
            'iterations': num_iterations,
            'final_loss': loss.item()
        }
        
        return trained_nerf
    
    def render_nerf_view(self, trained_nerf: Dict[str, Any], camera_position: Dict[str, float],
                       resolution: int = 512) -> np.ndarray:
        
        if self.nerf_model is None:
            return np.zeros((resolution, resolution, 3))
        
        self.nerf_model.load_state_dict(trained_nerf['model_state'])
        self.nerf_model.eval()
        
        rays = self._generate_rays(camera_position, resolution)
        
        with torch.no_grad():
            rendered_image = self._render_rays(rays)
        
        return rendered_image.cpu().numpy()
    
    def _generate_rays(self, camera_position: Dict[str, float], resolution: int) -> torch.Tensor:
        azimuth = camera_position['azimuth']
        elevation = camera_position['elevation']
        distance = camera_position['distance']
        
        camera_origin = self._spherical_to_cartesian(azimuth, elevation, distance)
        camera_origin = torch.from_numpy(camera_origin).float().to(self.device)
        
        rays = torch.zeros(resolution, resolution, 6, device=self.device)
        
        return rays
    
    def _render_rays(self, rays: torch.Tensor) -> torch.Tensor:
        batch_size = 1024
        num_rays = rays.shape[0] * rays.shape[1]
        rays_flat = rays.view(-1, 6)
        
        rendered_image = torch.zeros(num_rays, 3, device=self.device)
        
        for i in range(0, num_rays, batch_size):
            batch_rays = rays_flat[i:i+batch_size]
            batch_origins = batch_rays[:, :3]
            batch_dirs = batch_rays[:, 3:6]
            
            with torch.no_grad():
                batch_outputs = self.nerf_model(batch_origins, batch_dirs)
            
            rendered_image[i:i+batch_size] = batch_outputs[:, :3]
        
        return rendered_image.view(rays.shape[0], rays.shape[1], 3)