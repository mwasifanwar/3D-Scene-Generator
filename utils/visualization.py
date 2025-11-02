# utils/visualization.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import open3d as o3d
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from matplotlib import cm

class SceneVisualizer:
    def __init__(self):
        self.colors = {
            'mesh': 'lightblue',
            'points': 'blue',
            'wireframe': 'darkblue',
            'background': 'lightgray'
        }
    
    def create_interactive_viewer(self, rendered_scene: Dict[str, Any]) -> go.Figure:
        if 'mesh_data' in rendered_scene:
            return self._create_mesh_viewer(rendered_scene['mesh_data'])
        elif 'point_cloud' in rendered_scene:
            return self._create_point_cloud_viewer(rendered_scene['point_cloud'])
        else:
            return self._create_empty_viewer()
    
    def _create_mesh_viewer(self, mesh: o3d.geometry.TriangleMesh) -> go.Figure:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
        else:
            colors = np.ones((len(vertices), 3)) * 0.7
        
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        
        i, j, k = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        
        mesh_3d = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            vertexcolor=colors,
            opacity=0.8,
            lighting=dict(
                ambient=0.3,
                diffuse=0.8,
                fresnel=0.1,
                specular=0.5,
                roughness=0.5
            ),
            lightposition=dict(x=100, y=100, z=100)
        )
        
        fig = go.Figure(data=[mesh_3d])
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='white',
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=600
        )
        
        return fig
    
    def _create_point_cloud_viewer(self, point_cloud: o3d.geometry.PointCloud) -> go.Figure:
        points = np.asarray(point_cloud.points)
        
        if point_cloud.has_colors():
            colors = np.asarray(point_cloud.colors)
        else:
            colors = np.ones((len(points), 3)) * 0.7
        
        scatter_3d = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors,
                opacity=0.8
            )
        )
        
        fig = go.Figure(data=[scatter_3d])
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='white'
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=600
        )
        
        return fig
    
    def _create_empty_viewer(self) -> go.Figure:
        fig = go.Figure()
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='white'
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=600,
            annotations=[dict(
                text="No 3D data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )]
        )
        
        return fig
    
    def create_multiview_comparison(self, rendered_views: List[np.ndarray]) -> go.Figure:
        num_views = len(rendered_views)
        cols = min(4, num_views)
        rows = (num_views + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"View {i+1}" for i in range(num_views)]
        )
        
        for i, view in enumerate(rendered_views):
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(
                go.Image(z=view),
                row=row, col=col
            )
        
        fig.update_layout(
            height=200 * rows,
            showlegend=False,
            title_text="Multi-View Comparison"
        )
        
        return fig
    
    def create_depth_visualization(self, depth_maps: List[np.ndarray]) -> go.Figure:
        num_maps = len(depth_maps)
        cols = min(4, num_maps)
        rows = (num_maps + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"Depth {i+1}" for i in range(num_maps)]
        )
        
        for i, depth_map in enumerate(depth_maps):
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(
                go.Heatmap(z=depth_map, colorscale='Viridis'),
                row=row, col=col
            )
        
        fig.update_layout(
            height=200 * rows,
            showlegend=False,
            title_text="Depth Maps"
        )
        
        return fig
    
    def create_normal_visualization(self, normal_maps: List[np.ndarray]) -> go.Figure:
        num_maps = len(normal_maps)
        cols = min(4, num_maps)
        rows = (num_maps + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"Normals {i+1}" for i in range(num_maps)]
        )
        
        for i, normal_map in enumerate(normal_maps):
            row = i // cols + 1
            col = i % cols + 1
            
            rgb_normals = (normal_map + 1) / 2
            
            fig.add_trace(
                go.Image(z=rgb_normals),
                row=row, col=col
            )
        
        fig.update_layout(
            height=200 * rows,
            showlegend=False,
            title_text="Normal Maps"
        )
        
        return fig

class AdvancedSceneVisualizer(SceneVisualizer):
    def __init__(self):
        super().__init__()
    
    def create_animated_turntable(self, rendered_scene: Dict[str, Any], 
                                num_frames: int = 36) -> go.Figure:
        
        if 'mesh_data' not in rendered_scene:
            return self._create_empty_viewer()
        
        mesh = rendered_scene['mesh_data']
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
        else:
            colors = np.ones((len(vertices), 3)) * 0.7
        
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        i, j, k = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        
        frames = []
        for angle in np.linspace(0, 360, num_frames, endpoint=False):
            frames.append(go.Frame(
                data=[go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    vertexcolor=colors,
                    opacity=0.8
                )],
                layout=dict(
                    scene=dict(
                        camera=dict(
                            eye=dict(
                                x=2 * np.cos(np.radians(angle)),
                                y=2 * np.sin(np.radians(angle)),
                                z=1.5
                            )
                        )
                    )
                )
            ))
        
        fig = go.Figure(
            data=[go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                vertexcolor=colors,
                opacity=0.8
            )],
            frames=frames
        )
        
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                            method="animate",
                            args=[None, dict(frame=dict(duration=50, redraw=True),
                                            fromcurrent=True)])]
            )],
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='white'
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=600
        )
        
        return fig
    
    def create_quality_metrics_dashboard(self, scene_metrics: Dict[str, Any]) -> go.Figure:
        metrics = scene_metrics.get('quality_metrics', {})
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Mesh Quality', 'Texture Quality', 'Geometry Analysis', 'Performance'],
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        if 'mesh_quality' in metrics:
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics['mesh_quality'] * 100,
                title={'text': "Mesh Quality"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "blue"}},
            ), row=1, col=1)
        
        if 'texture_quality' in metrics:
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics['texture_quality'] * 100,
                title={'text': "Texture Quality"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "green"}},
            ), row=1, col=2)
        
        if 'geometry_stats' in metrics:
            stats = metrics['geometry_stats']
            fig.add_trace(go.Bar(
                x=list(stats.keys()),
                y=list(stats.values()),
                name="Geometry Stats"
            ), row=2, col=1)
        
        if 'performance_metrics' in metrics:
            perf = metrics['performance_metrics']
            fig.add_trace(go.Bar(
                x=list(perf.keys()),
                y=list(perf.values()),
                name="Performance"
            ), row=2, col=2)
        
        fig.update_layout(height=600, title_text="Scene Quality Metrics Dashboard")
        
        return fig
    
    def create_comparison_viewer(self, scenes: List[Dict[str, Any]], 
                               titles: List[str] = None) -> go.Figure:
        
        if titles is None:
            titles = [f"Scene {i+1}" for i in range(len(scenes))]
        
        num_scenes = len(scenes)
        cols = min(3, num_scenes)
        rows = (num_scenes + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=titles,
            specs=[[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
        )
        
        for i, scene in enumerate(scenes):
            row = i // cols + 1
            col = i % cols + 1
            
            if 'mesh_data' in scene:
                viewer = self._create_mesh_viewer(scene['mesh_data'])
                fig.add_trace(viewer.data[0], row=row, col=col)
            elif 'point_cloud' in scene:
                viewer = self._create_point_cloud_viewer(scene['point_cloud'])
                fig.add_trace(viewer.data[0], row=row, col=col)
        
        fig.update_layout(
            height=400 * rows,
            title_text="Scene Comparison"
        )
        
        for i in range(1, rows * cols + 1):
            fig.update_scenes(
                dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    bgcolor='white'
                ),
                row=(i-1)//cols + 1,
                col=(i-1)%cols + 1
            )
        
        return fig

class RealTimeVisualizer:
    def __init__(self):
        self.current_frame = 0
        self.animation_data = []
    
    def update_realtime_view(self, new_scene_data: Dict[str, Any]):
        self.animation_data.append(new_scene_data)
        self.current_frame += 1
        
        if len(self.animation_data) > 100:
            self.animation_data.pop(0)
    
    def create_realtime_animation(self) -> go.Figure:
        if not self.animation_data:
            return self._create_empty_viewer()
        
        frames = []
        for i, scene_data in enumerate(self.animation_data):
            if 'mesh_data' in scene_data:
                mesh = scene_data['mesh_data']
                vertices = np.asarray(mesh.vertices)
                triangles = np.asarray(mesh.triangles)
                
                x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
                i_idx, j, k = triangles[:, 0], triangles[:, 1], triangles[:, 2]
                
                frame = go.Frame(
                    data=[go.Mesh3d(
                        x=x, y=y, z=z,
                        i=i_idx, j=j, k=k,
                        color='lightblue',
                        opacity=0.8
                    )],
                    name=f"frame_{i}"
                )
                frames.append(frame)
        
        fig = go.Figure(
            data=[go.Mesh3d(
                x=[], y=[], z=[],
                i=[], j=[], k=[],
                color='lightblue',
                opacity=0.8
            )],
            frames=frames
        )
        
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                            method="animate",
                            args=[None, dict(frame=dict(duration=100, redraw=True),
                                            fromcurrent=True)])]
            )]
        )
        
        return fig