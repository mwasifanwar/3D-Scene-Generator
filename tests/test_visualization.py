# tests/test_visualization.py
import unittest
import numpy as np
import plotly.graph_objects as go
from utils.visualization import SceneVisualizer, AdvancedSceneVisualizer
import open3d as o3d

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.visualizer = SceneVisualizer()
        self.advanced_visualizer = AdvancedSceneVisualizer()
        
        self.test_mesh = o3d.geometry.TriangleMesh.create_sphere()
        self.test_mesh.compute_vertex_normals()
        
        self.test_point_cloud = o3d.geometry.PointCloud()
        self.test_point_cloud.points = o3d.utility.Vector3dVector(
            np.random.rand(100, 3)
        )
    
    def test_mesh_viewer_creation(self):
        fig = self.visualizer._create_mesh_viewer(self.test_mesh)
        
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)
        self.assertIsInstance(fig.data[0], go.Mesh3d)
    
    def test_point_cloud_viewer_creation(self):
        fig = self.visualizer._create_point_cloud_viewer(self.test_point_cloud)
        
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)
        self.assertIsInstance(fig.data[0], go.Scatter3d)
    
    def test_empty_viewer_creation(self):
        fig = self.visualizer._create_empty_viewer()
        
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 0)
    
    def test_multiview_comparison(self):
        test_views = [np.random.rand(64, 64, 3) for _ in range(4)]
        fig = self.visualizer.create_multiview_comparison(test_views)
        
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 4)
    
    def test_animated_turntable(self):
        test_scene = {'mesh_data': self.test_mesh}
        fig = self.advanced_visualizer.create_animated_turntable(test_scene)
        
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.frames), 0)
    
    def test_quality_metrics_dashboard(self):
        test_metrics = {
            'quality_metrics': {
                'mesh_quality': 0.85,
                'texture_quality': 0.92,
                'geometry_stats': {
                    'vertices': 1000,
                    'faces': 2000,
                    'edges': 3000
                },
                'performance_metrics': {
                    'render_time': 0.5,
                    'export_time': 1.2,
                    'memory_usage': 256
                }
            }
        }
        
        fig = self.advanced_visualizer.create_quality_metrics_dashboard(test_metrics)
        
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 4)
    
    def test_comparison_viewer(self):
        test_scenes = [
            {'mesh_data': self.test_mesh},
            {'point_cloud': self.test_point_cloud},
            {'mesh_data': self.test_mesh}
        ]
        
        titles = ['Scene 1', 'Scene 2', 'Scene 3']
        
        fig = self.advanced_visualizer.create_comparison_viewer(test_scenes, titles)
        
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 3)

if __name__ == '__main__':
    unittest.main()