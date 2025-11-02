# tests/test_scene_generation.py
import unittest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

from core.scene_generator import TextTo3DGenerator, AdvancedSceneGenerator
from core.nerf_renderer import NeRFRenderer, NeuralRadianceField
from core.diffusion_model import Diffusion3DModel
from core.mesh_converter import MeshGenerator

class TestSceneGeneration(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_prompt = "A simple test cube"
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_text_to_3d_generator_initialization(self):
        generator = TextTo3DGenerator(device=self.device)
        self.assertIsNotNone(generator)
        self.assertIsNotNone(generator.diffusion_model)
        self.assertIsNotNone(generator.clip_model)
    
    def test_scene_generation_basic(self):
        generator = TextTo3DGenerator(device=self.device)
        
        scene_data = generator.generate_scene(
            text_prompt=self.test_prompt,
            resolution=64,
            output_format="Point Cloud"
        )
        
        self.assertIn('point_cloud', scene_data)
        self.assertIn('metadata', scene_data)
        self.assertGreater(len(scene_data['point_cloud'].points), 0)
    
    def test_nerf_renderer_initialization(self):
        renderer = NeRFRenderer(device=self.device)
        self.assertIsNotNone(renderer)
        self.assertIsNotNone(renderer.renderer)
    
    def test_mesh_converter_export(self):
        generator = TextTo3DGenerator(device=self.device)
        mesh_converter = MeshGenerator()
        
        scene_data = generator.generate_scene(
            text_prompt=self.test_prompt,
            resolution=64
        )
        
        export_data = mesh_converter.export_scene(
            {'mesh_data': scene_data['mesh']},
            format='obj'
        )
        
        self.assertIn('file_data', export_data)
        self.assertIn('filename', export_data)
        self.assertGreater(len(export_data['file_data']), 0)
    
    def test_diffusion_model_initialization(self):
        diffusion_model = Diffusion3DModel(device=self.device)
        self.assertIsNotNone(diffusion_model)
        self.assertIsNotNone(diffusion_model.unet)
        self.assertIsNotNone(diffusion_model.text_encoder)
    
    def test_advanced_scene_generator(self):
        advanced_generator = AdvancedSceneGenerator(device=self.device)
        
        scene_composition = {
            'objects': [
                {'type': 'chair', 'position': [0.5, 0, 0], 'scale': 1.0},
                {'type': 'table', 'position': [0, 0, 0], 'scale': 1.2}
            ],
            'layout': 'organized'
        }
        
        complex_scene = advanced_generator.generate_complex_scene(
            text_prompt=self.test_prompt,
            scene_composition=scene_composition,
            enable_lighting=True
        )
        
        self.assertIn('composed_meshes', complex_scene)
        self.assertIn('lighting', complex_scene)
        self.assertGreater(len(complex_scene['composed_meshes']), 0)
    
    def test_neural_radiance_field_architecture(self):
        nerf_model = NeuralRadianceField().to(self.device)
        
        test_points = torch.randn(10, 3).to(self.device)
        test_dirs = torch.randn(10, 3).to(self.device)
        
        output = nerf_model(test_points, test_dirs)
        
        self.assertEqual(output.shape, (10, 4))
    
    def test_performance_benchmark(self):
        from scripts.performance_benchmark import PerformanceBenchmark
        
        benchmark = PerformanceBenchmark()
        test_prompts = ["Test scene 1", "Test scene 2"]
        
        results = benchmark.benchmark_generation(test_prompts, resolutions=[128])
        
        self.assertIn('generation', benchmark.results)
        self.assertIn(128, results['times'])
    
    def test_batch_processing(self):
        test_prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        
        test_csv = os.path.join(self.temp_dir, "test_prompts.csv")
        with open(test_csv, 'w') as f:
            f.write("prompt\n")
            for prompt in test_prompts:
                f.write(f"{prompt}\n")
        
        from scripts.batch_processor import process_batch
        
        config = {
            'resolution': 64,
            'format': 'obj',
            'export_format': 'obj'
        }
        
        results = process_batch(test_csv, self.temp_dir, config)
        
        self.assertEqual(len(results), len(test_prompts))
        self.assertTrue(any(r['status'] == 'success' for r in results))

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_prompt = "Integration test scene"
    
    def test_end_to_end_generation(self):
        generator = TextTo3DGenerator(device=self.device)
        renderer = NeRFRenderer(device=self.device)
        mesh_converter = MeshGenerator()
        
        scene_data = generator.generate_scene(
            text_prompt=self.test_prompt,
            resolution=128
        )
        
        rendered_scene = renderer.render_scene(scene_data)
        
        export_data = mesh_converter.export_scene(rendered_scene, format='obj')
        
        self.assertIn('file_data', export_data)
        self.assertGreater(len(export_data['file_data']), 0)
        self.assertIn('mesh_data', rendered_scene)
    
    def test_advanced_features_integration(self):
        advanced_generator = AdvancedSceneGenerator(device=self.device)
        renderer = NeRFRenderer(device=self.device)
        
        complex_scene = advanced_generator.generate_complex_scene(
            text_prompt=self.test_prompt,
            num_objects=3,
            enable_lighting=True
        )
        
        rendered_scene = renderer.render_scene(complex_scene)
        
        self.assertIn('composed_meshes', complex_scene)
        self.assertIn('lighting', complex_scene)
        self.assertIn('rendered_views', rendered_scene)

if __name__ == '__main__':
    unittest.main()