# scripts/performance_benchmark.py
import time
import torch
import numpy as np
import json
from pathlib import Path
from core.scene_generator import TextTo3DGenerator
from core.nerf_renderer import NeRFRenderer
from core.mesh_converter import MeshGenerator
import psutil
import GPUtil

class PerformanceBenchmark:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
    
    def benchmark_generation(self, prompts, resolutions=[128, 256, 512]):
        generator = TextTo3DGenerator(device=self.device)
        renderer = NeRFRenderer(device=self.device)
        
        generation_times = {}
        memory_usage = {}
        
        for resolution in resolutions:
            times = []
            memory_measurements = []
            
            for prompt in prompts[:3]:
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                scene_data = generator.generate_scene(
                    text_prompt=prompt,
                    resolution=resolution
                )
                
                rendered_scene = renderer.render_scene(scene_data)
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                times.append(end_time - start_time)
                memory_measurements.append(end_memory - start_memory)
            
            generation_times[resolution] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times)
            }
            
            memory_usage[resolution] = {
                'mean_mb': np.mean(memory_measurements) / (1024 * 1024),
                'max_mb': np.max(memory_measurements) / (1024 * 1024)
            }
        
        self.results['generation'] = {
            'times': generation_times,
            'memory': memory_usage
        }
        
        return self.results['generation']
    
    def benchmark_export(self, scene_data, formats=['obj', 'glb', 'ply']):
        mesh_converter = MeshGenerator()
        
        export_times = {}
        file_sizes = {}
        
        for format in formats:
            times = []
            sizes = []
            
            for _ in range(5):
                start_time = time.time()
                
                export_data = mesh_converter.export_scene(
                    scene_data, format=format
                )
                
                end_time = time.time()
                
                times.append(end_time - start_time)
                sizes.append(len(export_data['file_data']))
            
            export_times[format] = {
                'mean': np.mean(times),
                'std': np.std(times)
            }
            
            file_sizes[format] = {
                'mean_bytes': np.mean(sizes),
                'mean_mb': np.mean(sizes) / (1024 * 1024)
            }
        
        self.results['export'] = {
            'times': export_times,
            'sizes': file_sizes
        }
        
        return self.results['export']
    
    def benchmark_memory(self):
        memory_stats = {
            'cpu_ram': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'used_percent': psutil.virtual_memory().percent
            }
        }
        
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            gpu_stats = []
            
            for gpu in gpus:
                gpu_stats.append({
                    'name': gpu.name,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_free_mb': gpu.memoryFree,
                    'utilization_percent': gpu.load * 100
                })
            
            memory_stats['gpu'] = gpu_stats
        
        self.results['memory'] = memory_stats
        
        return memory_stats
    
    def benchmark_throughput(self, num_scenes=10):
        generator = TextTo3DGenerator(device=self.device)
        
        prompts = [f"Test scene {i}" for i in range(num_scenes)]
        
        start_time = time.time()
        
        for prompt in prompts:
            scene_data = generator.generate_scene(
                text_prompt=prompt,
                resolution=128
            )
        
        end_time = time.time()
        
        total_time = end_time - start_time
        scenes_per_second = num_scenes / total_time
        
        self.results['throughput'] = {
            'total_scenes': num_scenes,
            'total_time_seconds': total_time,
            'scenes_per_second': scenes_per_second,
            'seconds_per_scene': total_time / num_scenes
        }
        
        return self.results['throughput']
    
    def _get_memory_usage(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        else:
            process = psutil.Process()
            return process.memory_info().rss
    
    def run_complete_benchmark(self, test_prompts=None):
        if test_prompts is None:
            test_prompts = [
                "A simple cube on a table",
                "A living room with sofa and TV",
                "A futuristic spaceship interior"
            ]
        
        print("Starting comprehensive performance benchmark...")
        
        print("1. Benchmarking memory...")
        self.benchmark_memory()
        
        print("2. Benchmarking generation performance...")
        self.benchmark_generation(test_prompts)
        
        print("3. Benchmarking throughput...")
        self.benchmark_throughput()
        
        print("4. Generating test scene for export benchmark...")
        generator = TextTo3DGenerator(device=self.device)
        test_scene = generator.generate_scene(
            text_prompt=test_prompts[0],
            resolution=256
        )
        
        print("5. Benchmarking export performance...")
        self.benchmark_export(test_scene)
        
        print("Benchmark completed!")
        
        return self.results
    
    def save_results(self, output_path='benchmark_results.json'):
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        
        return output_path

def main():
    benchmark = PerformanceBenchmark()
    
    results = benchmark.run_complete_benchmark()
    
    benchmark.save_results()
    
    print("\nBenchmark Summary:")
    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"Device: {benchmark.device}")
    
    if 'throughput' in results:
        throughput = results['throughput']
        print(f"Throughput: {throughput['scenes_per_second']:.2f} scenes/second")
    
    if 'generation' in results:
        gen_times = results['generation']['times']
        print(f"Average generation time (256px): {gen_times[256]['mean']:.2f}s")

if __name__ == "__main__":
    main()