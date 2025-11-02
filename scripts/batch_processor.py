# scripts/batch_processor.py
import argparse
import json
import os
import sys
from pathlib import Path
from core.scene_generator import TextTo3DGenerator
from core.nerf_renderer import NeRFRenderer
from core.mesh_converter import MeshGenerator
import pandas as pd
from tqdm import tqdm

def process_batch(input_file, output_dir, config):
    generator = TextTo3DGenerator()
    renderer = NeRFRenderer()
    mesh_converter = MeshGenerator()
    
    os.makedirs(output_dir, exist_ok=True)
    
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
        prompts = df['prompt'].tolist()
    elif input_file.endswith('.json'):
        with open(input_file, 'r') as f:
            data = json.load(f)
        prompts = [item['prompt'] for item in data]
    elif input_file.endswith('.txt'):
        with open(input_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Unsupported input file format")
    
    results = []
    
    for i, prompt in enumerate(tqdm(promits, desc="Processing prompts")):
        try:
            scene_data = generator.generate_scene(
                text_prompt=prompt,
                resolution=config.get('resolution', 256),
                output_format=config.get('format', 'NeRF')
            )
            
            rendered_scene = renderer.render_scene(
                scene_data=scene_data,
                output_format=config.get('format', 'NeRF')
            )
            
            export_data = mesh_converter.export_scene(
                rendered_scene,
                format=config.get('export_format', 'glb')
            )
            
            output_filename = f"scene_{i:04d}.{config.get('export_format', 'glb')}"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'wb') as f:
                f.write(export_data['file_data'])
            
            results.append({
                'prompt': prompt,
                'output_file': output_path,
                'status': 'success',
                'vertices': len(scene_data['mesh'].vertices) if scene_data['mesh'] else 0,
                'faces': len(scene_data['mesh'].triangles) if scene_data['mesh'] else 0
            })
            
        except Exception as e:
            results.append({
                'prompt': prompt,
                'output_file': None,
                'status': 'failed',
                'error': str(e)
            })
    
    results_file = os.path.join(output_dir, 'processing_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"Processing complete: {success_count}/{len(prompts)} successful")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Batch process 3D scene generation')
    parser.add_argument('--input', type=str, required=True, help='Input file (CSV, JSON, or TXT)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--resolution', type=int, default=256, help='Output resolution')
    parser.add_argument('--format', type=str, default='glb', help='Output format')
    parser.add_argument('--num_views', type=int, default=8, help='Number of views')
    
    args = parser.parse_args()
    
    config = {
        'resolution': args.resolution,
        'format': args.format,
        'num_views': args.num_views,
        'export_format': args.format
    }
    
    process_batch(args.input, args.output, config)

if __name__ == "__main__":
    main()