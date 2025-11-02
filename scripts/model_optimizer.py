# scripts/model_optimizer.py
import torch
import torch.nn as nn
import argparse
import json
from core.nerf_renderer import NeuralRadianceField
from core.diffusion_model import Diffusion3DModel
from pathlib import Path

class ModelOptimizer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    def optimize_nerf_model(self, model_path, output_path, optimization_level='high'):
        model = NeuralRadianceField().to(self.device)
        
        if model_path and Path(model_path).exists():
            model.load_state_dict(torch.load(model_path))
        
        if optimization_level == 'high':
            optimized_model = self._apply_high_optimization(model)
        elif optimization_level == 'medium':
            optimized_model = self._apply_medium_optimization(model)
        else:
            optimized_model = self._apply_low_optimization(model)
        
        torch.save(optimized_model.state_dict(), output_path)
        
        stats = self._calculate_model_stats(optimized_model)
        
        return {
            'output_path': output_path,
            'optimization_level': optimization_level,
            'stats': stats
        }
    
    def _apply_high_optimization(self, model):
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        
        return quantized_model
    
    def _apply_medium_optimization(self, model):
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        scripted_model = torch.jit.script(model)
        
        return scripted_model
    
    def _apply_low_optimization(self, model):
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def _calculate_model_stats(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        memory_size = total_params * 4
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'estimated_memory_mb': memory_size / (1024 * 1024),
            'optimization_applied': True
        }
    
    def optimize_diffusion_model(self, model_type, output_path):
        diffusion_model = Diffusion3DModel(device=self.device)
        
        optimized_components = {}
        
        if hasattr(diffusion_model, 'unet'):
            diffusion_model.unet.eval()
            for param in diffusion_model.unet.parameters():
                param.requires_grad = False
            
            optimized_components['unet'] = 'frozen'
        
        if hasattr(diffusion_model, 'text_encoder'):
            diffusion_model.text_encoder.eval()
            for param in diffusion_model.text_encoder.parameters():
                param.requires_grad = False
            
            optimized_components['text_encoder'] = 'frozen'
        
        torch.save({
            'model_type': model_type,
            'optimized_components': optimized_components,
            'device': self.device
        }, output_path)
        
        return {
            'output_path': output_path,
            'optimized_components': list(optimized_components.keys()),
            'model_type': model_type
        }

def main():
    parser = argparse.ArgumentParser(description='Optimize 3D generation models')
    parser.add_argument('--model_type', type=str, required=True, 
                       choices=['nerf', 'diffusion'], help='Type of model to optimize')
    parser.add_argument('--input_path', type=str, help='Input model path')
    parser.add_argument('--output_path', type=str, required=True, help='Output model path')
    parser.add_argument('--optimization', type=str, default='medium',
                       choices=['low', 'medium', 'high'], help='Optimization level')
    
    args = parser.parse_args()
    
    optimizer = ModelOptimizer()
    
    if args.model_type == 'nerf':
        result = optimizer.optimize_nerf_model(
            args.input_path, args.output_path, args.optimization
        )
    elif args.model_type == 'diffusion':
        result = optimizer.optimize_diffusion_model(
            args.model_type, args.output_path
        )
    
    print(f"Optimization completed: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main()