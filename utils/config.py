# utils/config.py
import yaml
from pathlib import Path
from typing import Dict, Any
import os

def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    config_path = Path(config_path)
    
    if not config_path.exists():
        default_config = {
            "generation": {
                "model_architecture": "NeRF + Diffusion",
                "default_resolution": 256,
                "num_views": 8,
                "guidance_scale": 7.5,
                "consistency_weight": 0.5
            },
            "rendering": {
                "output_format": "NeRF",
                "enable_lighting": True,
                "enable_materials": True,
                "render_quality": "high"
            },
            "export": {
                "preferred_format": "glb",
                "include_textures": True,
                "include_materials": True,
                "generate_lods": True
            },
            "performance": {
                "use_gpu": True,
                "batch_size": 1,
                "cache_models": True,
                "optimize_memory": True
            },
            "visualization": {
                "interactive_quality": "medium",
                "enable_animations": True,
                "default_camera": "perspective"
            }
        }
        save_config(default_config, config_path)
        return default_config
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], config_path: str = "configs/default.yaml"):
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_generation_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("generation", {})

def get_rendering_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("rendering", {})

def get_export_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("export", {})

def update_config(section: str, key: str, value: Any):
    config = load_config()
    
    if section not in config:
        config[section] = {}
    
    config[section][key] = value
    save_config(config)

def get_model_paths() -> Dict[str, str]:
    return {
        "diffusion_model": "runwayml/stable-diffusion-v1-5",
        "clip_model": "openai/clip-vit-large-patch14",
        "depth_model": "Intel/dpt-large",
        "normal_model": "local/normal_estimator"
    }

def get_default_export_params() -> Dict[str, Any]:
    config = load_config()
    export_config = config.get("export", {})
    
    return {
        "format": export_config.get("preferred_format", "glb"),
        "include_textures": export_config.get("include_textures", True),
        "include_materials": export_config.get("include_materials", True),
        "generate_lods": export_config.get("generate_lods", True)
    }