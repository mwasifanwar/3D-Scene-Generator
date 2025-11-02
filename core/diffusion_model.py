# core/diffusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, Any, List, Tuple
import numpy as np
from einops import rearrange, repeat

class Diffusion3DModel:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet = None
        self.scheduler = None
        self.text_encoder = None
        self.tokenizer = None
        self.vae = None
        self.load_models()
    
    def load_models(self):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        
        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        self.scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )
    
    def generate_3d_consistent_views(self, text_prompt: str, num_views: int = 8,
                                   resolution: int = 256, guidance_scale: float = 7.5,
                                   consistency_weight: float = 0.5) -> List[torch.Tensor]:
        
        text_embeddings = self._encode_text(text_prompt)
        views = []
        
        base_latents = torch.randn(1, 4, resolution//8, resolution//8, device=self.device)
        
        for view_idx in range(num_views):
            view_embeddings = self._add_view_conditioning(text_embeddings, view_idx, num_views)
            
            view_latents = self._generate_view_latents(
                base_latents=base_latents,
                text_embeddings=view_embeddings,
                guidance_scale=guidance_scale,
                view_idx=view_idx,
                consistency_weight=consistency_weight
            )
            
            view_image = self._decode_latents(view_latents)
            views.append(view_image)
        
        return views
    
    def _encode_text(self, prompt: str) -> torch.Tensor:
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]
        
        return text_embeddings
    
    def _add_view_conditioning(self, text_embeddings: torch.Tensor, 
                             view_idx: int, num_views: int) -> torch.Tensor:
        
        angle = 2 * torch.pi * view_idx / num_views
        view_description = self._get_view_description(angle)
        
        view_inputs = self.tokenizer(
            view_description,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        view_input_ids = view_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            view_embeddings = self.text_encoder(view_input_ids)[0]
        
        combined_embeddings = torch.cat([text_embeddings, view_embeddings], dim=1)
        
        return combined_embeddings
    
    def _get_view_description(self, angle: float) -> str:
        angle_deg = torch.rad2deg(angle) % 360
        
        if 45 <= angle_deg < 135:
            view_desc = "side view"
        elif 135 <= angle_deg < 225:
            view_desc = "back view"
        elif 225 <= angle_deg < 315:
            view_desc = "side view"
        else:
            view_desc = "front view"
        
        return f"{view_desc}, consistent 3D, multi-view"
    
    def _generate_view_latents(self, base_latents: torch.Tensor, text_embeddings: torch.Tensor,
                             guidance_scale: float, view_idx: int, consistency_weight: float) -> torch.Tensor:
        
        self.scheduler.set_timesteps(50)
        latents = base_latents.clone()
        
        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            if view_idx > 0 and consistency_weight > 0:
                noise_pred = self._apply_consistency_constraint(noise_pred, base_latents, consistency_weight)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents
    
    def _apply_consistency_constraint(self, noise_pred: torch.Tensor, base_latents: torch.Tensor,
                                   consistency_weight: float) -> torch.Tensor:
        
        consistency_loss = F.mse_loss(noise_pred, torch.zeros_like(noise_pred))
        noise_pred = noise_pred - consistency_weight * consistency_loss * torch.randn_like(noise_pred)
        
        return noise_pred
    
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = 1 / 0.18215 * latents
        
        with torch.no_grad():
            image = self._vae_decode(latents)
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).astype(np.uint8)
        
        return torch.from_numpy(image).float() / 255.0
    
    def _vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents

class MultiViewDiffusionModel(Diffusion3DModel):
    def __init__(self, device=None):
        super().__init__(device)
        self.cross_view_attention = CrossViewAttention().to(self.device)
    
    def generate_consistent_multiview(self, text_prompt: str, num_views: int = 8,
                                    resolution: int = 256, cross_attention_strength: float = 0.7) -> List[torch.Tensor]:
        
        text_embeddings = self._encode_text(text_prompt)
        
        all_latents = []
        for i in range(num_views):
            latent = torch.randn(1, 4, resolution//8, resolution//8, device=self.device)
            all_latents.append(latent)
        
        all_latents = torch.stack(all_latents)
        
        for t in self.scheduler.timesteps:
            noise_preds = []
            
            for i in range(num_views):
                view_latent = all_latents[i]
                view_embeddings = self._add_view_conditioning(text_embeddings, i, num_views)
                
                latent_model_input = torch.cat([view_latent] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=view_embeddings
                    ).sample
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                
                if num_views > 1:
                    other_latents = torch.cat([all_latents[:i], all_latents[i+1:]])
                    noise_pred = self.cross_view_attention(noise_pred, other_latents, cross_attention_strength)
                
                noise_preds.append(noise_pred)
            
            noise_preds = torch.stack(noise_preds)
            
            for i in range(num_views):
                all_latents[i] = self.scheduler.step(noise_preds[i], t, all_latents[i]).prev_sample
        
        views = []
        for i in range(num_views):
            view_image = self._decode_latents(all_latents[i])
            views.append(view_image)
        
        return views

class CrossViewAttention(nn.Module):
    def __init__(self, dim=320, num_heads=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query: torch.Tensor, keys: torch.Tensor, attention_strength: float = 1.0) -> torch.Tensor:
        batch_size, channels, height, width = query.shape
        
        query_flat = query.view(batch_size, channels, -1).permute(0, 2, 1)
        keys_flat = keys.view(keys.shape[0], keys.shape[1], channels, -1)
        keys_flat = keys_flat.permute(0, 1, 3, 2).reshape(-1, height * width, channels)
        
        Q = self.q_linear(query_flat)
        K = self.k_linear(keys_flat)
        V = self.v_linear(keys_flat)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(-1, batch_size, -1, self.num_heads, self.head_dim).transpose(2, 3)
        V = V.view(-1, batch_size, -1, self.num_heads, self.head_dim).transpose(2, 3)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.dim)
        
        output = self.out_linear(attn_output)
        output = output.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        return query + attention_strength * output

class DepthAwareDiffusion(Diffusion3DModel):
    def __init__(self, device=None):
        super().__init__(device)
        self.depth_estimator = DepthEstimationModel().to(self.device)
    
    def generate_with_depth_guidance(self, text_prompt: str, num_views: int = 8,
                                   depth_consistency: float = 0.8) -> List[torch.Tensor]:
        
        views = self.generate_3d_consistent_views(text_prompt, num_views)
        
        depth_maps = []
        for view in views:
            depth_map = self.depth_estimator.estimate_depth(view.unsqueeze(0).to(self.device))
            depth_maps.append(depth_map.cpu())
        
        refined_views = self._refine_with_depth_consistency(views, depth_maps, depth_consistency)
        
        return refined_views
    
    def _refine_with_depth_consistency(self, views: List[torch.Tensor], 
                                     depth_maps: List[torch.Tensor],
                                     consistency_strength: float) -> List[torch.Tensor]:
        
        refined_views = []
        
        for i, (view, depth_map) in enumerate(zip(views, depth_maps)):
            if i == 0:
                refined_views.append(view)
                continue
            
            prev_depth = depth_maps[i-1]
            depth_diff = torch.abs(depth_map - prev_depth).mean()
            
            if depth_diff > 0.1:
                view_adjusted = view - consistency_strength * depth_diff * torch.randn_like(view)
                refined_views.append(view_adjusted.clamp(0, 1))
            else:
                refined_views.append(view)
        
        return refined_views

class DepthEstimationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 7, stride=2, padding=3, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth
    
    def estimate_depth(self, image):
        with torch.no_grad():
            depth = self.forward(image)
        return depth