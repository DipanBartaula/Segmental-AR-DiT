import torch
import torch.nn as nn
import math
from einops import rearrange
from transformers import CLIPTextModel

def sinusoidal_embedding(timesteps, dim):
    """Creates sinusoidal embeddings for the time steps."""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    return torch.cat([emb.sin(), emb.cos()], dim=-1)

class DiTBlock(nn.Module):
    """A block of the Diffusion Transformer."""
    def __init__(self, dim, num_heads, cond_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = nn.MultiheadAttention(dim, num_heads, kdim=cond_dim, vdim=cond_dim, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, t_emb, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        x_res = x; x = self.norm1(x); x = x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x, _ = self.attn1(x, x, x); x = x * gate_msa.unsqueeze(1) + x_res
        x_res = x; x = self.norm2(x); x, _ = self.attn2(x, cond, cond); x = x_res + x
        x_res = x; x = self.norm3(x); x = x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = self.ffn(x); x = x * gate_mlp.unsqueeze(1) + x_res
        return x

class SegmentalAR_DiT(nn.Module):
    """The main Segmental Autoregressive Diffusion Transformer model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.model_dim
        
        self.patch_proj = nn.Conv2d(1, self.dim, 
                                    kernel_size=(config.patch_size_mel, config.patch_size_time), 
                                    stride=(config.patch_size_mel, config.patch_size_time))
        
        self.time_embed = nn.Sequential(nn.Linear(self.dim, self.dim * 4), nn.SiLU(), nn.Linear(self.dim * 4, self.dim))

        # --- Get conditioning dimension from CLIP model config ---
        clip_config = CLIPTextModel.from_pretrained(config.text_encoder_name).config
        cond_dim = clip_config.hidden_size
        
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(self.dim, config.num_heads, cond_dim=cond_dim) for _ in range(config.num_layers)
        ])

        self.out_proj = nn.Linear(self.dim, config.patch_size_mel * config.patch_size_time)
        
        self.initialize_weights()

    def initialize_weights(self):
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        self.apply(_init)

    def forward(self, noisy_segment, time, text_cond, prev_segments_cond=None):
        B = noisy_segment.shape[0]
        t_emb = self.time_embed(sinusoidal_embedding(time, self.dim))
        x_patches = self.patch_proj(noisy_segment).flatten(2).transpose(1, 2)
        num_target_patches = x_patches.shape[1]

        if prev_segments_cond is not None and prev_segments_cond.shape[-1] > 0:
            cond_patches = self.patch_proj(prev_segments_cond).flatten(2).transpose(1, 2)
            patches = torch.cat((cond_patches, x_patches), dim=1)
        else:
            patches = x_patches

        pos_embed = nn.Parameter(torch.randn(1, patches.shape[1], self.dim, device=patches.device))
        patches = patches + pos_embed

        for block in self.transformer_blocks:
            patches = block(patches, t_emb, text_cond)
            
        pred_patches = patches[:, -num_target_patches:, :]
        pred_flat = self.out_proj(pred_patches)
        pred_mel = rearrange(pred_flat, 'b (h w) (p1 p2) -> b 1 (h p1) (w p2)', 
                               h=1, p1=self.config.patch_size_mel, p2=self.config.patch_size_time)
        
        return pred_mel
