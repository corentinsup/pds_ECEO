import numpy as np
import torch
import torch.nn as nn
import math

class SpectrumRangeProjection(nn.Module):
    """Patch Embedding of a sensor without patchify"""
    def __init__(
            self,
            spectral_range,
            spectrum_spec,
            patch_size,
            embed_dim,
            bias=True
    ):
        super().__init__()
        self.spectral_range = spectral_range
        self.name = spectrum_spec['name']
        self.min_wavelength = spectrum_spec['min_wavelength']
        self.max_wavelength = spectrum_spec['max_wavelength']
        self.sensors = spectrum_spec['sensors']
        self.nb_pixels = patch_size**2
        self.proj = nn.Linear(self.nb_pixels, embed_dim, bias=bias)
    
    def forward(self, x):
        return self.proj(x.view(-1, self.nb_pixels)) 

class SpectrumRangeProjectionAvg(nn.Module):
    """Patch Embedding of a sensor without patchify"""
    def __init__(
            self,
            spectrum_projections,
            spectrum_spec,
            embed_dim
    ):
        super().__init__()
        self.min_wavelength = spectrum_spec['min_wavelength']
        self.max_wavelength = spectrum_spec['max_wavelength']
        self.central_lambda = 0.5*(float(self.min_wavelength) + float(self.max_wavelength))
        self.spectrum_projections = spectrum_projections
        self.weights = []
        for spectrum_proj in self.spectrum_projections:
            central_lambda = 0.5*(float(spectrum_proj.min_wavelength) + float(spectrum_proj.max_wavelength))
            self.weights.append(abs(self.central_lambda-central_lambda))
        self.weights = np.array(self.weights) / sum(self.weights)
        self.embed_dim = embed_dim
    
    def forward(self, x):
        out = 0. #torch.zeros((len(x),self.embed_dim))
        for i, spectrum_proj in enumerate(self.spectrum_projections):
            out += spectrum_proj(x) * self.weights[i]
        return out
    

class SpectrumAwareProjection(nn.Module):
    """Patch Embedding of a sensor without patchify"""
    def __init__(
            self,
            spectrum_specs,
            patch_size,
            embed_dim,
            bias=True
    ):
        super().__init__()
        self.nb_pixels = patch_size**2
    
        self.spectrum_embeds = torch.nn.ModuleList()
        for spectral_range in sorted(spectrum_specs,key=lambda key:spectrum_specs[key]['projection_idx']):
            if ((spectrum_specs[spectral_range]['projection_idx'] != -1) and (len(spectrum_specs[spectral_range]['agg_projections']) == 0)) :
                self.spectrum_embeds.append(SpectrumRangeProjection(
                    spectral_range, spectrum_specs[spectral_range], patch_size, embed_dim
                ))

        for spectral_range in sorted(spectrum_specs,key=lambda key:spectrum_specs[key]['projection_idx']): 
            if ((spectrum_specs[spectral_range]['projection_idx'] != -1) and (len(spectrum_specs[spectral_range]['agg_projections']) > 0)):
                self.spectrum_embeds.append(
                    SpectrumRangeProjectionAvg(
                        [self.spectrum_embeds[agg_proj_idx] for agg_proj_idx in spectrum_specs[spectral_range]['agg_projections']], 
                        spectrum_specs[spectral_range],
                        embed_dim))
                
    def forward(self, x, projection_idx):
        if isinstance(projection_idx, torch.Tensor):
            projection_idx = int(projection_idx.item())
        return self.spectrum_embeds[int(projection_idx)](x)

# ----- UTIL TO COMPUTE WAWELENGTHS EMBEDDINGS -----

def wavelength_embedding(wavelengths, d_model):
    """
    Sinusoidal encoding of log-wavelength.
    
    Args:
        wavelengths: tensor of shape (N,) with wavelengths in logscale
        d_model: embedding dimension (must be even)
        base: frequency base (analogous to 10000 in standard positional encodings)
    
    Returns:
        Tensor of shape (N, d_model)
    """
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for wavelength embedding")
    
    # Frequency schedule
    half_d = d_model // 2
    '''min_freq = 2 * math.pi / 12.0   # ~0.52 — slow variation across full range
    max_freq = 2 * math.pi / 0.5   # ~62.8 — fast variation between adjacent bands
    
    # Geometric progression from min_freq to max_freq
    div_term = min_freq * (max_freq / min_freq) ** (
        torch.arange(half_d, device=wavelengths.device, dtype=torch.float32) / (half_d - 1)
    )'''
   
    div_term = torch.exp(torch.arange(0, half_d, device=wavelengths.device, dtype=torch.float32) * -(math.log(10000.0) / half_d))
    
    # Compute angles
    angles = wavelengths.unsqueeze(-1) * div_term 

    # Build embeddings
    pe = torch.zeros(wavelengths.shape[0], d_model, device=wavelengths.device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)

    return pe
