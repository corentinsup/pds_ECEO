import torch
import pandas as pd
import numpy as np
import os

from typing import Tuple
from torch.utils.data import IterableDataset
#from datasets.virtues_augmentations import MultiplexRandomCrop, MultiplexRandomSymmetry, ChannelDropout
from .terramesh import build_terramesh_dataset
from utils.utils import get_selected_bands_mask
from utils.masking import generate_mask

class TerraMeshDataset(IterableDataset):
    def __init__(
        self,
        #path: str,
        urls: list[str],
        modalities: list[str] | str,
        sensor_specs: dict,
        spectrum_specs: dict,
        #split: str = "val",
        transform=None,
        batch_size: int = 8,
        return_metadata: bool = False,
        shuffle: bool = None,
        shardshuffle: int = 100,
        deterministic: bool = False,
        seed: int = None,
        time_dim: bool = False,
        partial: bool = None,
        probs: list[int] = None,
        patch_size: int = 16,
        masking_ratio : Tuple[float, float] = (0.6, 1.0)
    ):
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio
        self.transform = transform
        self.modalities = modalities
        
        # Init indices 
        self.projection_conversion = {i: spectrum_specs[i]['projection_idx'] for i in spectrum_specs}
        # create a dict of sensor_idx to the selected bands for that sensor
        self.bands = {sensor_specs[sensor]['sensor_idx']: np.array(
            sensor_specs[sensor]['bands'])[np.array(sensor_specs[sensor]['selected_bands']).astype('int')] for sensor in sensor_specs}
        self.max_channels = max(len(bands) for bands in self.bands.values())
        # create a dict of sensor_idx to the projection indices for the selected bands for that sensor
        self.projection_indices = {
            sensor_specs[sensor]['sensor_idx']: np.array(
                [self.projection_conversion[i] for i in self.bands[sensor_specs[sensor]['sensor_idx']]]) for sensor in sensor_specs}
        
        self.channel_proj_indices = self._build_channel_proj_indices(sensor_specs)
        
        # Create a mapping from the selected channels
        self.channel_mask, self.channels_indices = get_selected_bands_mask(sensor_specs)

        # Build per-channel sensor ids and filter with the same mask used for channel selection.
        sensor_ids_per_channel = []
        for sensor in sensor_specs:
            sensor_idx = sensor_specs[sensor]['sensor_idx']
            num_bands = len(sensor_specs[sensor]['bands'])
            sensor_ids_per_channel.extend([sensor_idx] * num_bands)
        sensor_ids_per_channel = torch.tensor(sensor_ids_per_channel, dtype=torch.long)
        self.sensor_indices = sensor_ids_per_channel[self.channel_mask]
       
        # Build the WebDataset pipeline using the provided build_terramesh_dataset function
        self.dataset = build_terramesh_dataset(
            #path=path,
            urls=urls,
            modalities=modalities,
            #split=split,
            batch_size=batch_size,
            transform=transform,
            return_metadata=return_metadata,
            shuffle=shuffle,
            shardshuffle=shardshuffle,
            deterministic=deterministic,
            seed=seed,
            time_dim=time_dim,
            partial=partial,
            probs=probs,
        )

    def _build_channel_proj_indices(self, sensor_specs):
        """
        Build a 1D tensor of projection_idx, one per present channel,
        in the same order as channels appear after channel_mask is applied.
        """
        proj_list = []
        for sensor in sensor_specs:
            sensor_idx = sensor_specs[sensor]['sensor_idx']
            # projection_indices[sensor_idx] already holds the proj_idx
            # for each selected band of this sensor, in order
            proj_list.append(self.projection_indices[sensor_idx])
        
        # Concatenate across sensors → (C_present,)
        return torch.tensor(np.concatenate(proj_list), dtype=torch.long)

    def _process(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert one WebDataset output dict into (images, marker_indices, mask).

        Returns
        -------
        images : [B, C_total, H, W]
            All present modality bands concatenated along the channel axis.
        channel_indices : Tensor [C_total]
        mask : Tensor [C_total, Hg, Wg] or [B, C_total, Hg, Wg]
        """
        '''# print sample min-max for debugging
        for m in self.modalities:
            if m in sample:
                # Convert to float for mean/std computation if needed
                sample_m = sample[m].float() if not sample[m].is_floating_point() else sample[m]
                print(f"Sample modality '{m}' stats: dtype: {sample[m].dtype}, min: {sample[m].min().item()}, max: {sample[m].max().item()}, mean: {sample_m.mean().item()}, std: {sample_m.std().item()}")
            else:
                print(f"Sample modality '{m}' is missing.")'''
        
        # Concatenate & mask channels
        images = torch.cat([sample[m] for m in self.modalities], dim=-3)  # (B, C_total, H, W)
        images = images[:, self.channel_mask]                               # (B, C_present, H, W)
        B, C, H, W = images.shape

        # Build proj_indices by broadcasting channel axis spatially
        # channel_proj_indices: (C_present,)
        # → (C_present, 1, 1) → (C_present, H//p, W//p) → (B, C_present, H//p, W//p)
        Hg = H // self.patch_size
        Wg = W // self.patch_size

        proj_indices = (
            self.channel_proj_indices          # (C_present,)
            .view(1, C, 1, 1)                  # (1, C, 1, 1)
            .expand(B, -1,Hg, Wg)              # (B, C_present, Hg, Wg)
            .clone()
            .to(images.device)
        )

        channel_indices = (
            self.channels_indices  # (C,)
            .unsqueeze(0)           # (1, C_present)
            .expand(B, -1)          # (B, C_present)
            .clone()
            .to(images.device)
        )
        sensor_indices = (
            self.sensor_indices # (C,)
            .unsqueeze(0)       # (1, C_present)
            .expand(B, -1)      # (B, C_present)
            .clone()
            .to(images.device)
        )
        
        # Generate mask at patch-grid resolution (batched: [B, C, H, W])
        masks = torch.stack(
            [generate_mask(C, H // self.patch_size, W // self.patch_size, self.masking_ratio)
            for _ in range(B)]
            )

        return images, channel_indices, masks, sensor_indices, proj_indices

    def __iter__(self):
        """Yield (images, channel_indices, mask) for every sample in the WebDataset pipeline."""
        for sample in self.dataset:
            yield self._process(sample)