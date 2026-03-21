import torch
import pandas as pd
import numpy as np
import os

from typing import Tuple
from torch.utils.data import IterableDataset
#from datasets.virtues_augmentations import MultiplexRandomCrop, MultiplexRandomSymmetry, ChannelDropout
from datasets.terramesh import build_terramesh_dataset
from utils.utils import get_selected_bands_mask
from utils.masking import generate_mask

class TerraMeshDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        modalities: list[str] | str,
        sensor_specs: dict,
        spectrum_specs: dict,
        split: str = "val",
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
        masking_ratio : Tuple[float, float] = (0.6, 1.0),
        crop_size: int = 224,
        max_channels: int = 13 # max numbers of channels for any sensor after band selection, used for tiling the projection indices
    ):
        """
        image_dir: directory containing full satelite images
        channels_file: path to csv file containing channel information in the same order as the multiplex images
        split: split to use (train, test, all)
        masking_ratio: tuple indicating the range of masking ratios from which per-sample masking ratio is drawn uniformly
        """
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio
        self.transform = transform
        self.modalities = modalities
        self.nb_patch_length = int(crop_size // self.patch_size)  # Assuming all images are 512x512, this gives the number of patches along one dimension
        self.max_channels = max_channels

        # Init indices 
        '''self.projection_conversion = {i: spectrum_specs[i]['projection_idx'] for i in spectrum_specs}
        self.bands = np.array(sensor_specs['bands'])
        self.selected_band_indices =  np.array(sensor_specs['selected_bands']).astype('int')
        self.projection_indices = np.array([self.projection_conversion[i] for i in self.bands[self.selected_band_indices]])
'''
        self.projection_conversion = {i: spectrum_specs[i]['projection_idx'] for i in spectrum_specs}
        # create a dict of sensor_idx to the selected bands for that sensor
        self.bands = {sensor_specs[sensor]['sensor_idx']: np.array(
            sensor_specs[sensor]['bands'])[np.array(sensor_specs[sensor]['selected_bands']).astype('int')] for sensor in sensor_specs}
        # create a dict of sensor_idx to the projection indices for the selected bands for that sensor
        self.projection_indices = {
            sensor_specs[sensor]['sensor_idx']: np.array(
                [self.projection_conversion[i] for i in self.bands[sensor_specs[sensor]['sensor_idx']]]) for sensor in sensor_specs}
    

        # Create a mapping from the selected channels
        self.channel_mask, self.channels_indices = get_selected_bands_mask(sensor_specs)
        
        print("channel mask shape:", self.channel_mask.shape)
        print("channel mask:", self.channel_mask)
        print("channels indices shape:", self.channels_indices.shape)
        print("channels indices:", self.channels_indices)
        # Build the WebDataset pipeline using the provided build_terramesh_dataset function
        self.dataset = build_terramesh_dataset(
            path=path,
            modalities=modalities,
            split=split,
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

    def _process(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert one WebDataset output dict into (images, marker_indices, mask).

        Returns
        -------
        images : [B, C_total, H, W]
            All present modality bands concatenated along the channel axis.
        channel_indices : Tensor [C_total]
        mask : Tensor [C_total, Hg, Wg] or [B, C_total, Hg, Wg]
        """

        # Concanteante all modality bands along the channel axis
        images = torch.cat([sample[modality] for modality in self.modalities], dim=-3)  #[B, C_total, H, W]

        # apply channel mask to filter out channels without marker embeddings
        images = images[:, self.channel_mask]  # [B, C_present, H, W]

        channel_indices = self.channels_indices

        # Tile the projection indices to match the number of patches and the number of bands for each sensor 
        tiled_proj_indices = []

        for sensor_idx in self.projection_indices:
            proj = self.projection_indices[sensor_idx]                  # shape [nb_band]
            nb_band = len(proj)

            reps = int(np.ceil(self.max_channels / nb_band))                 # enough repeats
            tiled = np.tile(
                proj.reshape(1, 1, -1),
                (self.nb_patch_length, self.nb_patch_length, reps)
            ).astype(np.int32)

            tiled = tiled[:, :, :self.max_channels]                          # cap to max_channels
            tiled_proj_indices.append(tiled)

        tiled_proj_indices = np.concatenate(tiled_proj_indices, axis=-1) # shape [nb_patch_length, nb_patch_length, max_channels * nb_sensors]
        tiled_proj_indices = torch.as_tensor(np.expand_dims(tiled_proj_indices, axis=0)) # shape [1, nb_patch_length, nb_patch_length, max_channels]
        tiled_proj_indices = tiled_proj_indices.repeat(images.shape[0], 1, 1, 1).permute(0, 3, 1, 2) # shape [B, max_channels, nb_patch_length, nb_patch_length]

        # Generate mask(s) at patch-grid resolution (batched: [B, C, H, W])
        B, C, H, W = images.shape
        mask = torch.stack(
            [generate_mask(C, H // self.patch_size, W // self.patch_size, self.masking_ratio)
            for _ in range(B)]
            )
        
        return images, channel_indices, mask, tiled_proj_indices

    def __iter__(self):
        """Yield (images, channel_indices, mask) for every sample in the WebDataset pipeline."""
        for sample in self.dataset:
            yield self._process(sample)