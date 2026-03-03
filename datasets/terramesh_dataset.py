import torch
import numpy as np
import pandas as pd
import os

from typing import List, Tuple
from torchvision.transforms import v2
from torch.utils.data import IterableDataset
#from datasets.virtues_augmentations import MultiplexRandomCrop, MultiplexRandomSymmetry, ChannelDropout
from datasets.terramesh import build_terramesh_dataset
from utils.utils import load_marker_embedding_dict
from utils.masking import generate_mask

class TerraMeshDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        modalities: list[str] | str,
        marker_embedding_dir: str,
        channels_file : str,
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
        crop_size: int = 256,
        patch_size: int = 16,
        masking_ratio : Tuple[float, float] = (0.6, 1.0),
        channel_fraction : Tuple[float, float] = (0.75, 1.0),
    ):
        """
        image_dir: directory containing full satelite images
        channels_file: path to csv file containing channel information in the same order as the multiplex images
        marker_embedding_dir: directory containing marker embedding files
        split: split to use (train, test, all)
        crop_size: size of the training crops
        patch_size: size of the patches for masking
        masking_ratio: tuple indicating the range of masking ratios from which per-sample masking ratio is drawn uniformly
        channel_fraction: tuple indicating the range of channel fractions from which per-sample fraction for channel dropout is drawn uniformly
        """
        self.channels = pd.read_csv(channels_file)
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio
        self.transform = transform
        self.modalities = modalities

        self.channel_mask = [] 
        self.marker_indices = [] 
        marker_embedding_dict = load_marker_embedding_dict(marker_embedding_dir)
        
        for idx, row in self.channels.iterrows():
            channel_id = row['band_id']
            if channel_id in marker_embedding_dict:
                    print(f"Found marker embedding for channel {channel_id} (index {marker_embedding_dict[channel_id]}), including in dataset.")
                    self.channel_mask.append(True)
                    self.marker_indices.append(marker_embedding_dict[channel_id])
            else:
                self.channel_mask.append(False)
            
        self.channel_mask = torch.tensor(self.channel_mask, dtype=torch.bool)
        self.marker_indices = torch.tensor(self.marker_indices, dtype=torch.long) # Marker embeddings obtained from the PLM
        
        print("channel mask shape:", self.channel_mask.shape)
        print("channel mask:", self.channel_mask)
        print("marker indices shape:", self.marker_indices.shape)
        print("marker indices:", self.marker_indices)
        # Build the WebDataset pipeline using the provided build_terramesh_dataset function
        self.dataset = build_terramesh_dataset(
            path=path,
            modalities=modalities,
            split=split,
            batch_size=batch_size,
            transform= transform,
            return_metadata=return_metadata,
            shuffle=shuffle,
            shardshuffle=shardshuffle,
            deterministic=deterministic,
            seed=seed,
            time_dim=time_dim,
            partial=partial,
            probs=probs,
        )

    def _process(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert one WebDataset output dict into (images, marker_indices, mask).

        Returns
        -------
        images : Tensor [C_total, H, W] or [B, C_total, H, W]
            All present modality bands concatenated along the channel axis.
        marker_indices : Tensor [C_total]
        mask : Tensor [C_total, Hg, Wg] or [B, C_total, Hg, Wg]
        """
    
        # Concanteante all modality bands along the channel axis
        image = torch.cat([sample[modality] for modality in self.modalities], dim=-3)  # [C_total, H, W] or [B, C_total, H, W]

        # apply channel mask to filter out channels without marker embeddings
        image = image[:, self.channel_mask]  # [C_present, H, W] or [B, C_present, H, W]

        marker_indices = self.marker_indices

        # Generate mask(s) at patch-grid resolution
        '''if image.dim() == 3:          # single sample: [C, H, W]
            C, H, W = image.shape
            mask = generate_mask(C, H // self.patch_size, W // self.patch_size, self.masking_ratio)
        else: '''                          
        # batched: [B, C, H, W]
        B, C, H, W = image.shape
        mask = torch.stack(
            [generate_mask(C, H // self.patch_size, W // self.patch_size, self.masking_ratio)
            for _ in range(B)]
            )
        return image, marker_indices, mask

    def __iter__(self):
        """Yield (images, marker_indices, mask) for every sample in the WebDataset pipeline."""
        for sample in self.dataset:
            yield self._process(sample)