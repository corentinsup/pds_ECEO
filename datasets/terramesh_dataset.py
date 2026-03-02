import torch
import numpy as np
import pandas as pd
import os

from typing import List, Tuple
from torchvision.transforms import v2
from torch.utils.data import IterableDataset
#from datasets.virtues_augmentations import MultiplexRandomCrop, MultiplexRandomSymmetry, ChannelDropout
from datasets.terramesh import build_terramesh_dataset
from utils.masking import generate_mask


# Global band IDs: each band across all TerraMesh modalities gets a unique integer.
# This plays the same role as `marker_indices` in MultiplexDataset.
BAND_IDS = {
    "S2L1C": list(range(0,  13)),   # 13 bands (B1-B13)
    "S2L2A": list(range(13, 25)),   # 12 bands (B1-B8A, B9, B11, B12)
    "S1RTC": list(range(25, 27)),   # 2 bands  (VV, VH)
}
NUM_BANDS = 27  # total number of unique band IDs


class TerraMeshDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        modalities: list[str] | str,
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
        #image_dir: str,
        #channels_file : str,
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
        #self.image_dir = image_dir
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio
        self.transform = transform

        
        # Build marker_indices and channel_mask from BAND_IDS.
        # For satellite data there are no PLM protein embeddings; the integer band IDs
        # defined in BAND_IDS serve directly as marker indices (one per band channel).
        self.channel_mask = []
        self.marker_indices = []
        for mod, ids in BAND_IDS.items():
            if mod in modalities:
                self.channel_mask.extend([True]  * len(ids))
                self.marker_indices.extend(ids)
            else:
                self.channel_mask.extend([False] * len(ids))

        self.channel_mask   = torch.tensor(self.channel_mask,   dtype=torch.bool)
        self.marker_indices = torch.tensor(self.marker_indices, dtype=torch.long)

        print("channel mask shape:   ", self.channel_mask.shape)
        print("marker indices shape: ", self.marker_indices.shape)
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
    
        # Concatenate modality tensors along the channel axis (-3 works for both [C,H,W] and [B,C,H,W])
        images = torch.cat([torch.as_tensor(sample[mod]) for mod, _ in modalities], dim=-3)

        marker_indices = self.marker_indices

        # Generate mask(s) at patch-grid resolution
        if images.dim() == 3:          # single sample: [C, H, W]
            C, H, W = images.shape
            mask = generate_mask(C, H // self.patch_size, W // self.patch_size, self.masking_ratio)
        else:                           # batched: [B, C, H, W]
            B, C, H, W = images.shape
            mask = torch.stack(
                [generate_mask(C, H // self.patch_size, W // self.patch_size, self.masking_ratio)
                 for _ in range(B)]
            )

        return images, marker_indices, mask

    def __iter__(self):
        """Yield (images, marker_indices, mask) for every sample in the WebDataset pipeline."""
        for sample in self.dataset:
            yield self._process(sample)