import os
import io
import tarfile
import zarr
import fsspec
import numpy as np
import torch
from typing import List, Tuple
from torch.utils.data import Dataset

from datasets.virtues_augmentations import MultiplexRandomCrop, MultiplexRandomSymmetry, ChannelDropout
from utils.masking import generate_mask
from dataset.terramesh import statistics


# Global band IDs: each band across all TerraMesh modalities gets a unique integer.
# This plays the same role as `marker_indices` in MultiplexDataset.
BAND_IDS = {
    "S2L1C": list(range(0,  13)),   # 13 bands (B1-B13)
    "S2L2A": list(range(13, 25)),   # 12 bands (B1-B8A, B9, B11, B12)
    "S2RGB": list(range(25, 28)),   # 3 bands  (R, G, B)
    "S1GRD": list(range(28, 30)),   # 2 bands  (VV, VH)
    "S1RTC": list(range(30, 32)),   # 2 bands  (VV, VH)
    "NDVI":  list(range(32, 33)),   # 1 band
    "DEM":   list(range(33, 34)),   # 1 band
}
NUM_BANDS = 34  # total number of unique band IDs


class TerraMeshDataset(Dataset):
    """
    Map-style Dataset for local TerraMesh tar shards.
    Compatible with torch.utils.data.ConcatDataset.

    Returns the same 3-tuple as MultiplexDataset:
        crop      : Tensor [C, crop_size, crop_size]  — stacked & normalised bands
        band_ids  : Tensor [C]  (long)                — global band ID per channel
                    (analogue of marker_indices in MultiplexDataset)
        mask      : Bool Tensor                        — MAE-style spatial mask

    Directory layout expected under `path`:
        {path}/{split}/{modality}/shard_XXXXXX.tar

    Each tar member is a zarr.zip file named like:
        {sample_key}.zarr.zip
    The same sample_key appears in every modality's tar at the same shard index.
    """

    def __init__(
        self,
        path: str,
        modalities: List[str],
        split: str = "val",
        crop_size: int = 224,
        patch_size: int = 16,
        masking_ratio: Tuple[float, float] = (0.6, 1.0),
        channel_fraction: Tuple[float, float] = (0.75, 1.0),
        normalize: bool = True,
    ):
        """
        Args:
            path:             Root directory of TerraMesh (contains {split}/{modality}/*.tar).
            modalities:       Modalities to load, e.g. ["S2L2A", "S1RTC"].
                              All listed modalities must share the same shard files.
                              Do NOT mix S1GRD and S1RTC as they come from different subsets.
            split:            "train" or "val".
            crop_size:        Spatial size of output crops (pixels).
            patch_size:       Patch size for masking grid (pixels).
            masking_ratio:    (min, max) range for the per-sample masking ratio.
            channel_fraction: (min, max) range for channel dropout fraction.
            normalize:        Apply per-modality z-score normalization using TerraMesh statistics.
        """
        self.path = path
        self.modalities = modalities
        self.split = split
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio
        self.normalize = normalize

        # Build (tar_basename, zarr_member_name) index from the first modality
        self._index: List[Tuple[str, str]] = self._build_index()

        self.random_crop = MultiplexRandomCrop(size=(crop_size, crop_size))
        self.random_symmetry = MultiplexRandomSymmetry()
        self.drop_channels = ChannelDropout(channel_fraction=channel_fraction)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> List[Tuple[str, str]]:
        """
        Scans tar files of the reference modality once and records every
        (tar_basename, member_name) pair.  The same tar/member pair is
        later used for every other modality.
        """
        ref_modality = self.modalities[0]
        mod_dir = os.path.join(self.path, self.split, ref_modality)

        if not os.path.isdir(mod_dir):
            raise FileNotFoundError(
                f"Directory not found: {mod_dir}\n"
                f"Check that `path`, `split`, and `modalities` are correct."
            )

        tar_files = sorted(f for f in os.listdir(mod_dir) if f.endswith(".tar"))
        if not tar_files:
            raise FileNotFoundError(f"No .tar files found in {mod_dir}")

        index = []
        for tar_name in tar_files:
            tar_path = os.path.join(mod_dir, tar_name)
            with tarfile.open(tar_path, "r") as tf:
                for member in tf.getmembers():
                    if member.isfile() and member.name.endswith(".zarr.zip"):
                        index.append((tar_name, member.name))
        return index

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        tar_name, member_name = self._index[idx]

        bands_list: List[np.ndarray] = []
        band_ids_list: List[int] = []

        for modality in self.modalities:
            tar_path = os.path.join(self.path, self.split, modality, tar_name)
            arr = self._load_bands(tar_path, member_name)   # (C, H, W) float32

            if self.normalize:
                mean = np.array(statistics["mean"][modality], dtype=np.float32)[:, None, None]
                std  = np.array(statistics["std"][modality],  dtype=np.float32)[:, None, None]
                arr  = (arr - mean) / (std + 1e-9)

            bands_list.append(arr)
            band_ids_list.extend(BAND_IDS[modality])

        # Stack all modalities along the channel axis: (C_total, H, W)
        stacked  = torch.from_numpy(np.concatenate(bands_list, axis=0)).float()
        band_ids = torch.tensor(band_ids_list, dtype=torch.long)

        # Augmentations (same as MultiplexDataset)
        stacked               = self.random_crop(stacked)
        stacked               = self.random_symmetry(stacked)
        stacked, band_ids     = self.drop_channels(stacked, band_ids)

        # MAE mask  [C, H_patches, W_patches]
        C   = stacked.shape[0]
        H_p = W_p = self.crop_size // self.patch_size
        mask = generate_mask(C, H_p, W_p, self.masking_ratio)

        return stacked, band_ids, mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_bands(tar_path: str, member_name: str) -> np.ndarray:
        """
        Extracts a zarr.zip member from a tar file and returns its
        'bands' array as a float32 numpy array of shape (C, H, W).
        """
        with tarfile.open(tar_path, "r") as tf:
            raw = tf.extractfile(member_name).read()

        mapper = fsspec.filesystem("zip", fo=io.BytesIO(raw), block_size=None).get_mapper("")
        arr = zarr.open_consolidated(mapper, mode="r")["bands"][...]

        # Drop the time dimension if present: (1, C, H, W) -> (C, H, W)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr.squeeze(0)

        return arr.astype(np.float32)
