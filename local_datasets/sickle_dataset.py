import torch
import cv2
import numpy as np
from typing import Tuple
from utils.masking import generate_mask
from utils.sickle_utils import scale_band, cloud_mask_from_qa, BAND_NAME_TO_NPZ
from utils.utils import get_selected_bands_mask

LANDSAT8_STATISTICS = {
    "mean": [0.13905, 0.15033, 0.17182, 0.18539, 0.31322, 0.24249, 0.18577],
    "std":  [0.21312, 0.21637, 0.20593, 0.21525, 0.17757, 0.14363, 0.13362],
}

class Landsat8Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            file_paths,
            sensor_specs,
            spectrum_specs,
            sensor_name: str = "LANDSAT8",
            apply_mask: bool = True,
            patch_size: int = 16,
            masking_ratio : Tuple[float, float] = (0.6, 1.0),
            transform=None,
        ):
        self.file_paths = file_paths
        self.apply_mask = apply_mask
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio
        self.transform = transform

        # Restrict to the single SICKLE/Landsat sensor so that channel_mask,
        # proj_indices and sensor_indices describe only this sensor's bands.
        if sensor_name not in sensor_specs:
            raise KeyError(f"sensor_name '{sensor_name}' not in sensor_specs; available: {list(sensor_specs)}")
        sensor_specs = {sensor_name: sensor_specs[sensor_name]}
        self.sensor_specs = sensor_specs
        self.sensor_key = sensor_name
        self.band_names = list(sensor_specs[sensor_name]['bands'])  # canonical band order from YAML

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
            
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])

        # Scale each selected band to physical units, in canonical YAML order.
        # Canonical names (aerosol, blue_2, ...) -> Landsat raster keys (SR_B1, ...).
        npz_keys = {b: BAND_NAME_TO_NPZ[b] for b in self.band_names}
        scaled_bands = {b: scale_band(npz_keys[b], data[npz_keys[b]]) for b in self.band_names}

        # Flag cloud / shadow / fill pixels as NaN on the physical bands.
        # QA_PIXEL is a bit-packed integer; cast in case it was stored as float.
        if self.apply_mask and "QA_PIXEL" in data.files:
            good = cloud_mask_from_qa(data["QA_PIXEL"].astype(np.uint16))
            for b, v in scaled_bands.items():
                if npz_keys[b].startswith("SR_B") or npz_keys[b] == "ST_B10":
                    v[~good] = np.nan

        # Stack in YAML order, keep only the selected bands -> numpy (C_present, H, W).
        # Kept as channel-first numpy because the transform's Transpose([1,2,0])
        # expects CHW input and converts to HWC for albumentations.
        img = np.stack([scaled_bands[b] for b in self.band_names], axis=0).astype(np.float32)
        img = img[self.channel_mask.numpy()]


        # Normalize and resize inline rather than via albumentations.
        mean = np.asarray(LANDSAT8_STATISTICS["mean"], dtype=np.float32).reshape(-1, 1, 1)
        std  = np.asarray(LANDSAT8_STATISTICS["std" ], dtype=np.float32).reshape(-1, 1, 1)

        # Per-channel NN resize -> (C, 224, 224). cv2.resize on a 2D array is safe.
        img = np.stack(
            [cv2.resize(img[c], (224, 224), interpolation=cv2.INTER_NEAREST) for c in range(img.shape[0])],
            axis=0,
        )
        img = (img - mean) / std                                       # per-channel
        img = torch.from_numpy(img).float()

        # Per-pixel validity captured BEFORE zero-filling: True where the pixel is
        # finite (i.e. not cloud / fill / cirrus / NaN). Used downstream to exclude
        # invalid pixels from reconstruction metrics, since the zero-fill below
        # turns those pixels into exact 0s with no ground-truth signal.
        valid_mask = torch.isfinite(img)                               # (C, H, W) bool

        # Masked / invalid pixels -> 0 (post-norm this is ~the per-band mean) so
        # NaNs don't propagate through the network.
        img = torch.nan_to_num(img, nan=0.0)

        # Indices are built from the *post-crop* spatial size.
        C, H, W = img.shape
        Hg, Wg = H // self.patch_size, W // self.patch_size

        # projection_idx per band, broadcast across the patch grid -> (C, Hg, Wg)
        proj_indices = (
            self.channel_proj_indices  # (C,)
            .view(C, 1, 1)
            .expand(C, Hg, Wg)
            .clone()
        )

        # full mask on B02
        mask = torch.zeros(C, Hg, Wg, dtype=torch.bool, device='cuda')
        mask[1] = True
        
        return img, self.channels_indices, mask, self.sensor_indices, proj_indices, valid_mask
