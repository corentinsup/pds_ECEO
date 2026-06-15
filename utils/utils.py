import os
import math
import torch
import numpy as np
import random
import yaml
from typing import Tuple

POLARIZATIONS = {'none': 0, 'VV': 1, 'VH': 2}

def is_rank0():
    """
    Checks if the current process is rank 0 in a distributed setting.
    """
    return os.environ.get('RANK', '0') == '0'

def set_seed(seed : int):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_specs(sensors_specs_path : str, spectrum_specs_path : str):
    """
    Loads sensor and spectrum specifications from YAML files.
    sensors_specs_path: path to the YAML file containing sensor specifications
    spectrum_specs_path: path to the YAML file containing spectrum specifications
    Returns a tuple of (sensors_specs, spectrum_specs).
    """
    if not os.path.exists(sensors_specs_path):
        raise ValueError(f"Could not find sensors_specs_path {sensors_specs_path}")
    with open(sensors_specs_path) as f:
        sensors_specs = yaml.safe_load(f.read())
    if not os.path.exists(spectrum_specs_path):
        raise ValueError(f"Could not find spectrum_specs_path {spectrum_specs_path}")
    with open(spectrum_specs_path) as f:
        spectrum_specs = yaml.safe_load(f.read())
    return (sensors_specs, spectrum_specs)

def compute_wavelength_stats(sensor_specs: dict, spectrum_specs: dict):
    """
    Build per-projection wavelength/polarization tensors and the mean of
    log(central_wavelength) over the bands actually used by ``sensor_specs``.

    Pass the already-filtered ``sensor_specs`` (i.e. after restricting to the
    sensors selected in the config). Polarization is read from the ``name``
    field of each spectrum entry (e.g. 'VV', 'VH'); anything not matching a
    known polarization is treated as 'none'.

    Returns
    -------
    wavelengths_per_proj : torch.FloatTensor of shape (max_proj_idx + 1,)
        Central wavelength (nm) per projection_idx; unused slots are 0.
    polarization_per_proj : torch.LongTensor of shape (max_proj_idx + 1,)
        Polarization index per projection_idx (see ``POLARIZATIONS``).
    log_wl_mean : float
        Mean of log(central_wavelength) over the *used* bands only.
    """
    used_band_names = set()
    for sensor in sensor_specs.values():
        bands = sensor['bands']
        for i in sensor['selected_bands']:
            used_band_names.add(bands[int(i)])

    entries = []  # (proj_idx, central_wl, pol_idx)
    for band_name, spec in spectrum_specs.items():
        if band_name not in used_band_names:
            continue
        proj_idx = int(spec['projection_idx'])
        if proj_idx == -1:
            continue
        central_wl = 0.5 * (float(spec['min_wavelength']) + float(spec['max_wavelength']))
        pol_idx = POLARIZATIONS.get(spec.get('name', ''), POLARIZATIONS['none'])
        entries.append((proj_idx, central_wl, pol_idx))

    if not entries:
        raise ValueError("compute_wavelength_stats: no used bands found in spectrum_specs")

    max_proj = max(e[0] for e in entries)
    wavelengths_per_proj = torch.zeros(max_proj + 1, dtype=torch.float32)
    polarization_per_proj = torch.zeros(max_proj + 1, dtype=torch.long)
    for proj_idx, central_wl, pol_idx in entries:
        wavelengths_per_proj[proj_idx] = central_wl
        polarization_per_proj[proj_idx] = pol_idx

    log_wl_mean = float(np.mean([math.log(e[1]) for e in entries]))
    return wavelengths_per_proj, polarization_per_proj, log_wl_mean


def load_marker_embeddings(embedding_dir : str):
    """
    Loads all marker embeddings from the specified directory.
    embedding_dir: directory containing marker embeddings in .pt format
    Returns a tensor of shape (num_markers, embedding_dim)
    """
    if not os.path.exists(embedding_dir):
        raise ValueError(f"Could not find embedding_dir {embedding_dir}")
    files = os.listdir(embedding_dir)
    files = sorted(files)
    embeddings = []
    for file in files:
        if file.endswith(".pt"):
            embeddings.append(torch.load(os.path.join(embedding_dir, file), weights_only=True))
    embeddings = torch.stack(embeddings, dim=0)
    return embeddings

def to_device(x, device):
    """
    Moves tensor(s) to the specified device.
    x: tensor or list/tuple of tensors
    device: target device (e.g., 'cuda' or 'cpu')
    """
    if isinstance(x, list):
        return [to_device(i, device) for i in x]
    if isinstance(x, tuple):
        return [to_device(i, device) for i in x]
    if x is None:
        return None
    else:
        return x.to(device)

def get_selected_bands_mask(config: dict, sensor_names: list[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a boolean mask for selected bands based on sensor configuration.
    
    Parameters
    ----------
    config : dict
        Sensor configuration dictionary loaded from YAML
    sensor_names : list[str], optional
        List of sensor names to include. If None, all sensors in config are used.
    
    Returns
    -------
    torch.Tensor
        Boolean mask of shape (total_bands,) where True indicates a band is selected.
    torch.Tensor
        Tensor of shape (num_selected_bands,) containing the indices of selected channels.
    """
    if sensor_names is None:
        sensor_names = list(config.keys())
    
    total_mask = []
    band_offset = 0
    channel_indices = []  # List to store the indices of selected channels for each sensor
    
    for sensor_name in sensor_names:
        if sensor_name not in config:
            print(f"Warning: {sensor_name} not found in config")
            continue
            
        sensor_config = config[sensor_name]
        num_bands = len(sensor_config['bands'])
        selected_indices = sensor_config['selected_bands']
        
        # Create mask for this sensor and keep track of the indices for each sensor 
        sensor_mask = [False] * num_bands
        for idx in selected_indices:
            if 0 <= idx < num_bands:
                sensor_mask[idx] = True
                channel_indices.append(band_offset + idx)  # Store the global index of the selected channel
        
        total_mask.extend(sensor_mask)
        band_offset += num_bands

    return (torch.tensor(total_mask, dtype=torch.bool), torch.tensor(channel_indices, dtype=torch.long))

'''   
def set_specs(args):
    with open(args.eval_specs_path) as f:
        eval_ds = yaml.safe_load(f.read())
    args.eval_specs = eval_ds
    with open(args.sensors_specs_path) as f:
        sensors_specs = yaml.safe_load(f.read())
    args.sensors_specs = sensors_specs  # save on args so that it's prop'd to wandb
    with open(args.spectrum_specs_path) as f:
        spectrum_specs = yaml.safe_load(f.read())
    args.spectrum_specs = spectrum_specs
    '''

def build_urls(base_path, modalities, shard_names):
    """
    Build list of multi-modality URLs in the format multi_tarfile_samples expects:
    /base/[S2L2A,S1GRD,DEM]/majortom_shard_000001.tar
    """
    modality_str = ",".join(modalities)
    return [
        os.path.join(base_path, f"[{modality_str}]", shard)
        for shard in shard_names
    ]

