import os
import torch
import numpy as np
import random
import yaml
from typing import Tuple

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