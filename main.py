import os
from pathlib import Path
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from datasets.terramesh import Transpose, MultimodalTransforms, MultimodalNormalize, statistics
from datasets.terramesh_dataset import TerraMeshDataset
from utils.utils import load_specs

def train_epoch(modalities): 
    project_root = Path(__file__).resolve().parent
    config_dir = project_root / "config"

    # Define multimodal transform function that converts the data into the expected shape from albumentations 
    train_transform = MultimodalTransforms(
        transforms=A.Compose([  # We use albumentations because of the shared transform between image modalities
            Transpose([1, 2, 0]),  # Convert data to channel last (expected shape from albumentations)
            MultimodalNormalize(mean=statistics["mean"], std=statistics["std"]),
            #A.CenterCrop(224, 224),  # Use center crop in val split
            A.RandomCrop(224, 224),  # Use random crop in train split
            A.D4(),  # Optionally, use random flipping and rotation for the train split
            ToTensorV2(),  # Convert to tensor and back to channel first
        ],
            is_check_shapes=False,  # Not needed because of aligned data in TerraMesh
            additional_targets={m: "image" for m in modalities}  
        ),
        non_image_modalities=["__key__", "__url__"],  # Additional non-image keys
    )
    sensor_specs, spectrum_specs = load_specs(
        str(config_dir / "pretraining_sensors.yaml"),
        str(config_dir / "electromagnetic_spectrum.yaml"),
    )
    sensors = [1, 2]  # List of sensor indices to use (as defined in sensors_specs_path)
    sensor_specs = {k: v for k, v in sensor_specs.items() if v['sensor_idx'] in sensors}
    
    dataset = TerraMeshDataset(
        path="../../data/corentin/data/TerraMesh",
        modalities=modalities,
        shuffle=True,  
        split= "val",
        transform=train_transform,
        batch_size=8,
        sensor_specs=sensor_specs,  # Load sensor specs as needed
        spectrum_specs=spectrum_specs,  # Load spectrum specs as needed
        patch_size=16,
        masking_ratio=(0.6, 1.0)
    )

    # Set batch size to None because batching is handled by WebDataset.
    dataloader = DataLoader(dataset, batch_size=None, num_workers=4, persistent_workers=True, prefetch_factor=1)

    # Iterate over the dataloader
    counter = 0
    for images, channel_indices, masks, sensor_indices, proj_indices in dataloader:
        # images:         [B, C_total, H, W]  – all modality bands concatenated
        # channel_indices: [C_total] – indices of the channel embeddings
        # mask:           [B, C_total, Hg, Wg] – patch-level boolean mask (Hg = H // patch_size)
        # proj_indices: [B, max_channels, nb_patch_length, nb_patch_length] – tiled projection indices
        print("Images shape:        ", images.shape)
        print("Channel indices:     ", channel_indices)
        print("Mask shape:          ", masks.shape)
        print("Projection indices shape: ", proj_indices.shape)
        counter+=1
        if counter >= 2:  # Just check the first batches
            break


if __name__ == "__main__":
    modalities=["S2L2A", "S1RTC"]
    print("Training with modalities:", modalities)
    train_epoch(modalities)
