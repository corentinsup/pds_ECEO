import os
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from datasets.terramesh import Transpose, MultimodalTransforms, MultimodalNormalize, statistics
from datasets.terramesh_dataset import TerraMeshDataset


def train_epoch(modalities): 
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

    dataset = TerraMeshDataset(
        path='../../data/corentin/data/TerraMesh',  # Streaming or local path
        modalities=modalities, 
        shuffle=True,  # Set false for split="val"
        split="val",
        transform=train_transform,
        batch_size=8,
        marker_embedding_dir=os.path.join(os.path.dirname(__file__), 'assets/marker_embeddings'),
        channels_file=os.path.join(os.path.dirname(__file__), 'assets/channels.csv')
        
    )

    # Set batch size to None because batching is handled by WebDataset.
    dataloader = DataLoader(dataset, batch_size=None, num_workers=4, persistent_workers=True, prefetch_factor=1)

    # Iterate over the dataloader
    counter = 0
    for images, marker_indices, mask in dataloader:
        # images:         [B, C_total, H, W]  – all modality bands concatenated
        # marker_indices:
        # mask:           [B, C_total, Hg, Wg] – patch-level boolean mask (Hg = H // patch_size)
        print("Images shape:        ", images.shape)
        print("Marker indices:      ", marker_indices)
        print("Mask shape:          ", mask.shape)
        counter+=1
        if counter >= 4:  # Just check the first batches
            break


if __name__ == "__main__":
    modalities=["S2L2A", "S2L1C", "S1RTC"]
    print("Training with modalities:", modalities)
    train_epoch(modalities)
