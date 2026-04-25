import os
from pathlib import Path
import torch
import wandb
import glob
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments

from local_datasets.terramesh import Transpose, MultimodalTransforms, MultimodalNormalize, statistics
from local_datasets.terramesh_dataset import TerraMeshDataset
from utils.utils import is_rank0, set_seed, to_device, load_specs
from virtues.modules.multiplex_virtues import TerraMeshViT
from utils.utils import build_urls

class TerraMeshViTTrainer(Trainer):

    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = conf

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        multiplex, channel_ids, multiplex_mask, sensor_ids, projection_indices = inputs
        
        multiplex = to_device(multiplex, 'cuda') # list of tensors of shape C_i x H x W
        channel_ids = to_device(channel_ids, 'cuda') # list of tensors of shape C_i
        multiplex_mask = to_device(multiplex_mask, 'cuda') # list of tensors of shape C_i x H//patch_size x W//patch_size
        sensor_ids = to_device(sensor_ids, 'cuda') # list of tensors of shape C_i
        projection_indices = to_device(projection_indices, 'cuda') # list of tensors of shape max_channels x nb_patch_length x nb_patch_length

        outputs = model.forward(
                            multiplex=multiplex,
                            channel_ids=channel_ids,
                            multiplex_mask=multiplex_mask,
                            sensor_ids=sensor_ids,
                            projection_indices=projection_indices
                        )
        
        reconstructions = outputs.decoded_multiplex # list of tensors of shape C_i x H x W
        reconstructions = torch.concat(reconstructions, dim=0) # sum(C_i) x H x W
        targets = torch.concat(multiplex, dim=0) # sum(C_i) x H x W
        
        #loss = torch.mean(torch.pow(reconstructions - targets, 2))
        # compute loss only on masked patches
    
        mask = torch.concat(multiplex_mask, dim=0)  # (sum_C, H//p, W//p)
        p = self.conf.model.patch_size
        mask_full = mask.repeat_interleave(p, dim=-2).repeat_interleave(p, dim=-1)  # (sum_C, H, W)
        loss = torch.mean(torch.pow((reconstructions - targets)[mask_full], 2))
        
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        if DISABLE_EVAL:
            return {}
        else:
            eval_dl = self.get_eval_dataloader()
            sum = 0.0
            terms = 0
            for batch in eval_dl:
                with torch.no_grad():
                    loss = self.compute_loss(self.model, batch)
                    sum += loss.item()
                    terms += 1
            avg = sum / terms
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
            if world_size > 1:
                avg_tensor = torch.tensor(avg).cuda()
                torch.distributed.all_reduce(avg_tensor, op=torch.distributed.ReduceOp.SUM)
                avg = (avg_tensor / world_size).item()

            results = {
                "epoch" : self.state.epoch,
                "test_loss" : avg,
            }

            if WANDB_AVAILABLE and is_rank0():
                wandb.log(results)
            
            return results
        

def train_virtues(conf):
    """
    Trains VirTues according to the provided configuration.
    """
    sensor_specs, spectrum_specs = load_specs(conf.sensors_specs_path, conf.spectrum_specs_path)
    # Filter the sensor and spectrum specs to only include the sensors specified in the config
    sensor_specs = {k: v for k, v in sensor_specs.items() if v['sensor_idx'] in conf.data.sensors}
    
    # Build remap from old ids to contiguous [0..N-1]
    selected_old_ids = sorted({v["sensor_idx"] for v in sensor_specs.values()})
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(selected_old_ids)}

    # Apply remap in-place (or on a copied dict)
    for k in sensor_specs:
        sensor_specs[k]["sensor_idx"] = old_to_new[sensor_specs[k]["sensor_idx"]]
    num_sensors = len(conf.data.sensors) 

    # Define multimodal transform function that converts the data into the expected shape from albumentations
    transforms = MultimodalTransforms(
        transforms=A.Compose([
            Transpose([1, 2, 0]),  # Convert data to channel last (expected shape from albumentations)
            MultimodalNormalize(mean=statistics["mean"], std=statistics["std"]),
            A.RandomCrop(conf.data.crop_size, conf.data.crop_size),
            A.D4(),  # Random flipping and rotation
            ToTensorV2(),  # Convert to tensor and back to channel first
        ],
            is_check_shapes=False,  # Not needed because of aligned data in TerraMesh
            additional_targets={m: "image" for m in conf.data.modalities}  
        ),
        non_image_modalities=["__key__", "__url__"],  # Additional non-image keys
    )

    all_shard_names = sorted([
    os.path.basename(s) 
    for s in glob.glob(os.path.join(conf.dataset_path, conf.data.modalities[0], "*.tar"))
    if "majortom" in os.path.basename(s)  # ← only keep majortom shards
    ])
    
    print(f"DEBUG: Searching for tar files at: {os.path.join(conf.dataset_path, conf.data.modalities[0], '*.tar')}")
    print(f"DEBUG: Found {len(all_shard_names)} majortom shards")
    if all_shard_names:
        print(f"DEBUG: First 3 shards: {all_shard_names[:3]}")
    else:
        # Try to see what files actually exist
        all_files = glob.glob(os.path.join(conf.dataset_path, conf.data.modalities[0], "*.tar"))
        print(f"DEBUG: All tar files found: {all_files}")
        print(f"DEBUG: Dataset path: {conf.dataset_path}")
        print(f"DEBUG: First modality: {conf.data.modalities[0]}")
        raise ValueError(f"No majortom tar files found at {os.path.join(conf.dataset_path, conf.data.modalities[0])}")

    random.shuffle(all_shard_names)
    
    train_shard_names = all_shard_names[:6]
    test_shard_names = all_shard_names[6:]

    train_urls = build_urls(conf.dataset_path, conf.data.modalities, train_shard_names)
    print("Train URLs:")
    for url in train_urls:
        print(f"  {url}")
    val_urls = build_urls(conf.dataset_path, conf.data.modalities, test_shard_names)


    # Create train and test datasets
    train_dataset = TerraMeshDataset(
        #path=conf.dataset_path,
        urls=train_urls,
        modalities=conf.data.modalities,
        sensor_specs=sensor_specs,
        spectrum_specs=spectrum_specs,
        shuffle=conf.data.shuffle,  
        #split= "val",
        transform=transforms,
        seed=conf.experiment.seed,
        batch_size=conf.training.batch_size,
        patch_size=conf.model.patch_size,
        masking_ratio=tuple(conf.data.masking_ratio)
    )

    test_dataset = TerraMeshDataset(
        #path=conf.dataset_path,
        urls=val_urls,
        modalities=conf.data.modalities,
        sensor_specs=sensor_specs,
        spectrum_specs=spectrum_specs,
        shuffle=conf.data.shuffle,  
        #split= "val",
        transform=transforms,
        seed=conf.experiment.seed,
        batch_size=conf.training.batch_size,
        patch_size=conf.model.patch_size,
        masking_ratio=tuple(conf.data.masking_ratio)
    )

    def custom_collate_fn(batch):
        # if C is fixed, Webdataset returns a stacked tensor; otherwise, it returns a list.
        sample = batch[0]  # unpack the DataLoader wrapper
        multiplex, channel_ids, multiplex_mask, sensor_ids, projection_indices = sample

        # Handle both stacked tensor (fixed C) and list (variable C)
        if isinstance(multiplex, torch.Tensor):
            # Shape: (B, C, H, W)
            # Unbind along batch dim → List[Tensor(C, H, W)]
            multiplex          = list(multiplex.unbind(0))           # List[Tensor(C, H, W)]
            channel_ids        = list(channel_ids.unbind(0))         # List[Tensor(C,)]
            multiplex_mask     = list(multiplex_mask.unbind(0))      # List[Tensor(C, Hg, Wg)]
            sensor_ids         = list(sensor_ids.unbind(0))          # List[Tensor(C,)]
            projection_indices = list(projection_indices.unbind(0))  # List[Tensor(C, Hg, Wg)]
        else:
            # Already a list
            multiplex          = [mx.contiguous() for mx in multiplex]
            channel_ids        = [ch.contiguous() for ch in channel_ids]
            multiplex_mask     = [mx.contiguous() for mx in multiplex_mask]
            sensor_ids         = [s.contiguous() for s in sensor_ids]
            projection_indices = [p.contiguous() for p in projection_indices]

        # .contiguous() on the stacked case too
        multiplex          = [mx.contiguous() for mx in multiplex]
        multiplex_mask     = [mx.contiguous() for mx in multiplex_mask]

        assert all(mx.ndim == 3 for mx in multiplex), \
            f"Expected (C, H, W) tensors, got shapes: {[mx.shape for mx in multiplex]}"
        assert all(ch.ndim == 1 for ch in channel_ids), \
            f"Expected 1D channel_ids, got shapes: {[ch.shape for ch in channel_ids]}"
        assert len(multiplex) == len(channel_ids) == len(sensor_ids) == len(projection_indices), \
            "Mismatch in batch size across fields"

        return multiplex, channel_ids, multiplex_mask, sensor_ids, projection_indices

    model = TerraMeshViT(
        use_default_config = False,
        custom_config = None,
        spectrum_specs=spectrum_specs,
        num_sensors=num_sensors,
        prior_bias_embedding_type='wl',
        prior_bias_embedding_fusion_type='concatenate',
        patch_size=conf.model.patch_size,
        model_dim=conf.model.model_dim,
        feedforward_dim=conf.model.feedforward_dim,
        encoder_pattern=conf.model.encoder_pattern,
        num_encoder_heads=conf.model.num_encoder_heads,
        decoder_pattern=conf.model.decoder_pattern,
        num_decoder_heads=conf.model.num_decoder_heads,
        num_hidden_layers=conf.model.num_decoder_hidden_layers,
        positional_embedding_type=conf.model.positional_embedding_type,
        dropout=conf.model.dropout,
        group_layers=conf.model.group_layers,
        norm_after_encoder_decoder=conf.model.norm_after_encoder_decoder,
        verbose=False
    )
    model.cuda()

    training_args = TrainingArguments(
        output_dir=os.path.join(conf.experiments_dir, conf.experiment.name, 'checkpoints'),
        num_train_epochs=conf.training.epochs,
        max_steps=conf.training.max_steps if conf.training.max_steps > 0 else None,
        per_device_train_batch_size=1,  # We set batch size to 1 because we handle batching in the dataset with WebDataset
        per_device_eval_batch_size=1,
        eval_strategy="epoch" if not DISABLE_EVAL else "no",
        eval_steps=1, # epoch-wise evaluation
        save_strategy="epoch",  # Save the model at the end of each epoch
        save_total_limit=1,  # Only keep the last model
        logging_dir=f'{conf.experiments_dir}/{conf.experiment.name}/logs',
        logging_strategy="steps",
        logging_steps=100,
        fp16=conf.training.fp16,
        bf16=conf.training.bf16,
        dataloader_num_workers=conf.training.num_workers,
        report_to="wandb" if WANDB_AVAILABLE else "none",
        run_name=conf.experiment.name,
        gradient_accumulation_steps=conf.training.gradient_accumulation_steps,
        max_grad_norm=1.0,
        learning_rate=conf.training.lr,
        lr_scheduler_type=conf.training.lr_scheduler_type,
        weight_decay=conf.training.weight_decay,
        ddp_find_unused_parameters=True,
        warmup_ratio=(conf.training.warmup_epochs / conf.training.epochs) if conf.training.warmup_epochs > 0 else 0.0,
    )

    trainer = TerraMeshViTTrainer(
        conf=conf,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=custom_collate_fn,
    )

    trainer.train()


if __name__ == "__main__":
    conf = OmegaConf.load("pds_ECEO/virtues/configs/base_config.yaml")
    cli_conf = OmegaConf.from_cli()
    if hasattr(cli_conf, 'datasets_config') and cli_conf.datasets_config is not None:
        dataset_conf = OmegaConf.load(cli_conf.datasets_config)
        conf = OmegaConf.merge(conf, dataset_conf)
    elif hasattr(conf, 'datasets_config') and conf.datasets_config is not None:
        dataset_conf = OmegaConf.load(conf.datasets_config)
        conf = OmegaConf.merge(conf, dataset_conf)

    conf = OmegaConf.merge(conf, cli_conf)

    if is_rank0():
        print("OmegaConf merged config:")
        print(OmegaConf.to_yaml(conf))

    os.makedirs(os.path.join(conf.experiments_dir, conf.experiment.name), exist_ok=True)
    os.makedirs(os.path.join(conf.experiments_dir, conf.experiment.name, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(conf.experiments_dir, conf.experiment.name, 'logs'), exist_ok=True)

    OmegaConf.save(conf, os.path.join(conf.experiments_dir, conf.experiment.name, 'config.yaml'))

    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False
        
    if WANDB_AVAILABLE and is_rank0():
        wandb.init(
            entity=conf.experiment.wandb_entity,
            project=conf.experiment.wandb_project,
            name=conf.experiment.name,
            mode=conf.experiment.wandb_mode,
            dir=os.path.join(conf.experiments_dir, conf.experiment.name),
        )

    set_seed(conf.experiment.seed)

    if os.environ.get('LOCAL_RANK', None) is not None:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
    
    # Force torch to use specific GPU if specified in the config
    if conf.experiment.gpu is not None:
        CUDA_VISIBLE_DEVICES = str(conf.experiment.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        print(f"Using GPU {conf.experiment.gpu} for training.")

    DISABLE_EVAL = conf.training.disable_eval
    if DISABLE_EVAL and is_rank0():
        print("Evaluation is disabled.")

    train_virtues(conf)