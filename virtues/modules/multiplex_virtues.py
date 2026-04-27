import torch
from torch import nn
from loguru import logger
#from .layers.transformers_flashattention import MarkerAttentionEncoderBlock, ChannelAttentionEncoderBlock, FullAttentionEncoderBlock, PatchAttentionBlock
from .layers.transformers_sdpa import MarkerAttentionEncoderBlock, ChannelAttentionEncoderBlock, FullAttentionEncoderBlock, PatchAttentionBlock
from .layers.mask_utils_flashattention import build_self_attention_bias, build_self_attention_bias_channel_concat, SELF_ATTENTION_BIAS_CACHE
from .layers.positional_embeddings import LearnablePositionalEmbedding2D, PositionalEmbedding2D, RotaryPositionalEmbedding2D
from utils.model_utils import SpectrumAwareProjection
from einops import rearrange
from itertools import groupby 
from typing import Iterator, Optional, List, Tuple, Dict, Any, overload
from dataclasses import dataclass
from .layers.basic_modules import build_activation

@dataclass
class TerraMeshViTEncoderOutput:
    encoded_multiplex: List[torch.Tensor]  # List of encoded multiplex tensors, one per input multiplex
    patch_summary_tokens: List[torch.Tensor]  # List of patch summary tokens, one per input multiplex

@dataclass 
class TerraMeshViTDecoderOutput:
    decoded_multiplex: Tuple[torch.Tensor]  # List of decoded multiplex tensors, one per input multiplex

@dataclass
class TerraMeshViTOutput:
    decoded_multiplex: Tuple[torch.Tensor]  # List of decoded multiplex tensors, one per input multiplex
    patch_summary: List[torch.Tensor]  # List of patch summary tokens, one per input multiplex
    channel_token_embeddings: Optional[List[torch.Tensor]] = None  # Optional list of channel token embeddings

class TerraMeshViTEncoder(nn.Module):
    def __init__(
            self,
            spectrum_specs: Dict[str, Any],
            prior_bias_embeddings: torch.Tensor,
            prior_bias_embedding_type: Optional[str],
            patch_size: int,
            num_sensors: int,
            model_dim: int,
            feedforward_dim: int,
            encoder_pattern: str,
            num_encoder_heads: int,
            dropout: float,
            group_layers: float,
            norm_after_encoder_decoder: bool,
            positional_embedding_type: str,
            verbose: bool = True,
            prior_bias_embedding_fusion_type: str = 'add',
            **kwargs: Any
    ):
        super().__init__()
        # set all
        self.prior_bias_embedding_type = prior_bias_embedding_type
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.feedforward_dim = feedforward_dim
        self.encoder_pattern = encoder_pattern
        self.num_encoder_heads = num_encoder_heads
        self.dropout = dropout
        self.group_layers = group_layers
        self.norm_after_encoder_decoder = norm_after_encoder_decoder
        self.positional_embedding_type = positional_embedding_type
        self.verbose = verbose
        self.prior_bias_embedding_fusion_type = prior_bias_embedding_fusion_type

        self.use_prior_embedding = self.prior_bias_embedding_type != 'empty'
        if self.use_prior_embedding:
            #assert prior_bias_embeddings is not None, "prior_bias_embeddings must be provided if prior_bias_embedding_type is not 'empty'"
            if self.prior_bias_embedding_type == 'learnable':
                self.prior_bias_embeddings = nn.Parameter(self.prior_bias_embeddings)
            elif self.prior_bias_embedding_type == 'zero':
                self.prior_bias_embeddings = nn.Parameter(torch.zeros_like(self.prior_bias_embeddings))
            elif self.prior_bias_embedding_type == 'wl':
                self.spectrum_projection = SpectrumAwareProjection(
                    spectrum_specs=spectrum_specs, 
                    patch_size=self.patch_size, 
                    embed_dim=self.model_dim
                )
                # Creation of learnable embeddings for each sensor
                self.num_sensors = num_sensors
                self.sensor_embeddings = nn.Embedding(self.num_sensors, self.model_dim)
            else:
                self.register_buffer('prior_bias_embeddings', prior_bias_embeddings, persistent=False)
        

        if self.prior_bias_embedding_fusion_type == 'add':
            if self.verbose:
                print("Using addition for prior_embedding fusion")
        elif self.prior_bias_embedding_fusion_type == 'concatenate':
            if self.verbose:
                print("Using concatenation followed by linear layer for prior_embedding fusion")    
        elif self.prior_bias_embedding_fusion_type == 'cross-attn':
            raise NotImplementedError("Cross-attention fusion not implemented yet") # @EJ
        
        # model specific
        power = kwargs.get("parameter_init_power", 0.5)
        self.patch_summary_token = nn.Parameter(torch.randn(self.model_dim) / self.model_dim**power) # type: ignore
        self.num_registers = kwargs.get('num_registers', 0)
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(self.num_registers, self.model_dim) / self.model_dim**power) # type: ignore
            if self.verbose:
                print(f"Using {self.num_registers} register tokens")
        
        if self.positional_embedding_type == 'learnable':
            self.positional_embedding = LearnablePositionalEmbedding2D(self.model_dim, max_pos = kwargs.get('max_pos', 100)) # type: ignore
        elif self.positional_embedding_type == 'absolute_beginning':
            self.positional_embedding = PositionalEmbedding2D(self.model_dim, max_width_or_height = kwargs.get('max_width_or_height', 100)) # type: ignore
        else:
            self.positional_embedding = None

        self.masked_token = nn.Parameter(torch.randn(self.model_dim) / self.model_dim**power) # type: ignore

        self.patch_encoder = nn.Linear(self.patch_size * self.patch_size, self.model_dim)
        self.prior_embedding_encoder = nn.Linear(self.prior_bias_embeddings.shape[1], self.model_dim) if self.prior_bias_embedding_type != 'wl' else None 
        self.prior_embedding_linear = nn.Linear(self.model_dim * 2, self.model_dim) if self.use_prior_embedding and self.prior_bias_embedding_type == 'wl' else None

        # forming encoder
        enc_layers = []
        if group_layers:
            grouped_encoder_pattern = [(label, sum(1 for _ in group)) for label, group in groupby(self.encoder_pattern)]
        else:
            grouped_encoder_pattern = [(label, 1) for label in self.encoder_pattern]
        for block_type, count in grouped_encoder_pattern:
            if block_type == 'v':
                enc_layers.append(
                    MarkerAttentionEncoderBlock(model_dim=self.model_dim,
                                                feedforward_dim=self.feedforward_dim,
                                                num_heads=self.num_encoder_heads,
                                                dropout=dropout,
                                                inbuilt_pos_emb=None,
                                                num_layers=count)
                )
            elif block_type == 'h':
                enc_layers.append(
                    ChannelAttentionEncoderBlock(model_dim=self.model_dim,
                                                feedforward_dim=self.feedforward_dim,
                                                num_heads=self.num_encoder_heads,
                                                dropout=dropout,
                                                inbuilt_pos_emb=self.positional_embedding_type,
                                                num_layers=count)
                )
        
            elif block_type == 'f':
                enc_layers.append(
                    FullAttentionEncoderBlock(model_dim=self.model_dim,
                                                feedforward_dim=self.feedforward_dim,
                                                num_heads=self.num_encoder_heads,
                                                dropout=dropout,
                                                inbuilt_pos_emb=self.positional_embedding_type,
                                                num_layers=count)
                )

            elif block_type == 'p':
                enc_layers.append(
                    PatchAttentionBlock(model_dim=self.model_dim,
                                                feedforward_dim=self.feedforward_dim,
                                                num_heads=self.num_encoder_heads,
                                                dropout=dropout,
                                                inbuilt_pos_emb=self.positional_embedding_type,
                                                num_layers=count,
                    )
                )

        if self.norm_after_encoder_decoder:
            self.layer_norm = nn.LayerNorm(self.model_dim)

        self.encoder = nn.ModuleList(enc_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward_list(self, multiplex: List[torch.Tensor],
                     channel_ids: List[torch.Tensor],
                     multiplex_mask: Optional[List[torch.Tensor]] = None,
                     sensor_ids: Optional[List[torch.Tensor]] = None,
                     projection_indices: Optional[torch.Tensor | List[torch.Tensor]] = None,
    ) -> TerraMeshViTEncoderOutput:
        
        B = len(multiplex) # number of samples in the batch, i.e. number of multiplexes
        h, w = multiplex[0].shape[-2], multiplex[0].shape[-1]
        H, W = h // self.patch_size, w // self.patch_size # Number of patches along height and width
        
        multiplex_channels_per_sample = [len(ch) for ch in channel_ids] # should be the sum of all channels across modalities for each sample, i.e. C_i for each sample i
        multiplex = [(
            rearrange(mx_i, 'C (H p) (W q) -> C H W (p q)', p=self.patch_size, q=self.patch_size)
        ) for mx_i in multiplex] # List[(C_i) H W (p q)]
        cat_multiplex = torch.cat(multiplex, dim=0)  # (sum_C) H W (p q)
        cat_channel_ids = torch.cat(channel_ids, dim=0)  # (sum_C)
        sensor_embeds = None
        
        # Sensor Embeddings 
        if sensor_ids is not None and self.use_prior_embedding:
            cat_sensor_ids = torch.cat(sensor_ids, dim=0).long() # (sum_C)
            # print("cat_sensor_ids:", cat_sensor_ids)
            if torch.any(cat_sensor_ids < 0) or torch.any(cat_sensor_ids >= self.num_sensors):
                bad_min = int(cat_sensor_ids.min().item())
                bad_max = int(cat_sensor_ids.max().item())
                raise ValueError(
                    f"sensor_ids out of range for embedding: min={bad_min}, max={bad_max}, num_sensors={self.num_sensors}"
                )
            sensor_embeds = self.sensor_embeddings(cat_sensor_ids) # (sum_C) D
            sensor_embeds = sensor_embeds.unsqueeze(1).expand(-1, H*W, -1).clone()  # (sum_C) (H W) D

        # Wavelength-based Spectrum-aware Embeddings
        if projection_indices is not None:
            cat_proj_indices = torch.cat(projection_indices, dim=0)  # (sum_C) H W

            # proj_idx is spatially constant per channel → take the value at the first spatial position for each channel
            flat_proj_indices = cat_proj_indices[:, 0, 0].long()  # (sum_C_per_sample)
            
            # flat_proj_indices: (sum_C,) — broadcast spatially
            spatial_proj = flat_proj_indices[:, None, None].expand(-1, H, W).clone()  # (sum_C, H, W)

            # Now apply spectrum projection patch by patch
            # cat_multiplex: (sum_C, H, W, p*q)
            # we need output: (sum_C, H, W, D)
            patch_embeds = None
            
            for proj_idx in torch.unique(spatial_proj):
                mask = (spatial_proj == proj_idx)                    # (sum_C, H, W) bool
                patches = cat_multiplex[mask]                        # (N_match, P²)
                proj_out = self.spectrum_projection(patches, proj_idx.item())
                if patch_embeds is None:
                    patch_embeds = torch.zeros(
                        (cat_multiplex.shape[0], H, W, self.model_dim),
                        device=cat_multiplex.device,
                        dtype=proj_out.dtype,
                    )
                patch_embeds[mask] = proj_out.to(dtype=patch_embeds.dtype)

            cat_multiplex = patch_embeds  # (sum_C, H, W, D)
        else:
            # Fallback: single generic linear projection
            cat_multiplex = self.patch_encoder(cat_multiplex)  # (sum_C, H, W, D)

        if multiplex_mask is not None:
            cat_multiplex_mask = torch.cat(multiplex_mask, dim=0)  # (sum_C) H W

        num_multiplex_channels = cat_multiplex.shape[0]
        sum_C = num_multiplex_channels + (1 + self.num_registers) * B

        pos_multiplex = torch.stack(
            torch.meshgrid(
                torch.arange(H, device=cat_multiplex.device),
                torch.arange(W, device=cat_multiplex.device),
                indexing='ij'
            ),
            dim=-1  
        )
        # pos_multiplex = pos_multiplex.expand(cat_multiplex.shape[0], H, W, 2)

        pos_multiplex = pos_multiplex.expand(sum_C, H, W, 2)
        pos_multiplex = rearrange(pos_multiplex, 'C H W d -> C (H W) d')

        if multiplex_mask is not None:
           cat_multiplex = torch.where(cat_multiplex_mask.unsqueeze(-1),
                                       self.masked_token.expand(cat_multiplex.shape),
                                        cat_multiplex)
           cat_multiplex_mask = rearrange(cat_multiplex_mask, 'C H W -> C (H W)')

        cat_multiplex = rearrange(cat_multiplex, "C H W D -> C (H W) D")

        # Concatenate sensor embeddings with cat_multiplex if sensor embeddings are used
        '''if sensor_embeds is not None:
            if self.prior_bias_embedding_fusion_type == 'concatenate':
                cat_multiplex = torch.cat([cat_multiplex, sensor_embeds], dim=-1)  # (sum_C) (H W) 2*D
                cat_multiplex = self.prior_embedding_linear(cat_multiplex)  # (sum_C) (H W) D
            else:
                cat_multiplex = cat_multiplex + sensor_embeds'''

        if self.prior_bias_embedding_type != 'wl':
            prior_embeddings = self.prior_bias_embeddings[cat_channel_ids]  # (sum_C) D
            prior_embeddings = self.prior_embedding_encoder(prior_embeddings)  # (sum_C) model_dim
            prior_embeddings = prior_embeddings.unsqueeze(1).expand(*cat_multiplex.shape)  # (sum_C) (H W) model_dim
            if self.prior_bias_embedding_fusion_type == 'add':
                cat_multiplex = cat_multiplex + prior_embeddings
        
        cat_multiplex = torch.split(cat_multiplex, multiplex_channels_per_sample, dim=0)  # List[(C_i) (H W) D]
        if self.num_registers > 0:
            x = [
                torch.cat(
                    [
                        self.patch_summary_token.expand(1, H*W, self.patch_summary_token.shape[-1]),  # (1) (H W) D
                        self.register_tokens.unsqueeze(1).expand(self.num_registers, H*W, self.register_tokens.shape[-1]),  # (num_registers) (H W) D
                        mx_i,
                    ], dim=0)
                    for mx_i in cat_multiplex
            ]
        else:
            x = [
                torch.cat(
                    [
                        self.patch_summary_token.expand(1, H*W, self.patch_summary_token.shape[-1]),  # (1) (H W) D
                        mx_i,
                    ], dim=0)
                    for mx_i in cat_multiplex
            ]
        x = torch.cat(x, dim=0)  # (sum_C + num_registers + 1) (H W) D
        x_channels_per_sample = [mc + 1 + (self.num_registers if self.num_registers > 0 else 0) for mc in multiplex_channels_per_sample]
        if multiplex_mask is not None:
            mask = [
                torch.concat(
                    [torch.zeros(1+self.num_registers, H*W, dtype=torch.bool, device=cat_multiplex_mask.device),  # (1+num_registers) (H W  )]
                    mx_i_mask],
                    dim=0
                )
                for mx_i_mask in torch.split(cat_multiplex_mask, multiplex_channels_per_sample, dim=0)  # List[(C_i) (H W)]
            ]
            mask = torch.cat(mask, dim=0)  # (sum_C + num_registers + 1) (H W)

        if self.positional_embedding is not None:
            x = self.positional_embedding(x, pos_multiplex)  # (sum_C + num_registers + 1) (H W) D

        
        for layer in self.encoder:
            if multiplex_mask is None:
                layer_output = layer.forward_cc(x,
                                                pos_multiplex,
                                                x_channels_per_sample) # type: ignore
                x = layer_output
            else:
                layer_output = layer.forward_cc_masked(
                    x,
                    pos_multiplex,
                    mask,
                    x_channels_per_sample,
                ) # pyright: ignore[reportCallIssue]
                x = layer_output

        if self.norm_after_encoder_decoder:
            x = self.layer_norm(x)

        SELF_ATTENTION_BIAS_CACHE.clear()
        x = rearrange(x, "C (H W) D -> C H W D", H=H, W=W)
        x = torch.split(x, x_channels_per_sample, dim=0)  # List[(C_i + num_registers + 1) H W D]
        ps = [x_i[0] for x_i in x]  # List[(H W) D]
        x = [x_i[1+self.num_registers:] for x_i in x]  # List[(C_i) H W D]

        return TerraMeshViTEncoderOutput(
            encoded_multiplex=x,
            patch_summary_tokens=ps,
        )
    

class TerraMeshViTDecoder(nn.Module):
    def __init__(self,
                 patch_size: int,
                 model_dim: int,
                 feedforward_dim: int,
                 decoder_pattern: str,
                 num_decoder_heads: int,
                 dropout: float,
                 group_layers: float,
                 norm_after_encoder_decoder: bool,
                 positional_embedding_type: str,
                 num_hidden_layers_head: int,
                 **kwargs: Any
                 ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.model_dim = model_dim
        self.feedforward_dim = feedforward_dim
        self.decoder_pattern = decoder_pattern
        self.num_decoder_heads = num_decoder_heads
        self.dropout = dropout
        self.group_layers = group_layers
        self.norm_after_encoder_decoder = norm_after_encoder_decoder
        self.positional_embedding_type = positional_embedding_type
        self.num_hidden_layers_head = num_hidden_layers_head

        mlp_decoder_layers = []
        if num_hidden_layers_head > 0:
            for _ in range(num_hidden_layers_head - 1):
                mlp_decoder_layers.append(nn.Linear(self.model_dim, self.model_dim))
                mlp_decoder_layers.append(build_activation('gelu'))
        mlp_decoder_layers.append(nn.Linear(self.model_dim, self.patch_size * self.patch_size))
        self.multiplex_decoder_mlp = nn.Sequential(*mlp_decoder_layers)

        if self.group_layers:
            grouped_decoder_pattern = [(label, sum(1 for _ in group)) for label, group in groupby(self.decoder_pattern)]
        else:
            grouped_decoder_pattern = [(label, 1) for label in self.decoder_pattern]

        dec_layers = []
        for block_type, count in grouped_decoder_pattern:
            if block_type == 'v':
                dec_layers.append(
                    MarkerAttentionEncoderBlock(model_dim=self.model_dim,
                                                feedforward_dim=self.feedforward_dim,
                                                num_heads=self.num_decoder_heads,
                                                dropout=dropout,
                                                inbuilt_pos_emb=None, # @EJ Check -> different from flex-dual-virtues
                                                num_layers=count)
                )
            elif block_type == 'h':
                dec_layers.append(
                    ChannelAttentionEncoderBlock(model_dim=self.model_dim,
                                                feedforward_dim=self.feedforward_dim,
                                                num_heads=self.num_decoder_heads,
                                                dropout=dropout,
                                                inbuilt_pos_emb=self.positional_embedding_type,
                                                num_layers=count)
                )
        
            elif block_type == 'f':
                dec_layers.append(
                    FullAttentionEncoderBlock(model_dim=self.model_dim,
                                                feedforward_dim=self.feedforward_dim,
                                                num_heads=self.num_decoder_heads,
                                                dropout=dropout,
                                                inbuilt_pos_emb=self.positional_embedding_type,
                                                num_layers=count)
                )

            elif block_type == 'p':
                dec_layers.append(
                    PatchAttentionBlock(model_dim=self.model_dim,
                                                feedforward_dim=self.feedforward_dim,
                                                num_heads=self.num_decoder_heads,
                                                dropout=dropout,
                                                inbuilt_pos_emb=self.positional_embedding_type,
                                                num_layers=count,
                    )
                )

        self.decoder = nn.ModuleList(dec_layers)
        if self.norm_after_encoder_decoder:
            self.layer_norm = nn.LayerNorm(self.model_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward_list(self, x: List[torch.Tensor], 
                     patch_summary_tokens: List[torch.Tensor],
                     x_channels_per_sample: List[int],
                     ) -> TerraMeshViTDecoderOutput:
        H, W, D = x[0].shape[-3], x[0].shape[-2], x[0].shape[-1]
        x_copy = torch.empty(
            sum(x_channels_per_sample), 2, H , W, D,
            dtype=x[0].dtype,
            device=x[0].device
        )
        curr_idx = 0
        for (x_i, ps_i) in zip(x, patch_summary_tokens):
            num_channels = x_i.shape[0]
            x_copy[curr_idx: curr_idx + num_channels, 0] = ps_i.expand(num_channels, H, W, D)
            x_copy[curr_idx: curr_idx + num_channels, 1] = x_i
            curr_idx += num_channels
        x_copy = torch.split(x_copy, x_channels_per_sample, dim=0)  # List[(C_i) 2 H W D]
        multiplex = torch.cat(
            [x_i[:c] for (x_i, c) in zip(x_copy, x_channels_per_sample)], dim=0
        )
        multiplex = rearrange(multiplex, "C E H W D -> (C E) (H W) D")
        multiplex_examples = multiplex.shape[0] // 2


        multiplex_pos = torch.stack(
            torch.meshgrid(
                torch.arange(H, device=multiplex.device),
                torch.arange(W, device=multiplex.device),
                indexing='ij'
            ),
            dim=-1
        )
        multiplex_pos = multiplex_pos.expand(multiplex.shape[0], H, W, 2)
        multiplex_pos = rearrange(multiplex_pos, 'C H W d -> C (H W) d')
        for layer in self.decoder:
            layer_output = layer.forward_cc(
                multiplex,
                multiplex_pos,
                [2] * multiplex_examples,
            ) # type: ignore
            multiplex = layer_output
        if self.norm_after_encoder_decoder:
            multiplex = self.layer_norm(multiplex)
        SELF_ATTENTION_BIAS_CACHE.clear()
        multiplex = multiplex[1::2] 
        multiplex = self.multiplex_decoder_mlp(multiplex)  # (sum_C) (H W) (p q)
        multiplex = rearrange(multiplex, "(C) (H W) (p q) -> C (H p) (W q)", H=H, W=W, p=self.patch_size, q=self.patch_size)
        multiplex = torch.split(multiplex, x_channels_per_sample, dim=0)  # List[(C_i) (H p) (W q)]

        return TerraMeshViTDecoderOutput(
            decoded_multiplex=multiplex,
        )

        

        



                
    


            
class TerraMeshViT(nn.Module):

    VALID_PRIOR_BIAS_EMBEDDING_TYPES = {'zero', 'learnable', 'wl', 'dem', 'one_hot', 'empty'}
    VALID_PATTERN_BLOCKS = {'h', 'v', 'f', 'p'}
    VALID_PRIOR_BIAS_EMBEDDING_FUSION_TYPES = {'add', 'cross-attn', 'concatenate'}
    VALID_POSITIONAL_EMBEDDING_TYPES = {'learnable', 'absolute_beginning', 'rope'}
    def __init__(self,
                 use_default_config: bool,
                 custom_config: Dict[str, Any] | None = None,
                 prior_bias_embeddings: torch.Tensor | None = None,
                 prior_bias_embedding_type: str | None = None ,
                 patch_size: int | None = None ,
                 model_dim: int | None = None ,
                 feedforward_dim: int | None = None ,
                 encoder_pattern: str | None = None ,
                 num_encoder_heads: int | None = None ,
                 decoder_pattern: str | None = None ,
                 num_decoder_heads: int | None = None ,
                 num_hidden_layers: int | None = None ,
                 positional_embedding_type: str | None = None ,
                 dropout: float | None = None ,
                 group_layers: float | None = None ,
                 norm_after_encoder_decoder: bool | None = None,
                 verbose: bool = True,
                 prior_bias_embedding_fusion_type: str | None = 'add',
                 **kwargs: Any):
        """
        Used kwargs:
            num_registers: int = 0,
            max_pos: int = 100,
            max_width_or_height: int = 100,
            parameter_init_power: float = 0.5,
        """
        super().__init__()
        self.verbose = verbose
        

        self.config_params: Dict[str, Any] = {
            'prior_bias_embeddings': prior_bias_embeddings,
            'prior_bias_embedding_type': prior_bias_embedding_type,
            'patch_size': patch_size,
            'model_dim': model_dim,
            'feedforward_dim': feedforward_dim,
            'encoder_pattern': encoder_pattern,
            'num_encoder_heads': num_encoder_heads,
            'decoder_pattern': decoder_pattern,
            'num_decoder_heads': num_decoder_heads,
            'num_hidden_layers': num_hidden_layers,
            'positional_embedding_type': positional_embedding_type,
            'dropout': dropout,
            'group_layers': group_layers,
            'norm_after_encoder_decoder': norm_after_encoder_decoder,
            'prior_bias_embedding_fusion_type': prior_bias_embedding_fusion_type,
            **kwargs
        }

        if use_default_config:
            self._set_default_config()
        elif custom_config is not None:
            for key, value in custom_config.items():
                if key in self.config_params:
                    self.config_params[key] = value
                else:
                    raise ValueError(f"Unknown configuration key: {key}")
        self._check_config_params()
    

        self.encoder = TerraMeshViTEncoder(
            prior_bias_embeddings=prior_bias_embeddings,
            prior_bias_embedding_type=prior_bias_embedding_type,
            patch_size=patch_size,
            model_dim=model_dim,
            feedforward_dim=feedforward_dim,
            encoder_pattern=encoder_pattern,
            num_encoder_heads=num_encoder_heads,
            dropout=dropout,
            group_layers=group_layers,
            norm_after_encoder_decoder=norm_after_encoder_decoder,
            positional_embedding_type=positional_embedding_type,
            verbose=verbose,
            **kwargs
        )

        self.decoder = TerraMeshViTDecoder(
            patch_size=patch_size,
            model_dim=model_dim,
            feedforward_dim=feedforward_dim,
            decoder_pattern=decoder_pattern,
            num_decoder_heads=num_decoder_heads,
            dropout=dropout,
            group_layers=group_layers,
            norm_after_encoder_decoder=norm_after_encoder_decoder,
            positional_embedding_type=positional_embedding_type,
            num_hidden_layers_head=num_hidden_layers,
            **kwargs)
        

    def forward(self, 
                multiplex: List[torch.Tensor],
                channel_ids: List[torch.Tensor],
                multiplex_mask: Optional[List[torch.Tensor]] = None,
                sensor_ids: Optional[List[torch.Tensor]] = None,
                projection_indices: Optional[torch.Tensor | List[torch.Tensor]] = None,
    ) -> TerraMeshViTOutput:
        mx_channels_per_sample = [len(ch) for ch in channel_ids]
        encoder_output = self.encoder.forward_list(
            multiplex,
            channel_ids,
            multiplex_mask=multiplex_mask,
            sensor_ids=sensor_ids,
            projection_indices=projection_indices,
        )
        decoder_output = self.decoder.forward_list(
            encoder_output.encoded_multiplex,
            encoder_output.patch_summary_tokens,
            mx_channels_per_sample,
        )
        return TerraMeshViTOutput(
            decoded_multiplex=decoder_output.decoded_multiplex,
            patch_summary=encoder_output.patch_summary_tokens,
            channel_token_embeddings=encoder_output.encoded_multiplex
        )




        
            
    def _set_default_config(self):
        from .configs.default_multiplex_config import DEFAULT_MULTIPLEX_CONFIG
        for key, value in DEFAULT_MULTIPLEX_CONFIG.items():
            if self.config_params.get(key) is None:
                self.config_params[key] = value
                if self.verbose:
                    print(f"Setting default for {key}: {value}")

    def _check_config_params(self):
        assert self.config_params['prior_bias_embedding_type'] in self.VALID_PRIOR_BIAS_EMBEDDING_TYPES, f"prior_bias_embedding_type must be one of {self.VALID_PRIOR_BIAS_EMBEDDING_TYPES}"
        encoder_pattern_set = set(self.config_params['encoder_pattern'])
        decoder_pattern_set = set(self.config_params['decoder_pattern'])
        assert encoder_pattern_set.issubset(self.VALID_PATTERN_BLOCKS), f"encoder_pattern can only contain {self.VALID_PATTERN_BLOCKS}"
        assert decoder_pattern_set.issubset(self.VALID_PATTERN_BLOCKS), f"decoder_pattern can only contain {self.VALID_PATTERN_BLOCKS}"        
        assert self.config_params['prior_bias_embedding_fusion_type'] in self.VALID_PRIOR_BIAS_EMBEDDING_FUSION_TYPES, f"prior_bias_embedding_fusion_type must be one of {self.VALID_PRIOR_BIAS_EMBEDDING_FUSION_TYPES}"
        assert self.config_params['positional_embedding_type'] in self.VALID_POSITIONAL_EMBEDDING_TYPES, f"positional_embedding_type must be one of {self.VALID_POSITIONAL_EMBEDDING_TYPES}"
        assert self.config_params['model_dim'] % self.config_params['num_encoder_heads'] == 0, "model_dim must be divisible by num_encoder_heads"
        assert self.config_params['model_dim'] % self.config_params['num_decoder_heads'] == 0, "model_dim must be divisible by num_decoder_heads"




    def compile_rope(self):
        for module in self.modules():
            if isinstance(module, (MarkerAttentionEncoderBlock, ChannelAttentionEncoderBlock, FullAttentionEncoderBlock)):
                for submodule in module.modules():
                    if isinstance(submodule, (RotaryPositionalEmbedding2D)):
                        submodule.compile()