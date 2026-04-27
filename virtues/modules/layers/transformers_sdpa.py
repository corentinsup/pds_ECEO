from typing import Optional, Tuple, Sequence, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

#from .attention_multiheads import MHAwithPosEmb
from .attention_sdpa import MHAwithPosEmb
from .basic_modules import build_feedforward
from .mask_utils_flashattention import (
    build_self_attention_bias,
    build_self_attention_bias_channel_concat,
    get_non_zero_indices,
)


def _block_diagonal_keep_mask(seg_lens: torch.Tensor) -> torch.Tensor:
    """Build a (N, N) bool mask where True = positions can attend to each other.
    Two positions can attend iff they belong to the same segment.

    Args:
        seg_lens: 1D int tensor of segment lengths summing to N.
    Returns:
        (N, N) bool tensor; True means "keep" (SDPA convention).
    """
    if seg_lens.numel() == 0:
        return torch.zeros(0, 0, dtype=torch.bool, device=seg_lens.device)
    seg_id = torch.repeat_interleave(
        torch.arange(seg_lens.numel(), device=seg_lens.device, dtype=torch.long),
        seg_lens.to(torch.long),
    )
    return seg_id[:, None] == seg_id[None, :]


class TransformerEncoder(nn.Module):
    """
    Stacked pre-LN Transformer encoder.

    Args:
        d_model (int): Model width.
        num_heads (int): Number of attention heads.
        dim_feedforward (int): Hidden size of FFN.
        dropout (float): Dropout prob used inside attention and FFN.
        activation (str): Activation used in FFN (e.g., "gelu" / "relu").
        bias (bool): Whether to use bias in linear projections / LayerNorm.
        inbuilt_pos_emb (str|None): Passed to MHAwithPosEmb (e.g., "absolute", "rope", ...).
        num_layers (int): Number of encoder layers.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "gelu",
        bias: bool = True,
        inbuilt_pos_emb: Optional[str] = "absolute",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    bias=bias,
                    inbuilt_pos_emb=inbuilt_pos_emb,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        src: torch.Tensor,                        # (B, S, d_model)
        src_pos: Optional[torch.Tensor] = None,   # (B, S, 2)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, S) bool or additive
        attn_mask: Optional[torch.Tensor] = None,
        #cu_seq_len: Optional[torch.Tensor] = None,            # FlashAttention varlen
        #max_seq_len: Optional[int] = None,                    # FlashAttention varlen
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            (B, S, d_model)
        """
        x = src
        for layer in self.layers:
            x = layer(
                x,
                src_pos=src_pos,
                src_key_padding_mask=src_key_padding_mask,
                attn_mask=attn_mask,
                #cu_seq_len=cu_seq_len,
                #max_seq_len=max_seq_len,
            )
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Single pre-LN Transformer encoder block:
        x = x + MHA(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "gelu",
        bias: bool = True,
        inbuilt_pos_emb: Optional[str] = "absolute",
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = float(dropout)
        self.activation = activation

        self.multi_head_attention = MHAwithPosEmb(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            inbuilt_pos_emb=inbuilt_pos_emb,
        )
        # Feedforward: d_model -> dim_feedforward -> d_model with activation/dropout
        self.feedforward = build_feedforward(
            in_dim=d_model,
            out_dim=d_model,
            hidden_dims=dim_feedforward,
            activation_fn=activation,
            use_dropout=True,
            dropout_prob=dropout,
        )

        self.layernorm1 = nn.LayerNorm(d_model, bias=bias)
        self.layernorm2 = nn.LayerNorm(d_model, bias=bias)

    def forward(
        self,
        src: torch.Tensor,                        # (B, S, d_model)
        src_pos: Optional[torch.Tensor] = None,   # (B, S, 2)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, S)
        attn_mask: Optional[torch.Tensor] = None, # (S, S) or (B, S, S) bool keep-mask, or additive
        #cu_seq_len: Optional[torch.Tensor] = None,
        #max_seq_len: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Pre-LN MHA
        x = src
        x_norm = self.layernorm1(x)
        # print shapes of all in next line
        # print(f"[TransformerEncoderBlock] x_norm: {x_norm.shape}, src_pos: {src_pos.shape if src_pos is not None else None}, src_key_padding_mask: {src_key_padding_mask.shape if src_key_padding_mask is not None else None}, cu_seq_len: {cu_seq_len.shape if cu_seq_len is not None else None}, max_seq_len: {max_seq_len}")
        x = x + self.multi_head_attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            query_pos=src_pos,
            key_pos=src_pos,
            key_padding_mask=src_key_padding_mask,
            attn_mask=attn_mask,
            #cu_seq_len=cu_seq_len,
            #max_seq_len=max_seq_len,
        )

        # Pre-LN FFN
        x_norm = self.layernorm2(x)
        ff_out = self.feedforward(x_norm)

        x = x + ff_out
        return x


class ChannelAttentionEncoderBlock(nn.Module):
    """
    Encoder over concatenated channels per sample (C x S x D), with varlen FlashAttention.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        inbuilt_pos_emb: Optional[str] = "rope",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers > 1:
            self.encoder_layer = TransformerEncoder(
                d_model=model_dim,
                num_heads=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
                num_layers=num_layers,
            )
        else:
            self.encoder_layer = TransformerEncoderBlock(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
            )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
            x: (B, C, S, D)
            pos: (B, C, S, 2)
        """
        raise NotImplementedError("Not yet implemented for FlashAttention path.")

    def forward_masked(
        self,
        x: torch.Tensor,           # (B, C, S, D)
        pos: torch.Tensor,         # (B, C, S, 2)
        mask: torch.Tensor,        # (B, C, S) True means masked token
        channels_per_sample: Optional[Sequence[int]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Masked variant using varlen FlashAttention over *unmasked* tokens.

        Args:
            x: (B, C, S, D)
            pos: (B, C, S, 2)
            mask: (B, C, S), True means masked.
        """
        raise NotImplementedError("Not yet implemented for FlashAttention path.")

    def forward_cc(
        self,
        x: torch.Tensor,           # (C, S, D)
        pos: torch.Tensor,         # (C, S, 2)
        channels_per_sample: Sequence[int],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Attention over channels concatenated per sample (no masking).

        Args:
            x: (C, S, D)
            pos: (C, S, 2)
            channels_per_sample: list of channel counts per sample.

        Returns:
            x': (C, S, D) 
        """
        mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)  # (C, S) all unmasked
        return self.forward_cc_masked(x, pos, mask, channels_per_sample)

    def forward_cc_masked(
        self,
        x: torch.Tensor,           # (C, S, D)
        pos: torch.Tensor,         # (C, S, 2)
        mask: torch.Tensor,        # (C, S) True means masked
        channels_per_sample: Sequence[int],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Channel attention: spatial tokens within each channel attend to each other.
        Different channels do NOT interact.

        Returns:
            x': (C, S, D) 
        """
        # Select unmasked rows for varlen pass
        mask_indices = get_non_zero_indices("ChannelAttention_cc_Masked_Mask_indices", ~mask)
        x_false = x[mask_indices].unsqueeze(0)     # (1, N, D)
        pos_false = pos[mask_indices].unsqueeze(0) # (1, N, 2)

        # Build block-diagonal keep-mask: tokens can only attend within their own channel.
        # `mask` has shape (C, S); after flattening with `~mask`, the order is row-major (channel-major).
        # So segment lengths = number of unmasked tokens per channel (per row of `mask`).
        unmasked_per_channel = (~mask).sum(dim=1).to(torch.int64)  # (C,)
        # Drop channels with 0 unmasked tokens (they contribute nothing)
        unmasked_per_channel = unmasked_per_channel[unmasked_per_channel > 0]
        keep_mask = _block_diagonal_keep_mask(unmasked_per_channel)  # (N, N)

        out = self.encoder_layer(
            src=x_false,
            src_pos=pos_false,
            attn_mask=keep_mask,
        )
        x_proc = out  # (1, N, D)
        x[mask_indices] = x_proc[0]
        return x


class MarkerAttentionEncoderBlock(nn.Module):
    """
    Encoder attending across markers for each spatial position (C as sequence).
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        inbuilt_pos_emb: Optional[str] = "rope",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers > 1:
            self.encoder_layer = TransformerEncoder(
                d_model=model_dim,
                num_heads=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
                num_layers=num_layers,
            )
        else:
            self.encoder_layer = TransformerEncoderBlock(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
            )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
            x: (B, C, S, D)
            pos: (B, C, S, 2)
        """
        raise NotImplementedError("Not yet implemented for FlashAttention path.")

    def forward_masked(
        self,
        x: torch.Tensor,            # (B, C, S, D)
        pos: torch.Tensor,          # (B, C, S, 2)
        mask: torch.Tensor,         # (B, C, S) True means masked
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Varlen FlashAttention over unmasked items across channels for each spatial position.
        """
        B, C, S, D = x.shape
        x_flat = rearrange(x, "B C S D -> (B S) C D")
        pos_flat = rearrange(pos, "B C S D -> (B S) C D")
        mask_flat = rearrange(mask, "B C S -> (B S) C")

        mask_indices = get_non_zero_indices("MarkerAttention_masked_Mask_indices", ~mask_flat)
        x_false = x_flat[mask_indices].unsqueeze(0)     # (1, N, D)
        pos_false = pos_flat[mask_indices].unsqueeze(0) # (1, N, 2)
        
        # Segments = unmasked channels per (b, s) row
        seg_lens = (~mask_flat).sum(dim=1).to(torch.long)        # (B*S,)
        seg_lens = seg_lens[seg_lens > 0]
        keep_mask = _block_diagonal_keep_mask(seg_lens)  # (N, N) bool
 
        out = self.encoder_layer(
            src=x_false,
            src_pos=pos_false,
            attn_mask=keep_mask,
        )

        x_proc = out
        x_flat[mask_indices] = x_proc[0]
        x = rearrange(x_flat, "(B S) C D -> B C S D", B=B)
        return x

    def forward_cc(
        self,
        x: torch.Tensor,               # (C, S, D)
        pos: torch.Tensor,             # (C, S, 2)
        channels_per_sample: Sequence[int],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Marker attention (unmasked): tokens at the same spatial position within the
        same sample attend to each other across channels. Different samples and
        different spatial positions do NOT interact.
        """
        S = x.shape[1]
        # Per-(spatial_position, sample) segment lengths.
        # The rearrange below iterates s outer, c inner; within each s, c ranges over
        # the concatenated channels-per-sample. So segments are repeated per s:
        # [c_0, c_1, ..., c_{B-1}] x S
        cps = list(channels_per_sample)
        q_lens = torch.as_tensor(cps * int(S), device=x.device, dtype=torch.long)

        # Flatten to a single batch
        x_pack = rearrange(x, "C S D -> (S C) D").unsqueeze(0)    # (1, total, D)
        pos_pack = rearrange(pos, "C S D -> (S C) D").unsqueeze(0)

        keep_mask = _block_diagonal_keep_mask(q_lens)  # (N, N)
        out = self.encoder_layer(
            src=x_pack,
            src_pos=pos_pack,
            attn_mask=keep_mask,
        )
        x_proc = out.squeeze(0)
        x_rec = rearrange(x_proc, "(S C) D -> C S D", S=S)
        return x_rec

    def forward_cc_masked(
        self,
        x: torch.Tensor,               # (C, S, D)
        pos: torch.Tensor,             # (C, S, 2)
        mask: torch.Tensor,            # (C, S) True means masked
        channels_per_sample: Sequence[int],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Marker attention: tokens at the same spatial position within the same sample
        attend to each other across channels. Different samples and different spatial
        positions do NOT interact.
        """
        x_seq = rearrange(x, "C S D -> S C D")
        pos_seq = rearrange(pos, "C S D -> S C D")
        mask_seq = rearrange(mask, "C S -> S C")  # (S, C) True = masked

        mask_indices = get_non_zero_indices("MarkerAttention_cc_Masked_Mask_indices", ~mask_seq)
        x_false = x_seq[mask_indices].unsqueeze(0)     # (1, N, D)
        pos_false = pos_seq[mask_indices].unsqueeze(0) # (1, N, 2)

        # For each spatial position s and each sample b, count unmasked channels.
        # Order in flattened (S, C) is s-major: s=0,b=0; s=0,b=1; ...; s=1,b=0; ...
        S_dim = mask_seq.shape[0]
        unmasked_per_pos_per_sample = torch.zeros(
            S_dim, len(channels_per_sample), device=x.device, dtype=torch.long
        )
        c_offset = 0
        for b, c_b in enumerate(channels_per_sample):
            unmasked_per_pos_per_sample[:, b] = keep[:, c_offset:c_offset + c_b].sum(dim=1)
            c_offset += c_b
 
        seg_lens = unmasked_per_pos_per_sample.reshape(-1)  # (S * B,) s-major
        seg_lens = seg_lens[seg_lens > 0]
        keep_mask = _block_diagonal_keep_mask(seg_lens)     # (N, N) bool

        out = self.encoder_layer(
            src=x_false,
            src_pos=pos_false,
            attn_mask=keep_mask,
        )

        x_proc = out
        x_seq[mask_indices] = x_proc[0]
        x_rec = rearrange(x_seq, "S C D -> C S D")
        return x_rec


class FullAttentionEncoderBlock(nn.Module):
    """
    Encoder over the full (C × S) sequence per sample.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        inbuilt_pos_emb: Optional[str] = "rope",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers > 1:
            self.encoder_layer = TransformerEncoder(
                d_model=model_dim,
                num_heads=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
                num_layers=num_layers,
            )
        else:
            self.encoder_layer = TransformerEncoderBlock(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
            )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
            x: (B, C, S, D)
            pos: (B, C, S, 2)
        """
        raise NotImplementedError("Not yet implemented for FlashAttention path.")

    def forward_masked(
        self,
        x: torch.Tensor,         # (B, C, S, D)
        pos: torch.Tensor,       # (B, C, S, 2)
        mask: torch.Tensor,      # (B, C, S)
    ) -> torch.Tensor:
        """
        Masked full attention path (not implemented in varlen form).
        """
        raise NotImplementedError("Not yet implemented for FlashAttention path.")

    def forward_cc(
        self,
        x: torch.Tensor,                       # (C, S, D)
        pos: torch.Tensor,                     # (C, S, 2)
        channels_per_sample: Sequence[int],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full attention on concatenated (C × S) sequences per sample.

        Args:
            x: (C, S, D)
            pos: (C, S, 2)
            channels_per_sample: list of channel counts per sample.

        Returns:
            x': (C, S, D) 
        """
        S = x.shape[1]
        # Per-sample lengths in the flattened (C S) sequence
        q_lens = torch.as_tensor([c * S for c in channels_per_sample], device=x.device, dtype=torch.long)

        x_pack = rearrange(x, "C S D -> (C S) D").unsqueeze(0)
        pos_pack = rearrange(pos, "C S D -> (C S) D").unsqueeze(0)

        keep_mask = _block_diagonal_keep_mask(q_lens)  # (N, N)

        out = self.encoder_layer(
            src=x_pack,
            src_pos=pos_pack,
            attn_mask=keep_mask,
        )

        x_proc = out.squeeze(0)
        x_rec = rearrange(x_proc, "(C S) D -> C S D", S=S)
        return x_rec

    def forward_cc_masked(
        self,
        x: torch.Tensor,                       # (C, S, D)
        pos: torch.Tensor,                     # (C, S, 2)
        mask: torch.Tensor,                    # (C, S) True means masked
        channels_per_sample: Sequence[int],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full attention within each sample's tokens, masked variant. Only unmasked tokens
        are processed; different samples do NOT interact.
        """
        S = x.shape[1]

        x_flat = rearrange(x, "C S D -> (C S) D")
        pos_flat = rearrange(pos, "C S D -> (C S) D")
        mask_flat = rearrange(mask, "C S -> (C S)")

        mask_indices = get_non_zero_indices("FullAttention_cc_Masked_Mask_indices", ~mask_flat)
        x_false = x_flat[mask_indices].unsqueeze(0)     # (1, N, D)
        pos_false = pos_flat[mask_indices].unsqueeze(0) # (1, N, 2)

        # Segments are samples. After flattening (C, S) row-major, sample b's tokens
        # occupy a contiguous range of c_b * S positions in (C S).
        # For each sample, count unmasked tokens in its slice.
        device = x.device
        seg_lens_list = []
        c_offset = 0
        for c_b in channels_per_sample:
            sample_mask = mask[c_offset:c_offset + c_b]  # (c_b, S)
            n_unmasked = int((~sample_mask).sum().item())
            if n_unmasked > 0:
                seg_lens_list.append(n_unmasked)
            c_offset += c_b
        seg_lens = torch.as_tensor(seg_lens_list, device=device, dtype=torch.long)
        keep_mask = _block_diagonal_keep_mask(seg_lens)  # (N, N)

        out = self.encoder_layer(
            src=x_false,
            src_pos=pos_false,
            attn_mask=keep_mask,
        )

        x_proc = out
        x_flat[mask_indices] = x_proc[0]
        x_rec = rearrange(x_flat, "(C S) D -> C S D", S=S)
        return x_rec


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention from x_query to x_keyval with positional embeddings.

    Args:
        model_dim (int): Model width.
        num_heads (int): Number of heads.
        dropout (float): Dropout prob.
        pos_type (str): Passed to MHAwithPosEmb (e.g., "learnable", "absolute", "rope").
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        pos_type: Optional[str] = "learnable",
    ) -> None:
        super().__init__()
        self.attention_module = MHAwithPosEmb(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            inbuilt_pos_emb=pos_type,
        )

    def forward(
        self,
        x_query: torch.Tensor,             # (C_total, S, D)
        x_keyval: torch.Tensor,            # (C_total, S, D)
        pos: torch.Tensor,                 # (C_total, S, 2)
        multiplex_channels_per_sample: Sequence[int],
    ) -> torch.Tensor:
        """
        Pack sequences by sample (length = channels_i * S) and apply cross-attention.

        Returns:
            (C_total, S, D)
        """
        C_total, S, D = x_query.shape
        q_lens = torch.as_tensor([c * S for c in multiplex_channels_per_sample], device=x_query.device, dtype=torch.long)

        # Pack as a single batch item
        _x_attn = rearrange(x_query, "C S D -> (C S) D").unsqueeze(0)  # (1, sumL, D)
        _prot = rearrange(x_keyval, "C S D -> (C S) D").unsqueeze(0)
        _pos = rearrange(pos, "C S D -> (C S) D").unsqueeze(0)

        # Block-diagonal keep-mask: query in sample b can only attend to keys in sample b.
        # Q and KV have the same length here, so this is the same square mask we use elsewhere.
        keep_mask = _block_diagonal_keep_mask(seg_lens)  # (N, N) bool
 
        ca = self.attention_module(
            query=_x_attn,
            key=_prot,
            value=_prot,
            query_pos=_pos,
            key_pos=_pos,
            attn_mask=keep_mask,
        )
        ca = rearrange(ca.squeeze(0), "(C S) D -> C S D", S=S)
        return ca


class PatchAttentionBlock(nn.Module):
    """
    Self-attention among patch summary tokens (first token per patch/channel).

    Useful after blocks that produce per-patch summary tokens you want to refine jointly.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        inbuilt_pos_emb: Optional[str] = "rope",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers > 1:
            self.encoder_layer = TransformerEncoder(
                d_model=model_dim,
                num_heads=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
                num_layers=num_layers,
            )
        else:
            self.encoder_layer = TransformerEncoderBlock(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                inbuilt_pos_emb=inbuilt_pos_emb,
            )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (B, C, S, D)
            pos: (B, C, S, 2)

        Returns:
            x': (B, C, S, D) with updated first (summary) token per C.
        """
        B, C, S, D = x.shape

        # Take only the patch summary tokens: the first token across channels (index 0)
        ps = x[:, 0]      # (B, S, D)
        psp = pos[:, 0]   # (B, S, 2)

        # Pack into single long sequence
        ps = rearrange(ps, "B S D -> (B S) D").unsqueeze(0)    # (1, B*S, D)
        psp = rearrange(psp, "B S D -> (B S) D").unsqueeze(0)  # (1, B*S, 2)

        # Block-diagonal keep-mask: each sample's S tokens form a segment.
        seg_lens = torch.full((B,), int(S), device=x.device, dtype=torch.long)
        keep_mask = _block_diagonal_keep_mask(seg_lens)  # (B*S, B*S) bool

        ps = self.encoder_layer(src=ps, src_pos=psp, attn_mask=keep_mask)
        ps = rearrange(ps.squeeze(0), "(B S) D -> B S D", S=S)

        # Write back refined summary tokens
        x[:, 0] = ps
        return x

    def forward_masked(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Patch summary tokens are assumed to be always present; just call forward().
        """
        return self.forward(x, pos, **kwargs)

    def forward_cc(
        self,
        x: torch.Tensor,                       # (C, S, D)
        pos: torch.Tensor,                     # (C, S, 2)
        channels_per_sample: Sequence[int],
        **kwargs,
    ) -> torch.Tensor:
        """
        Gather the first token of each channel group (per sample) and refine via attention.

        Args:
            x: (C, S, D)
            pos: (C, S, 2)
            channels_per_sample: list with counts per sample; positions of first tokens
                                 are cumulative sums of these counts.
        """
        C, S, D = x.shape

        # Indices of the first token per channel group: cumulative sums across channel counts
        ps_position = np.cumsum(channels_per_sample)
        ps_position -= ps_position[0]  # shift to start at 0

        ps = x[ps_position]       # (B, S, D) where B = len(ps_position)
        psp = pos[ps_position]    # (B, S, 2)

        ps = rearrange(ps, "B S D -> (B S) D")
        psp = rearrange(psp, "B S D -> (B S) D")

        B_eff = len(ps_position)
        seq_lens = torch.full((B_eff,), int(S), device=x.device, dtype=torch.long)
        keep_mask = _block_diagonal_keep_mask(seq_lens)

        ps = self.encoder_layer(
            src=ps.unsqueeze(0), 
            src_pos=psp.unsqueeze(0), 
            attn_mask=keep_mask
        )

        ps = rearrange(ps.squeeze(0), "(B S) D -> B S D", S=S)

        x[ps_position] = ps
        return x

    def forward_cc_masked(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        channels_per_sample: Sequence[int],
        **kwargs,
    ) -> torch.Tensor:
        """
        Same as forward_cc (summary tokens assumed always present).
        """
        return self.forward_cc(x, pos, channels_per_sample, **kwargs)