# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------


import numpy as np

import torch


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [n_cls_token+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token > 0:
        pos_embed = np.concatenate(
            [np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


# ----------------------------------------------------------
# RoPE2D: RoPE implementation in 2D
# ----------------------------------------------------------

try:
    from models.curope import cuRoPE2D

    RoPE2D = cuRoPE2D
except ImportError:
    print(
        "Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead"
    )

    class RoPE2D(torch.nn.Module):
        def __init__(self, freq=100.0, F0=1.0):
            """
            freq: the base for computing the inverse frequency (denom = base^(d/D))
            F0: the forward multiplier (typically 1.0)
            """
            super().__init__()
            self.base = freq
            self.F0 = F0
            self.cache = {}  # Cache for precomputed cosine-sine tables

        def get_cos_sin(self, D, seq_len, device, dtype, offset):
            """
            Builds (and caches) a cosine-sine lookup table for positions.
            The table is built for indices 0 ... seq_len-1, but the angle for index t is computed as:
            angle = F0 * (t - offset) / (base^(d/D))
            This way, if positions (p) range from a negative value to a positive value, we can
            use lookup indices p + offset.
            """
            key = (D, seq_len, device, dtype, offset, self.F0)
            if key not in self.cache:
                # Compute inverse frequencies for even indices [0, D)
                inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))
                # Build table indices from 0 to seq_len - 1 and shift by offset
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype) - offset
                # Outer product to get angles for each time step and each frequency
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                # Incorporate the forward multiplier F0
                freqs = self.F0 * freqs
                # Duplicate to cover both even and odd positions in the feature dimension
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos()
                sin = freqs.sin()
                self.cache[key] = (cos, sin)
            return self.cache[key]

        @staticmethod
        def rotate_half(x):
            """
            Splits the last dimension into two halves and rotates them:
            Given x = [x1, x2] returns [-x2, x1].
            """
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rope1d(self, tokens, pos1d, cos, sin, offset):
            """
            Applies 1D RoPE to tokens using a lookup table.
            
            Args:
                tokens: Tensor of shape [B, nheads, ntokens, dim]
                pos1d: Tensor of shape [B, ntokens] with positions (which can be negative)
                cos, sin: Lookup tables of shape [seq_len, dim]
                offset: The shift that was applied to the table.
            
            Returns:
                Tensor with RoPE applied.
            """
            # Adjust positions so that the lookup index is nonnegative.
            pos_adj = (pos1d + offset).long()
            # Lookup cosine and sine values.
            cos_emb = F.embedding(pos_adj, cos)[:, None, :, :]
            sin_emb = F.embedding(pos_adj, sin)[:, None, :, :]
            return tokens * cos_emb + self.rotate_half(tokens) * sin_emb

        def forward(self, tokens, positions):
            """
            Args:
                tokens: Tensor of shape [batch_size, nheads, ntokens, dim] 
                        (dim must be even)
                positions: Tensor of shape [batch_size, ntokens, 2] with (y, x) positions.
                        Positions may be negative.
            
            Returns:
                Tensor of shape [batch_size, nheads, ntokens, dim] with RoPE2D applied.
            """
            # Ensure the feature dimension is even.
            assert tokens.size(3) % 2 == 0, "Feature dimension should be even."
            D = tokens.size(3) // 2
            assert positions.ndim == 3 and positions.shape[-1] == 2, "positions must be [B, ntokens, 2]"
            
            # Determine the offset: if there are negative positions, we need to shift them.
            pos_min = positions.min()
            offset = 0
            if pos_min < 0:
                offset = -int(pos_min.item())
            seq_len = int((positions + offset).max().item()) + 1

            cos, sin = self.get_cos_sin(D, seq_len, tokens.device, tokens.dtype, offset)
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:, :, 0], cos, sin, offset)
            x = self.apply_rope1d(x, positions[:, :, 1], cos, sin, offset)
            tokens = torch.cat((y, x), dim=-1)
            return tokens