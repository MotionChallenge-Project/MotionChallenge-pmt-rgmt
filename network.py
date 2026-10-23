# Inference architecture from PMT; see THIRD_PARTY_LICENSES.
# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause
import math
import torch
from torch import nn


def MLP(input_dim, output_dim, hidden_dims, activation='elu'):
    if activation != 'elu':
        raise ValueError('unsupported activation')
    dims = [input_dim, *hidden_dims, output_dim]
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers.extend([nn.Linear(a, b), nn.ELU()])
    return nn.Sequential(*layers[:-1])


class Normalizer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer('_mean', torch.zeros(1, size))
        self.register_buffer('_var', torch.ones(1, size))
        self.register_buffer('_std', torch.ones(1, size))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))

    def forward(self, value):
        return (value - self._mean) / (self._std + .01)


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None, activation: str = "elu") -> None:
        super().__init__()
        hidden_dim = out_dim if hidden_dim is None else hidden_dim
        self.net = MLP(in_dim, out_dim, [hidden_dim], activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        t = x.shape[1]
        return x + self.pe[:t].to(dtype=x.dtype).unsqueeze(0)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.ReLU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, *, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x_ln = self.ln_attn(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln_ff(x))
        return self.out_ln(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.ReLU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: [B, 1, D], kv: [B, S, D]
        q_ln = self.ln_q(q)
        attn_out, _ = self.attn(q_ln, kv, kv, need_weights=False)
        x = q + attn_out
        x = x + self.ff(self.ln_ff(x))
        return self.out_ln(x)

class RGMTActor(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.history_embed = TwoLayerMLP(93, 128)
        self.history_pos = SinusoidalPositionalEncoding(128, 64)
        self.history_block = SelfAttentionBlock(128, 4)
        self.q_mlp = TwoLayerMLP(128, 128)
        self.cmd_embed = TwoLayerMLP(38, 128)
        self.cmd_pos = SinusoidalPositionalEncoding(128, 64)
        self.cmd_block = CrossAttentionBlock(128, 4)
        self.actor_trunk = MLP(221, 128, [512, 256])
        self.mean_head = nn.Linear(128, 29)
        self.actor_obs_normalizer = Normalizer(93)
        self.history_obs_normalizer = Normalizer(93)
        self.command_obs_normalizer = Normalizer(38)
        self.register_buffer('causal_mask', torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1), persistent=False)
        unused = ('critic.', 'critic_obs_normalizer.', 'raw_std_param')
        self.load_state_dict({k: v for k, v in state.items() if not k.startswith(unused)}, strict=True)
        self.eval()

    def forward(self, current, history, command):
        history = self.history_obs_normalizer(history.reshape(-1, 93)).reshape(-1, 10, 93)
        command = self.command_obs_normalizer(command.reshape(-1, 38)).reshape(-1, 21, 38)
        h = self.history_pos(self.history_embed(history))
        h = self.history_block(h, attn_mask=self.causal_mask).max(dim=1).values
        query = self.q_mlp(h).unsqueeze(1)
        tokens = self.cmd_pos(self.cmd_embed(command))
        latent = self.cmd_block(query, tokens).squeeze(1)
        inputs = torch.cat([self.actor_obs_normalizer(current), latent], dim=-1)
        return self.mean_head(self.actor_trunk(inputs))
