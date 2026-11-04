# The kernels in this file are the pure-torch reference implementations from
# transformers' Qwen3.5 model (transformers/models/qwen3_5/modeling_qwen3_5.py,
# v5.15), Copyright The HuggingFace Team, Apache License 2.0. They are vendored so
# that gated-DeltaNet models run on the transformers version tokasaurus pins.
"""Gated delta-rule linear attention (gated DeltaNet) kernels."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# chunk length of the chunked (prefill) delta-rule scan
CHUNK_SIZE = 64


def l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def causal_conv1d(x: Tensor, weight: Tensor) -> Tensor:
    """Depthwise causal conv + SiLU. x: [B, C, L], weight: [C, K] -> [B, C, L]."""
    seq_len = x.shape[-1]
    out = F.conv1d(
        x.to(weight.dtype),
        weight.unsqueeze(1),
        padding=weight.shape[-1] - 1,
        groups=x.shape[1],
    )[:, :, :seq_len]
    return F.silu(out).to(x.dtype)


def causal_conv1d_step(x: Tensor, conv_state: Tensor, weight: Tensor) -> Tensor:
    """One-token causal conv + SiLU that updates conv_state ([B, C, K]) in place.
    x: [B, C, 1] -> [B, C, 1]."""
    state_len = conv_state.shape[-1]
    window = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
    conv_state.copy_(window[:, :, -state_len:])
    out = F.conv1d(window, weight.unsqueeze(1), groups=x.shape[1])[:, :, -1:]
    return F.silu(out).to(x.dtype)


def chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[Tensor, Tensor]:
    """Chunked delta-rule scan for prefill.

    query/key: [B, L, H, Dk], value: [B, L, H, Dv], g/beta: [B, L, H].
    Returns (out [B, L, H, Dv], final state [B, H, Dk, Dv] fp32).
    """
    initial_dtype = query.dtype
    query = l2norm(query)
    key = l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    total_len = seq_len + pad
    query = query * k_head_dim**-0.5

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float().tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(upper, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    out = torch.zeros_like(value)
    for i in range(total_len // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_new = v_i - k_cumdecay[:, :, i] @ state
        out[:, :, i] = (q_i * g[:, :, i, :, None].exp()) @ state + attn @ v_new
        state = (
            state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])[:, :, :seq_len]
    return out.transpose(1, 2).contiguous().to(initial_dtype), state


def recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor,
) -> tuple[Tensor, Tensor]:
    """Token-by-token delta-rule recurrence for decode. Same shapes as the chunked scan."""
    initial_dtype = query.dtype
    query = l2norm(query)
    key = l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, seq_len, _ = key.shape
    query = query * key.shape[-1] ** -0.5
    out = torch.zeros(
        batch_size, num_heads, seq_len, value.shape[-1], device=value.device
    )
    state = initial_state.to(value)

    for i in range(seq_len):
        state = state * g[:, :, i].exp()[..., None, None]
        kv_mem = (state * key[:, :, i].unsqueeze(-1)).sum(dim=-2)
        delta = (value[:, :, i] - kv_mem) * beta[:, :, i].unsqueeze(-1)
        state = state + key[:, :, i].unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * query[:, :, i].unsqueeze(-1)).sum(dim=-2)

    return out.transpose(1, 2).contiguous().to(initial_dtype), state


class RMSNormGated(nn.Module):
    """RMSNorm then SiLU gate. Plain `* weight`, applied after casting back to the
    input dtype (unlike the (1 + w) norms elsewhere in Qwen 3.5)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor, gate: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight * x.to(input_dtype)
        return (x * F.silu(gate.float())).to(input_dtype)
