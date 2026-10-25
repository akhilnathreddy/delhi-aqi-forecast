"""Temporal Fusion Transformer (Lim et al., 2021) in plain PyTorch.

Faithful to the paper's building blocks: gated residual networks (GRN),
variable selection networks for static / past / known-future inputs, static
covariate encoders that condition selection, the LSTM state and enrichment,
a sequence-to-sequence LSTM, static enrichment, interpretable multi-head
attention (shared values so attention weights average into one
interpretable pattern) and quantile outputs.

One simplification for CPU training: each variable gets its own linear
embedding but the per-variable GRNs are replaced by a single GRN applied to
the selection-weighted sum. Selection weights stay per variable and are the
importance scores reported in ``reports/``.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class GLU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, 2 * d_out)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, -1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """η = LayerNorm(skip(x) + GLU(W1 ELU(W2 x + W3 c)))."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, d_context: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ctx = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x, c=None):
        h = self.fc1(x)
        if self.ctx is not None and c is not None:
            h = h + self.ctx(c)
        h = self.fc2(nn.functional.elu(h))
        return self.norm(self.skip(x) + self.glu(self.drop(h)))


class VariableSelection(nn.Module):
    """Per-variable embeddings, softmax selection weights conditioned on static context."""

    def __init__(self, n_vars: int, d: int, d_context: int | None, dropout: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_vars, d) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n_vars, d))
        self.selector = GRN(n_vars * d, d, n_vars, d_context, dropout)
        self.post = GRN(d, d, d, dropout=dropout)

    def forward(self, x, c=None):
        # x: [..., n_vars] → embeddings [..., n_vars, d]
        e = x.unsqueeze(-1) * self.weight + self.bias
        w = torch.softmax(self.selector(e.flatten(-2), c), -1)
        out = self.post((w.unsqueeze(-1) * e).sum(-2))
        return out, w


class InterpretableAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, heads * self.dk)
        self.k = nn.Linear(d, heads * self.dk)
        self.v = nn.Linear(d, self.dk)  # shared across heads
        self.out = nn.Linear(self.dk, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        B, Tq, _ = q.shape
        Tk = k.shape[1]
        qh = self.q(q).view(B, Tq, self.h, self.dk).transpose(1, 2)
        kh = self.k(k).view(B, Tk, self.h, self.dk).transpose(1, 2)
        att = qh @ kh.transpose(-1, -2) / math.sqrt(self.dk)
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        att = self.drop(att)
        vh = self.v(v).unsqueeze(1)  # [B, 1, Tk, dk]
        head_out = att @ vh  # [B, h, Tq, dk]
        return self.out(head_out.mean(1)), att.mean(1)


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        n_past: int,
        n_future: int,
        n_stations: int,
        n_networks: int,
        horizon: int,
        quantiles: list[float],
        d: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.H, self.quantiles = horizon, quantiles
        self.station_emb = nn.Embedding(n_stations, d)
        self.net_emb = nn.Embedding(n_networks, d)
        self.latlon = nn.Linear(2, d)
        self.static_proj = nn.Linear(3 * d, d)
        self.c_sel, self.c_enr, self.c_h, self.c_c = (GRN(d, d, d, dropout=dropout) for _ in range(4))
        self.past_vsn = VariableSelection(n_past, d, d, dropout)
        self.fut_vsn = VariableSelection(n_future, d, d, dropout)
        self.enc = nn.LSTM(d, d, batch_first=True)
        self.dec = nn.LSTM(d, d, batch_first=True)
        self.post_lstm = GLU(d, d)
        self.post_lstm_norm = nn.LayerNorm(d)
        self.enrich = GRN(d, d, d, d, dropout)
        self.attn = InterpretableAttention(d, heads, dropout)
        self.post_attn = GLU(d, d)
        self.post_attn_norm = nn.LayerNorm(d)
        self.ff = GRN(d, d, d, dropout=dropout)
        self.out_gate = GLU(d, d)
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(quantiles))

    def forward(self, past, future, station, network, latlon, return_weights: bool = False):
        """past [B, L, Fp], future [B, H, Ff] → quantiles [B, H, Q] (scaled log PM2.5)."""
        B, L, _ = past.shape
        s = torch.cat([self.station_emb(station), self.net_emb(network), self.latlon(latlon)], -1)
        s = self.static_proj(s)
        cs, ce, ch, cc = self.c_sel(s), self.c_enr(s), self.c_h(s), self.c_c(s)

        xp, wp = self.past_vsn(past, cs.unsqueeze(1))
        xf, wf = self.fut_vsn(future, cs.unsqueeze(1))
        enc_out, state = self.enc(xp, (ch.unsqueeze(0).contiguous(), cc.unsqueeze(0).contiguous()))
        dec_out, _ = self.dec(xf, state)
        x_in = torch.cat([xp, xf], 1)
        lstm = torch.cat([enc_out, dec_out], 1)
        phi = self.post_lstm_norm(x_in + self.post_lstm(lstm))

        theta = self.enrich(phi, ce.unsqueeze(1))
        T = L + self.H
        q_pos = torch.arange(L, T, device=past.device)
        k_pos = torch.arange(T, device=past.device)
        mask = (k_pos[None, :] <= q_pos[:, None]).view(1, 1, self.H, T)
        att, att_w = self.attn(theta[:, L:], theta, theta, mask)
        delta = self.post_attn_norm(theta[:, L:] + self.post_attn(att))
        psi = self.ff(delta)
        out = self.out_norm(phi[:, L:] + self.out_gate(psi))
        q = self.head(out)
        # Monotone quantiles: median + cumulative positive offsets either side.
        mid = len(self.quantiles) // 2
        base = q[..., mid : mid + 1]
        up = torch.cumsum(nn.functional.softplus(q[..., mid + 1 :]), -1)
        down = torch.cumsum(nn.functional.softplus(q[..., :mid].flip(-1)), -1).flip(-1)
        q = torch.cat([base - down, base, base + up], -1)
        if return_weights:
            return q, {"past": wp, "future": wf, "attention": att_w}
        return q
