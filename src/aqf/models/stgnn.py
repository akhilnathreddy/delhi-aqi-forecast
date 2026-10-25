"""Wind-aware spatio-temporal graph network over the NCR monitoring network.

Each forecast is one graph: every station is a node and the model predicts
all of them jointly for 1–72 h.

* **Temporal encoder** — a GRU shared by all nodes reads each station's
  history.
* **Diffusion graph convolution** (Li et al., DCRNN) over three supports:
  - a fixed distance kernel ``exp(-(d/σ)²)``;
  - an adaptive adjacency learnt from node embeddings (Graph WaveNet);
  - a **wind-advection** adjacency rebuilt for every time step from the
    forecast wind: edge ``j → i`` is weighted by how directly the wind
    blows from station *j* towards station *i* (``relu(cos θ)``), so
    pollution is allowed to propagate downwind.
* **Decoder** — a GRU over the known-future inputs (GFS forecast, SAI,
  calendar), initialised from the spatially mixed encoder state, followed
  by a per-horizon wind-aware graph convolution and quantile head.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..features import haversine_km


def distance_support(lat: np.ndarray, lon: np.ndarray, sigma_km: float = 15.0) -> torch.Tensor:
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    a = np.exp(-((d / sigma_km) ** 2))
    a[d > 3 * sigma_km] = 0
    np.fill_diagonal(a, 0)
    a = a / np.maximum(a.sum(1, keepdims=True), 1e-6)
    return torch.tensor(a, dtype=torch.float32)


def edge_geometry(lat: np.ndarray, lon: np.ndarray, sigma_km: float = 25.0):
    """Unit vectors j→i (east, north) and distance weights, shape [N, N]."""
    de = (lon[None, :] - lon[:, None]) * np.cos(np.radians(lat.mean())) * 111.0  # i → j, east km
    dn = (lat[None, :] - lat[:, None]) * 111.0
    dist = np.hypot(de, dn) + 1e-6
    # Direction from source j to receiver i is minus (i → j).
    ue, un = -de / dist, -dn / dist
    w = np.exp(-((dist / sigma_km) ** 2))
    np.fill_diagonal(w, 0)
    return (torch.tensor(ue, dtype=torch.float32), torch.tensor(un, dtype=torch.float32), torch.tensor(w, dtype=torch.float32))


def wind_support(u: torch.Tensor, v: torch.Tensor, ue, un, w) -> torch.Tensor:
    """u, v: [B, T] network-mean wind → row-normalised adjacency [B, T, N, N]."""
    speed = torch.sqrt(u**2 + v**2).clamp_min(1e-3)
    cos = (u[..., None, None] * ue + v[..., None, None] * un) / speed[..., None, None]
    a = torch.relu(cos) * w * torch.tanh(speed / 2)[..., None, None]
    return a / a.sum(-1, keepdim=True).clamp_min(1e-6)


class DiffusionConv(nn.Module):
    """Σ_supports Σ_k A^k X W_{s,k}  (k = 0 shared)."""

    def __init__(self, d_in: int, d_out: int, n_supports: int, hops: int = 2):
        super().__init__()
        self.hops = hops
        self.lin = nn.Linear(d_in * (1 + n_supports * hops), d_out)

    def forward(self, x, supports):
        # x: [..., N, d]; each support broadcastable to [..., N, N]
        outs = [x]
        for a in supports:
            h = x
            for _ in range(self.hops):
                h = a @ h
                outs.append(h)
        return self.lin(torch.cat(outs, -1))


class WindAwareSTGNN(nn.Module):
    def __init__(
        self,
        n_past: int,
        n_future: int,
        lat: np.ndarray,
        lon: np.ndarray,
        horizon: int,
        quantiles: list[float],
        d: int = 64,
        emb: int = 16,
        dropout: float = 0.1,
        wind_cols: tuple[int, int] = (0, 1),
    ):
        super().__init__()
        n = len(lat)
        self.H, self.quantiles, self.wind_cols = horizon, quantiles, wind_cols
        self.register_buffer("a_dist", distance_support(lat, lon))
        ue, un, w = edge_geometry(lat, lon)
        self.register_buffer("ue", ue)
        self.register_buffer("un", un)
        self.register_buffer("w_geo", w)
        self.e1 = nn.Parameter(torch.randn(n, emb) * 0.1)
        self.e2 = nn.Parameter(torch.randn(n, emb) * 0.1)
        self.node = nn.Parameter(torch.randn(n, d) * 0.1)
        self.inp = nn.Linear(n_past, d)
        self.enc = nn.GRU(d, d, batch_first=True)
        self.gc_enc = nn.ModuleList([DiffusionConv(d, d, 3) for _ in range(2)])
        self.norm_enc = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.fut = nn.Linear(n_future, d)
        self.hemb = nn.Embedding(horizon, d)
        self.dec = nn.GRU(d, d, batch_first=True)
        self.gc_dec = DiffusionConv(d, d, 3, hops=1)
        self.norm_dec = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(quantiles)))

    def adaptive(self) -> torch.Tensor:
        return torch.softmax(torch.relu(self.e1 @ self.e2.T), -1)

    def forward(self, past, future, past_wind, fut_wind):
        """past [B, L, N, Fp], future [B, H, N, Ff], winds [B, L|H, 2] (u, v raw m/s).

        Returns quantiles [B, H, N, Q] in scaled log-PM2.5.
        """
        B, L, N, _ = past.shape
        a_ad = self.adaptive()
        x = self.inp(past).permute(0, 2, 1, 3).reshape(B * N, L, -1)
        _, h = self.enc(x)
        h = h[-1].view(B, N, -1) + self.node
        a_w0 = wind_support(past_wind[:, -1, 0], past_wind[:, -1, 1], self.ue, self.un, self.w_geo)
        for gc, norm in zip(self.gc_enc, self.norm_enc):
            h = norm(h + self.drop(torch.relu(gc(h, [self.a_dist, a_ad, a_w0]))))

        f = self.fut(future) + self.hemb.weight[None, :, None, :]  # [B, H, N, d]
        f = f.permute(0, 2, 1, 3).reshape(B * N, self.H, -1)
        dec, _ = self.dec(f, h.reshape(1, B * N, -1).contiguous())
        dec = dec.view(B, N, self.H, -1).permute(0, 2, 1, 3)  # [B, H, N, d]
        a_wf = wind_support(fut_wind[..., 0], fut_wind[..., 1], self.ue, self.un, self.w_geo)  # [B, H, N, N]
        g = self.norm_dec(dec + self.drop(torch.relu(self.gc_dec(dec, [self.a_dist, a_ad, a_wf]))))
        q = self.head(torch.cat([g, dec], -1))
        mid = len(self.quantiles) // 2
        base = q[..., mid : mid + 1]
        up = torch.cumsum(nn.functional.softplus(q[..., mid + 1 :]), -1)
        down = torch.cumsum(nn.functional.softplus(q[..., :mid].flip(-1)), -1).flip(-1)
        return torch.cat([base - down, base, base + up], -1)
