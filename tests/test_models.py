"""Shape, quantile ordering and causality checks for the neural models."""

import numpy as np
import torch

from aqf.models.stgnn import WindAwareSTGNN, edge_geometry, wind_support
from aqf.models.tft import TemporalFusionTransformer
from aqf.train import quantile_loss

Q = [0.1, 0.5, 0.9]


def test_tft_shapes_monotone_and_causal():
    torch.manual_seed(0)
    m = TemporalFusionTransformer(n_past=7, n_future=5, n_stations=3, n_networks=2, horizon=12, quantiles=Q, d=16, heads=2, dropout=0.0)
    m.eval()
    past, fut = torch.randn(4, 24, 7), torch.randn(4, 12, 5)
    st, net, ll = torch.tensor([0, 1, 2, 0]), torch.tensor([0, 1, 1, 0]), torch.randn(4, 2)
    q = m(past, fut, st, net, ll)
    assert q.shape == (4, 12, 3)
    assert torch.all(q[..., 0] <= q[..., 1]) and torch.all(q[..., 1] <= q[..., 2])
    # Changing known inputs at lead 8 must not change predictions for leads < 8.
    fut2 = fut.clone()
    fut2[:, 8:] += 5
    q2 = m(past, fut2, st, net, ll)
    assert torch.allclose(q[:, :8], q2[:, :8], atol=1e-5)
    assert not torch.allclose(q[:, 8:], q2[:, 8:])


def test_stgnn_shapes_and_gradient_step():
    torch.manual_seed(0)
    lat = np.array([28.5, 28.6, 28.7, 28.65])
    lon = np.array([77.1, 77.2, 77.3, 77.0])
    m = WindAwareSTGNN(n_past=6, n_future=4, lat=lat, lon=lon, horizon=10, quantiles=Q, d=16)
    past, fut = torch.randn(2, 20, 4, 6), torch.randn(2, 10, 4, 4)
    pw, fw = torch.randn(2, 20, 2), torch.randn(2, 10, 2)
    q = m(past, fut, pw, fw)
    assert q.shape == (2, 10, 4, 3)
    y = torch.randn(2, 10, 4)
    y[0, 0, 0] = float("nan")
    loss = quantile_loss(q, y, Q)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_wind_support_points_downwind():
    lat = np.array([28.6, 28.6])
    lon = np.array([77.0, 77.2])  # station 1 is east of station 0
    ue, un, w = edge_geometry(lat, lon)
    a = wind_support(torch.tensor([[5.0]]), torch.tensor([[0.0]]), ue, un, w)[0, 0]  # westerly wind (blowing east)
    assert a[1, 0] > 0  # the eastern station receives from the western one
    assert a[0, 1] == 0
