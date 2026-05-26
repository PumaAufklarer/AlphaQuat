"""Tests for Transformer model forward pass."""

import torch

from alpha_quat.model.nn.transformer.models.transformer import StockTransformer


def test_transformer_output_shape():
    model = StockTransformer(n_features=6, d_model=32, nhead=2, n_layers=2, n_bins=100)
    x = torch.randn(4, 60, 6)  # (batch=4, seq=60, features=6)
    out = model(x)
    assert out.shape == (4, 6, 100)


def test_transformer_output_probabilities():
    model = StockTransformer(n_features=6, d_model=32, nhead=2, n_layers=2, n_bins=100)
    model.eval()
    x = torch.randn(2, 60, 6)
    with torch.no_grad():
        out = model(x)
    probs = torch.softmax(out, dim=-1)
    # Each head sums to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 6), atol=1e-5)


def test_transformer_gradient_flow():
    model = StockTransformer(n_features=6, d_model=16, nhead=2, n_layers=2, n_bins=20)
    x = torch.randn(4, 60, 6, requires_grad=True)
    y = torch.randn(4, 6, 20)
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out.view(-1, 20), y.view(-1, 20))
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
