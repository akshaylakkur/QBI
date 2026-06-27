"""GATv2 surrogate for steric exposure regression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class ExposureGNN(nn.Module):
    """2-layer GATv2 regressor predicting steric exposure in [0, 1]."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        out_channels: int = 32,
        edge_dim: int = 4,
        dropout: float = 0.2,
        heads: int = 2,
    ):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_channels, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True
        )
        self.conv2 = GATv2Conv(
            hidden_dim * heads, out_channels, heads=1, edge_dim=edge_dim, concat=False
        )
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return torch.sigmoid(self.head(x))
