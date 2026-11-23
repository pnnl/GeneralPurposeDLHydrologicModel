#!/usr/bin/env python3
"""
GNN model v15 — Directed message passing with edge features.

Same architecture as v14, packaged for v15 PUB training.
Key features:
- DirectedRiverGNN replaces RiverGNN: separate GATConvs for upstream and downstream
- Edge features (6-dim) modulate attention weights in GATConv
- Forward signature accepts directed edge indices + edge attributes
- Gated up/down combiner (learned directional weighting)
- LSTM→output skip connection (pre-GNN concat post-GNN)
- Static feature injection into GNN pathway
- Clean output head (no dropout before final prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional, Tuple

from config import ModelConfig, get_config_v15


class DirectedRiverGNN(nn.Module):
    """
    Directed Graph Neural Network for river network message passing.

    Each layer has two separate GATConvs:
    - up_conv: aggregates messages from upstream neighbors (water flows toward this node)
    - down_conv: aggregates messages from downstream neighbors (information propagates upstream)

    Edge features (6-dim) modulate attention weights via GATConv's edge_dim parameter.
    Up/down messages are combined via a learned gate (sigmoid) that weights
    upstream vs downstream information per node.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        edge_feat_dim: int = 6,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        self.gate_projs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        self.up_convs.append(
            GATConv(input_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_feat_dim)
        )
        self.down_convs.append(
            GATConv(input_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_feat_dim)
        )
        self.gate_projs.append(nn.Linear(hidden_dim * 2, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Remaining layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.up_convs.append(
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_feat_dim)
            )
            self.down_convs.append(
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_feat_dim)
            )
            self.gate_projs.append(nn.Linear(hidden_dim * 2, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index_up: torch.Tensor,
        edge_attr_up: torch.Tensor,
        edge_index_down: torch.Tensor,
        edge_attr_down: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [n_nodes, dim] node features
            edge_index_up: [2, n_up_edges] upstream-to-downstream edges (src=upstream, tgt=downstream)
            edge_attr_up: [n_up_edges, edge_feat_dim] features for upstream edges
            edge_index_down: [2, n_down_edges] downstream-to-upstream edges (src=downstream, tgt=upstream)
            edge_attr_down: [n_down_edges, edge_feat_dim] features for downstream edges
        """
        for i in range(self.num_layers):
            x_prev = x

            # Aggregate from upstream neighbors
            h_up = self.up_convs[i](x, edge_index_up, edge_attr_up)

            # Aggregate from downstream neighbors
            h_down = self.down_convs[i](x, edge_index_down, edge_attr_down)

            # Gated combination: learned weighting of upstream vs downstream
            gate = torch.sigmoid(self.gate_projs[i](torch.cat([h_up, h_down], dim=-1)))
            x = gate * h_up + (1 - gate) * h_down
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection
            if x_prev.shape == x.shape:
                x = x + x_prev

        return x


class StreamflowGNN(nn.Module):
    """
    Kratzert-style LSTM + Directed GNN for streamflow prediction (v15).

    Architecture:
        Warmup phase (LSTM only — no GNN):
            For each warmup day t:
                x_t = concat(forcing_t, static_features)
                h_lstm = LSTM(x_t, h_lstm)

        Prediction phase (LSTM + Directed GNN + skip connection):
            For each prediction day t:
                x_t = concat(forcing_t, static_features)
                h_lstm = LSTM(x_t, h_lstm)
                h_pre  = lstm_to_gnn(h_lstm) + static_proj(x_static)
                h_post = DirectedGNN(h_pre, edges...)
                pred_t = output_head(concat(h_pre, h_post)) → Softplus

    Key design choices:
    - Static features projected and added to GNN input so the GNN can
      distinguish basin types during message passing (critical for PUB)
    - Skip connection preserves per-basin LSTM signal through the GNN;
      the output head sees both local (pre-GNN) and spatial (post-GNN)
    - No dropout in output head to avoid corrupting KGE correlation signal
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()

        if config is None:
            config = get_config_v15().model

        self.config = config

        # LSTM: input = forcings + static features concatenated
        self.lstm = nn.LSTM(
            input_size=config.lstm_input_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            batch_first=False,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
        )

        # Input projection: LSTM hidden → GNN input
        self.lstm_to_gnn = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim, config.gnn_hidden_dim),
            nn.ReLU(),
        )

        # Static feature projection → GNN input (gives GNN direct access to
        # basin characteristics for spatial transfer / PUB)
        self.static_proj = nn.Sequential(
            nn.Linear(config.static_input_dim, config.gnn_hidden_dim),
            nn.ReLU(),
        )

        # Directed GNN layers
        self.gnn = DirectedRiverGNN(
            input_dim=config.gnn_hidden_dim,
            hidden_dim=config.gnn_hidden_dim,
            num_layers=config.gnn_num_layers,
            dropout=config.gnn_dropout,
            edge_feat_dim=config.edge_feat_dim,
        )

        # Output head: skip connection feeds concat(pre-GNN, post-GNN)
        # No dropout — avoids corrupting correlation signal in KGE loss
        self.output_head = nn.Sequential(
            nn.Linear(config.gnn_hidden_dim * 2, config.gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.gnn_hidden_dim // 2, config.output_dim),
            nn.Softplus(),
        )

    def lstm_step(
        self,
        x_t: torch.Tensor,
        h_lstm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        LSTM-only step for warmup (no GNN, no output head).

        Args:
            x_t: [n_nodes, n_forcing_vars + n_static_vars]
            h_lstm: (h_n, c_n) tuple or None

        Returns:
            lstm_out: [n_nodes, hidden_dim]
            h_lstm: (h_n, c_n) updated
        """
        x_t_unsqueezed = x_t.unsqueeze(0)
        lstm_out, h_lstm_new = self.lstm(x_t_unsqueezed, h_lstm)
        return lstm_out.squeeze(0), h_lstm_new

    def forward(
        self,
        x_t: torch.Tensor,
        edge_index_up: torch.Tensor,
        edge_attr_up: torch.Tensor,
        edge_index_down: torch.Tensor,
        edge_attr_down: torch.Tensor,
        h_lstm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        x_static: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full forward pass for prediction days (LSTM + Directed GNN + output).

        Args:
            x_t: [n_nodes, n_forcing_vars + n_static_vars]
            edge_index_up: [2, n_up_edges] upstream-to-downstream edges
            edge_attr_up: [n_up_edges, edge_feat_dim]
            edge_index_down: [2, n_down_edges] downstream-to-upstream edges
            edge_attr_down: [n_down_edges, edge_feat_dim]
            h_lstm: (h_n, c_n) tuple or None
            x_static: [n_nodes, n_static_vars] optional raw static features
                      for GNN conditioning (enables basin-type-aware message passing)

        Returns:
            pred: [n_nodes, 1] predicted specific discharge (mm/day)
            h_lstm: (h_n, c_n) updated
        """
        # LSTM step
        x_t_unsqueezed = x_t.unsqueeze(0)
        lstm_out, h_lstm_new = self.lstm(x_t_unsqueezed, h_lstm)
        h_node = lstm_out.squeeze(0)

        # Project to GNN dimension
        h_pre_gnn = self.lstm_to_gnn(h_node)

        # Inject static features so GNN can distinguish basin types
        if x_static is not None:
            h_pre_gnn = h_pre_gnn + self.static_proj(x_static)

        # Directed GNN message passing
        h_post_gnn = self.gnn(h_pre_gnn, edge_index_up, edge_attr_up, edge_index_down, edge_attr_down)

        # Skip connection: concat pre-GNN (local) + post-GNN (spatial)
        h_combined = torch.cat([h_pre_gnn, h_post_gnn], dim=-1)

        # Predict specific discharge (mm/day)
        pred = self.output_head(h_combined)

        return pred, h_lstm_new


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = get_config_v15()

    model = StreamflowGNN(config.model)
    print(f"Model v15 parameters: {count_parameters(model):,}")
    print(f"  LSTM input: {config.model.lstm_input_dim} (forcing + static)")
    print(f"  LSTM: {config.model.lstm_hidden_dim} hidden, {config.model.lstm_num_layers} layers")
    print(f"  GNN: {config.model.gnn_hidden_dim} hidden, {config.model.gnn_num_layers} layers (Directed GAT)")
    print(f"  Edge features: {config.model.edge_feat_dim}")
    print(f"  Static proj: {config.model.static_input_dim} → {config.model.gnn_hidden_dim}")
    print(f"  Skip connection: pre-GNN + post-GNN → output")
    print(f"  Output: Softplus → mm/day")

    # Test forward pass with directed edges
    n_nodes = 100
    n_input = config.model.lstm_input_dim
    n_static = config.model.static_input_dim
    n_warmup = 10
    n_pred = 5
    edge_feat_dim = config.model.edge_feat_dim

    # Build directed edges: simple chain graph
    # up2down: upstream node → downstream node (i → i+1)
    up2down_src = list(range(n_nodes - 1))
    up2down_tgt = list(range(1, n_nodes))
    edge_index_up = torch.tensor([up2down_src, up2down_tgt], dtype=torch.long)
    edge_attr_up = torch.randn(len(up2down_src), edge_feat_dim)

    # down2up: downstream node → upstream node (i+1 → i)
    edge_index_down = torch.tensor([up2down_tgt, up2down_src], dtype=torch.long)
    edge_attr_down = torch.randn(len(up2down_src), edge_feat_dim)

    # Static features (constant per basin)
    x_static = torch.randn(n_nodes, n_static)

    print(f"\nTesting directed edges ({edge_index_up.shape[1]} up, {edge_index_down.shape[1]} down)...")
    print(f"Testing warmup skip ({n_warmup} warmup LSTM-only + {n_pred} prediction LSTM+GNN)...")

    h_lstm = None
    predictions = []

    # Warmup: LSTM only (no GNN)
    for t in range(n_warmup):
        x_t = torch.randn(n_nodes, n_input)
        _, h_lstm = model.lstm_step(x_t, h_lstm)

    # Detach after warmup
    h_lstm = (h_lstm[0].detach(), h_lstm[1].detach())

    # Prediction: full forward with directed edges + static features
    for t in range(n_pred):
        x_t = torch.randn(n_nodes, n_input)
        pred, h_lstm = model(
            x_t, edge_index_up, edge_attr_up, edge_index_down, edge_attr_down,
            h_lstm, x_static=x_static,
        )
        predictions.append(pred)

    predictions = torch.stack(predictions, dim=0)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    print(f"  (in mm/day, non-negative via Softplus)")

    # Also test backward compatibility (no x_static)
    h_lstm2 = None
    for t in range(n_warmup):
        x_t = torch.randn(n_nodes, n_input)
        _, h_lstm2 = model.lstm_step(x_t, h_lstm2)
    h_lstm2 = (h_lstm2[0].detach(), h_lstm2[1].detach())
    x_t = torch.randn(n_nodes, n_input)
    pred2, _ = model(x_t, edge_index_up, edge_attr_up, edge_index_down, edge_attr_down, h_lstm2)
    print(f"Backward compat (no x_static): pred shape {pred2.shape} — OK")

    print("Test PASSED!")
