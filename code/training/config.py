#!/usr/bin/env python3
"""
Configuration for GNN streamflow prediction model.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch


def _bundle_root() -> Path:
    """Folder that contains code/, data/, and weights/."""
    return Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    """Data paths and parameters."""
    # Paths are relative to the repository root (the directory that contains code/).
    base_dir: Path = field(default_factory=_bundle_root)
    catchment_network: Path = field(default_factory=lambda: Path("data/catchment_network/catchment_network.gpkg"))
    # Full MERIT delineation. Not included. The forcing store is indexed by hruid.
    catchment_attributes: Path = field(default_factory=lambda: Path("data/delineations/conus_MERIT_v1_catch.nov25.gpkg"))
    # Raw VIC parameters are not included.
    static_params: Path = field(default_factory=lambda: Path("data/static/namerica_params.nc"))
    forcings_dir: Path = field(default_factory=lambda: Path("data/forcings_processed_hruid"))
    # Preprocessed daily catchment forcings Zarr store (float16).
    forcings_processed_dir: Path = field(default_factory=lambda: Path("data/forcings_processed_hruid"))
    streamflow_dir: Path = field(default_factory=lambda: Path("data/streamflow"))

    # Forcing variables (from WRF TGW)
    forcing_vars: List[str] = field(default_factory=lambda: [
        'precipitation_mm',
        'temperature_mean_C',
        'temperature_min_C',
        'temperature_max_C',
        'shortwave_radiation_Wm2',
        'longwave_radiation_Wm2',
        'wind_speed_ms',
        'vapor_pressure_Pa',
    ])

    # Static attributes (from catchment_static_attributes.parquet)
    static_vars: List[str] = field(default_factory=lambda: [
        'elevation_m',          # Elevation
        'avg_temperature_C',    # Average temperature
        'annual_precip_mm',     # Annual precipitation
        'july_avg_temp_C',      # July average temperature
        'infiltration_param',   # Infiltration parameter
        'baseflow_Ds',          # Baseflow parameter Ds
        'baseflow_Dsmax',       # Baseflow parameter Dsmax
        'soil_moisture_Ws',     # Soil moisture parameter
        'exponent_c',           # Exponent c
        'soil_depth_dp',        # Soil depth
        'roughness',            # Surface roughness
        'snow_roughness',       # Snow roughness
        'area_m2',              # Catchment area
    ])

    # Temporal parameters
    lookback_days: int = 180  # Days of forcing history for each prediction
    forecast_days: int = 1   # Days ahead to predict (1 = next day)

    # Train/val/test split
    train_years: List[int] = field(default_factory=lambda: list(range(1985, 2015)))  # 30 years
    val_years: List[int] = field(default_factory=lambda: list(range(2015, 2020)))    # 5 years
    test_years: List[int] = field(default_factory=lambda: list(range(2020, 2025)))   # 5 years

    # Split strategy
    # temporal: train/val/test are split by year ranges above
    # basin: train/eval are split by gauged basin IDs; all configured years are used
    split_mode: str = "temporal"    # temporal or basin
    train_basin_fraction: float = 0.8
    basin_split_seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture parameters."""
    # LSTM encoder
    lstm_input_dim: int = 8       # Number of forcing variables
    lstm_hidden_dim: int = 128    # LSTM hidden dimension
    lstm_num_layers: int = 2      # Number of LSTM layers
    lstm_dropout: float = 0.2

    # Static attribute encoder
    static_input_dim: int = 13    # Number of static attributes (from catchment_static_attributes.parquet)
    static_hidden_dim: int = 32   # Static encoder output dim

    # GNN layers
    gnn_hidden_dim: int = 128     # GNN hidden dimension
    gnn_num_layers: int = 3       # Number of GNN layers
    gnn_dropout: float = 0.2
    gnn_type: str = "GAT"         # GCN, GAT, or GraphSAGE

    # Edge features (v11+)
    edge_feat_dim: int = 6        # Number of edge features for directed GNN

    # Output head
    output_dim: int = 1           # Predict single flow value


@dataclass
class TrainConfig:
    """Training parameters."""
    # Optimization
    batch_size: int = 32          # Number of subgraphs per batch
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15            # Early stopping patience

    # Subgraph sampling
    num_neighbors: List[int] = field(default_factory=lambda: [10, 10, 10])  # Per GNN layer
    subgraph_size: int = 512      # Max nodes per subgraph

    # Loss
    loss_type: str = "mse"        # mse or nse (Nash-Sutcliffe)

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpoints
    checkpoint_dir: Path = field(default_factory=lambda: _bundle_root() / "outputs" / "checkpoints")
    save_every: int = 1           # Save checkpoint every epoch (for resume)


@dataclass
class Config:
    """Complete configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        # Ensure paths are Path objects
        if isinstance(self.data.base_dir, str):
            self.data.base_dir = Path(self.data.base_dir)

        # Create checkpoint directory
        self.train.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Get default configuration."""
    return Config()


def get_config_v7() -> Config:
    """
    Configuration for v7 training with improved features.

    Changes from default:
    - Larger LSTM (256 hidden, 3 layers) and GNN (256 hidden)
    - 18 static features from v2 extraction (all spatially varying)
    - Larger static encoder to match richer features
    - Slightly lower learning rate for larger model stability
    """
    config = Config()

    # Model capacity increase
    config.model.lstm_hidden_dim = 256
    config.model.lstm_num_layers = 3
    config.model.gnn_hidden_dim = 256
    config.model.static_input_dim = 11   # v2: 4 climate/terrain + 1 area + 6 geomorphic
    config.model.static_hidden_dim = 64  # larger to match richer features

    # Training
    config.train.learning_rate = 5e-4    # lower for larger model

    return config


def get_config_v8() -> Config:
    """
    Configuration for v8 Kratzert-style day-by-day LSTM training.

    Changes from v7:
    - lstm_input_dim = n_forcing_vars + n_static_vars (concatenated input)
    - No separate StaticEncoder — static features go into LSTM directly
    - 2 LSTM layers (sufficient without lookback window redundancy)
    - Higher learning rate (normalized inputs allow faster convergence)
    """
    config = Config()

    # Model
    config.model.lstm_input_dim = 8 + 11  # forcing_vars(8) + static_vars(11), overridden by actual data
    config.model.lstm_hidden_dim = 256
    config.model.lstm_num_layers = 2      # 2 layers sufficient with day-by-day processing
    config.model.gnn_hidden_dim = 256
    config.model.gnn_num_layers = 3
    config.model.static_input_dim = 11    # will be overridden by actual data
    config.model.static_hidden_dim = 64   # unused in v8 but kept for config compat

    # Training
    config.train.learning_rate = 1e-3     # higher with normalized inputs

    return config


def get_config_v9() -> Config:
    """
    Configuration for v9: larger model + log1p targets + GNN skip during warmup.

    Changes from v8:
    - Larger LSTM: hidden=512, 3 layers (was 256, 2) — more capacity
    - GNN hidden=256 (same as v8)
    - Lower learning rate for larger model stability
    - log1p(specific_discharge) target space (linear output, no Softplus)
    - GNN skipped during warmup for ~2x epoch speedup
    """
    config = Config()

    # Model — larger LSTM
    config.model.lstm_input_dim = 8 + 11  # forcing_vars(8) + static_vars(11), overridden by actual data
    config.model.lstm_hidden_dim = 512    # 2x v8
    config.model.lstm_num_layers = 3      # was 2 in v8
    config.model.lstm_dropout = 0.2
    config.model.gnn_hidden_dim = 256     # same as v8
    config.model.gnn_num_layers = 3
    config.model.static_input_dim = 11    # will be overridden by actual data
    config.model.static_hidden_dim = 64   # unused in v9 but kept for config compat

    # Training — lower lr for larger model
    config.train.learning_rate = 5e-4

    return config


def get_config_v11() -> Config:
    """
    Configuration for v11: Directed message passing + edge features.

    Changes from v10:
    - Same LSTM/GNN dimensions (256x3 / 256x3)
    - edge_feat_dim = 6 (upstream_area_ratio, elev_diff, slope, distance, order_diff, sinuosity)
    - DirectedRiverGNN with separate up/down GATConvs
    """
    config = Config()

    # Model — same as v10
    config.model.lstm_input_dim = 8 + 11  # forcing_vars(8) + static_vars(11), overridden by actual data
    config.model.lstm_hidden_dim = 256
    config.model.lstm_num_layers = 3
    config.model.lstm_dropout = 0.2
    config.model.gnn_hidden_dim = 256
    config.model.gnn_num_layers = 3
    config.model.static_input_dim = 11    # will be overridden by actual data
    config.model.static_hidden_dim = 64   # unused but kept for config compat
    config.model.edge_feat_dim = 6        # directed edge features

    # Training
    config.train.learning_rate = 5e-4

    return config


def get_config_v12() -> Config:
    """
    Configuration for v12: Gauge dropout for PUB (Prediction in Ungauged Basins).

    Same architecture as v11 (LSTM 256x3, DirectedRiverGNN 256x3, 6-dim edge features).
    Training change: randomly mask gauge_dropout_rate (default 0.3) of gauged basins
    from the loss each batch, forcing the GNN to learn spatial transfer.

    gauge_dropout_rate is a CLI arg in train_multi_gpu_v12.py, not stored in config.
    """
    config = Config()

    # Model — same as v11
    config.model.lstm_input_dim = 8 + 11  # forcing_vars(8) + static_vars(11), overridden by actual data
    config.model.lstm_hidden_dim = 256
    config.model.lstm_num_layers = 3
    config.model.lstm_dropout = 0.2
    config.model.gnn_hidden_dim = 256
    config.model.gnn_num_layers = 3
    config.model.static_input_dim = 11    # will be overridden by actual data
    config.model.static_hidden_dim = 64   # unused but kept for config compat
    config.model.edge_feat_dim = 6        # directed edge features

    # Training
    config.train.learning_rate = 5e-4

    return config


def get_config_v13() -> Config:
    """
    Configuration for v13: Pure spatial split for PUB.

    Same architecture as v11/v12 (LSTM 256x3, DirectedRiverGNN 256x3, 6-dim edge features).
    Training changes:
    - Pure spatial split: 95% train / 5% val, all years for both
    - Direct KGE loss from epoch 1 (no MSE warmup)
    - No beta clamp in KGE loss
    """
    config = Config()

    # Model — same as v11/v12
    config.model.lstm_input_dim = 8 + 11  # forcing_vars(8) + static_vars(11), overridden by actual data
    config.model.lstm_hidden_dim = 256
    config.model.lstm_num_layers = 3
    config.model.lstm_dropout = 0.2
    config.model.gnn_hidden_dim = 256
    config.model.gnn_num_layers = 3
    config.model.static_input_dim = 11    # will be overridden by actual data
    config.model.static_hidden_dim = 64   # unused but kept for config compat
    config.model.edge_feat_dim = 6        # directed edge features

    # Training
    config.train.learning_rate = 5e-4

    return config


def get_config_v15() -> Config:
    """
    Configuration for v15: Larger LSTM (512) + spatial PUB split.

    Builds on v14's larger LSTM + directed GNN architecture, but switches the
    data split defaults to basin holdout for PUB experiments.
    Architecture: LSTM 512x3 + DirectedRiverGNN 256x3 + 6-dim edge features.
    Training defaults: basin split, gauge dropout handled in CLI/training script.
    """
    config = get_config_v14()

    # PUB-oriented data split defaults
    config.data.split_mode = "basin"
    config.data.train_basin_fraction = 0.8
    config.data.basin_split_seed = 42

    return config


def get_config_v16() -> Config:
    """
    Configuration for v16: Two-stage PUB pretraining.

    Same architecture as v15 (LSTM 512x3 + DirectedRiverGNN 256x3 + 6-dim edge features).
    Key changes from v15:
    - train_basin_fraction 0.9 (was 0.8) — more training data, 10% holdout still sufficient
    - Two-stage training handled in train_multi_gpu_v16.py:
      Stage 1 (pretrain): temporal split on train basins only, no gauge dropout
      Stage 2 (finetune): basin holdout, gauge dropout 0.15, lower LR
    """
    config = get_config_v14()

    # PUB-oriented data split defaults (overridden by CLI in training script)
    config.data.split_mode = "basin"
    config.data.train_basin_fraction = 0.9
    config.data.basin_split_seed = 42

    return config


def get_config_v17() -> Config:
    """
    Configuration for v17: PUB-oriented architectures (EA-LSTM, Decoupled GNN, Contrastive).

    Same base as v14 (LSTM 512x3) with two-stage PUB training (like v16).
    Model-specific architecture differences are handled in model_pub.py:
    - EA-LSTM: no GNN, static attrs modulate LSTM input gate
    - Decoupled GNN: 64-dim GNN on statics only, LSTM untouched by GNN
    - Contrastive PUB: 64-dim basin encoder + LSTM
    """
    config = get_config_v14()

    # PUB-oriented data split (same as v16)
    config.data.split_mode = "basin"
    config.data.train_basin_fraction = 0.9
    config.data.basin_split_seed = 42

    return config


def get_config_v18() -> Config:
    """
    Configuration for v18: v16 LSTM+DirectedGAT + contrastive basin embeddings.

    Same architecture as v16 (LSTM 512x3 + DirectedRiverGNN 256x3 + 6-dim edge features)
    but with contrastive basin embedding (64-dim) concatenated to LSTM input.
    LSTM input: forcings(8) + static(51) + basin_embed(64) = 123.
    Static input uses CAMELS-style 51-feature attributes (climate, soil, geology, LAI).

    Two-stage PUB training (same as v16):
      Stage 1 (pretrain): Temporal split on train basins
      Stage 2 (finetune): Basin holdout PUB evaluation
    """
    config = get_config_v14()

    # CAMELS-style static attributes (51 features)
    config.model.static_input_dim = 51

    # LSTM input: forcings(8) + static(51) + basin_embed(64) = 123
    config.model.lstm_input_dim = 8 + 51 + 64

    # PUB-oriented data split (same as v16)
    config.data.split_mode = "basin"
    config.data.train_basin_fraction = 0.9
    config.data.basin_split_seed = 42

    return config


def get_config_v19() -> Config:
    """
    Configuration for v19: v16 LSTM+DirectedGAT with CAMELS-style 51-feature attributes.

    Identical architecture to v16 (LSTM 512x3 + DirectedRiverGNN 256x3 + 6-dim edge features).
    Only change: static_input_dim = 51 (CAMELS-style attributes) instead of 13 (old).
    LSTM input: forcings(8) + static(51) = 59.

    This isolates the effect of better attributes from v18's contrastive embeddings.
    """
    config = get_config_v14()

    # CAMELS-style static attributes (51 features)
    config.model.static_input_dim = 51

    # LSTM input: forcings(8) + static(51) = 59
    config.model.lstm_input_dim = 8 + 51

    # PUB-oriented data split (same as v16)
    config.data.split_mode = "basin"
    config.data.train_basin_fraction = 0.9
    config.data.basin_split_seed = 42

    return config


def get_config_v24() -> Config:
    """
    Configuration for v24: LSTM-only baseline with forcings only (no static attributes).

    Naive PUB baseline to quantify how much static attributes contribute.
    LSTM input: forcings only (8 dims). No static attributes, no GNN.
    """
    config = get_config_v14()

    config.model.static_input_dim = 0
    config.model.lstm_input_dim = 8  # forcings only

    config.data.split_mode = "basin"
    config.data.train_basin_fraction = 0.9
    config.data.basin_split_seed = 42

    return config


def get_config_v21() -> Config:
    """
    Configuration for v21: EA-LSTM with v20's curated 40-feature attributes.

    Same as v17 (PUB architectures) but with v20's curated features:
    - Excludes 11 categorical/redundant features (categoricals, redundant, near-constant)
    - Log-transforms 6 heavily skewed features before z-scoring
    - 40 features instead of 51. LSTM input: forcings(8) + static(40) = 48.
    """
    config = get_config_v14()

    config.model.static_input_dim = 40
    config.model.lstm_input_dim = 8 + 40

    config.data.split_mode = "basin"
    config.data.train_basin_fraction = 0.9
    config.data.basin_split_seed = 42

    return config


def get_config_v20() -> Config:
    """
    Configuration for v20: v16 LSTM+DirectedGAT with curated 40-feature attributes.

    Same architecture as v16/v19. Changes from v19:
    - Excludes 11 categorical/redundant features (categoricals, redundant, near-constant)
    - Log-transforms 6 heavily skewed features before z-scoring
    - 40 features instead of 51. LSTM input: forcings(8) + static(40) = 48.
    """
    config = get_config_v14()

    config.model.static_input_dim = 40
    config.model.lstm_input_dim = 8 + 40

    config.data.split_mode = "basin"
    config.data.train_basin_fraction = 0.9
    config.data.basin_split_seed = 42

    return config


def get_config_v14() -> Config:
    """
    Configuration for v14: Larger LSTM (512) with directed GNN.

    Same directed architecture as v11 (DirectedRiverGNN 256x3, 6-dim edge features).
    LSTM increased from 256 to 512 hidden (~5.8M+ params vs v11's 3.5M).
    Rationale: v9 had 6.5M params (LSTM 512x3) and performed well with undirected GNN;
    v14 combines the larger LSTM capacity with v11's superior directed message passing.
    Data defaults remain the original temporal split.
    """
    config = Config()

    # Model — larger LSTM, same directed GNN as v11
    config.model.lstm_input_dim = 8 + 11  # forcing_vars(8) + static_vars(11), overridden by actual data
    config.model.lstm_hidden_dim = 512    # 2x v11 (same as v9)
    config.model.lstm_num_layers = 3
    config.model.lstm_dropout = 0.2
    config.model.gnn_hidden_dim = 256     # same as v11
    config.model.gnn_num_layers = 3
    config.model.static_input_dim = 11    # will be overridden by actual data
    config.model.static_hidden_dim = 64   # unused but kept for config compat
    config.model.edge_feat_dim = 6        # directed edge features

    # Restore original v14 temporal data defaults explicitly
    config.data.train_years = list(range(1985, 2015))
    config.data.val_years = list(range(2015, 2020))
    config.data.test_years = list(range(2020, 2025))
    config.data.split_mode = "temporal"
    config.data.train_basin_fraction = 0.8
    config.data.basin_split_seed = 42

    # Training
    config.train.learning_rate = 5e-4

    return config


def get_config_v10() -> Config:
    """
    Configuration for v10: mid-size model (between v8 and v9).

    Changes from v9:
    - LSTM: hidden=256, 3 layers (was 512, 3) — reduce overfitting
    - GNN hidden=256 (same as v8/v9)
    - Same lr as v9 (5e-4), relies on ReduceLROnPlateau
    """
    config = Config()

    # Model — mid-size LSTM (3.7x fewer params than v9)
    config.model.lstm_input_dim = 8 + 11  # forcing_vars(8) + static_vars(11), overridden by actual data
    config.model.lstm_hidden_dim = 256    # same as v8 (was 512 in v9)
    config.model.lstm_num_layers = 3      # same as v9 (was 2 in v8)
    config.model.lstm_dropout = 0.2
    config.model.gnn_hidden_dim = 256     # same as v8/v9
    config.model.gnn_num_layers = 3
    config.model.static_input_dim = 11    # will be overridden by actual data
    config.model.static_hidden_dim = 64   # unused but kept for config compat

    # Training
    config.train.learning_rate = 5e-4

    return config


def get_config_v24() -> Config:
    """
    Configuration for v24: GNN (v20 architecture) with area-verified streamflow.

    Same as v20 (LSTM+DirectedGAT, curated 40-feature attributes, WRF TGW forcings).
    Streamflow is data/streamflow under the bundle root.
    - Processed catchment forcings (8 vars, float16 zarr)
    - MERIT streamflow in data/streamflow
    - LSTM input: forcings(8) + static(40) = 48
    """
    config = get_config_v14()

    config.model.static_input_dim = 40
    config.model.lstm_input_dim = 8 + 40

    config.data.split_mode = "basin"
    config.data.train_basin_fraction = 0.9
    config.data.basin_split_seed = 42

    return config


def get_config_v25() -> Config:
    """
    Configuration for v25: EA-LSTM (v21 architecture) with area-verified streamflow.

    Same as v21 (EA-LSTM, curated 40-feature attributes, WRF TGW forcings).
    Streamflow is data/streamflow under the bundle root.
    - Processed catchment forcings (8 vars, float16 zarr)
    - MERIT streamflow in data/streamflow
    - LSTM input: forcings(8) + static(40) = 48
    """
    config = get_config_v24()
    return config


def get_config_v26() -> Config:
    """
    Configuration for v26: v24 trunk + monotonic quantile head (Stage 3).

    Same as v24 architecture and data, but adds 7 quantile outputs
    [P10, P25, P50, P75, P90, P95, P99] for distributional forecasting,
    especially extreme events.

    Stage 3 training: load v24 finetune best, freeze trunk, train only the
    new quantile head with multi-quantile pinball loss.
    """
    config = get_config_v24()
    return config

