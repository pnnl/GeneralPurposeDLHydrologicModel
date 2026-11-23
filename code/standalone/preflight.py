#!/usr/bin/env python3
"""Check dependencies, architecture and paper inputs before GPU allocation."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parents[1]
sys.path.insert(1, str(BUNDLE / "code" / "training"))

import torch
from config import get_config_v24
from model_v15 import StreamflowGNN
from model_d2 import StreamflowGNNMDN
from train_multi_gpu_d2 import D2Trainer
from train_multi_gpu_v16 import V16Trainer
from train_multi_gpu_v20 import SequenceDataModuleV20

config = get_config_v24()
assert config.model.lstm_input_dim == 48
assert config.model.static_input_dim == 40
assert config.data.basin_split_seed == 42
assert config.data.train_basin_fraction == 0.9
paths = [
    config.data.base_dir / config.data.catchment_network,
    config.data.base_dir / "data" / "static" / "catchment_static_attributes.parquet",
    config.data.base_dir / config.data.forcings_processed_dir / "forcings_f16.zarr",
    config.data.base_dir / config.data.streamflow_dir / "usgs",
    BUNDLE / "weights" / "checkpoints_v24" / "finetune" / "split_metadata.json",
    BUNDLE / "weights" / "checkpoints_d2" / "finetune_frozen" / "best_model.pt",
]
missing = [str(p) for p in paths if not p.exists()]
if missing:
    raise FileNotFoundError("Missing paper inputs:\n" + "\n".join(missing))
split = json.loads(paths[4].read_text())
assert (split["n_train_basins"], split["n_eval_basins"]) == (8677, 965)
point = StreamflowGNN(config.model)
mdn = StreamflowGNNMDN(StreamflowGNN(config.model), n_components=3)
from train_multi_gpu_v26 import normalize_state_dict
point_ckpt = torch.load(
    BUNDLE / "weights" / "checkpoints_v24" / "finetune" / "best_model.pt",
    map_location="cpu", weights_only=False)
point.load_state_dict(normalize_state_dict(point_ckpt["model_state_dict"]), strict=True)
mdn_ckpt = torch.load(paths[5], map_location="cpu", weights_only=False)
mdn.load_state_dict(normalize_state_dict(mdn_ckpt["model_state_dict"]), strict=True)
print(json.dumps({
    "status": "ok", "torch": torch.__version__,
    "point_parameters": sum(p.numel() for p in point.parameters()),
    "mdn_parameters": sum(p.numel() for p in mdn.parameters()),
    "training_source": str(BUNDLE / "code" / "training"),
    "train_basins": 8677, "holdout_basins": 965,
}, indent=2))
