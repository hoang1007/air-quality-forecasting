import torch
from models.geo.model import SpatialCorrelation
from hydra import initialize, compose
from dataset import AirQualityDataModule

with initialize(version_base=None, config_path="../config/model"):
  cfg = compose(config_name="geo.yaml")


dtm = AirQualityDataModule(
        "data",
        # normalize_mean={"humidity":0, "temperature": 0, "PM2.5": 0},
        # normalize_std={"humidity": 1, "temperature": 1, "PM2.5": 1},
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        droprate=1.0,
        split_mode="timestamp",
        train_ratio=0.75,
        batch_size=8
    )
dtm.setup()


model = SpatialCorrelation(cfg.training)

for batch in dtm.train_dataloader():
    model(batch["features"], batch["src_locs"], None, None)
    break