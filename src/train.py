import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from model import DAQFFModel
from dataset import *
from utils.export import export
import torch


@hydra.main(config_path="C:/Users/hoang/OneDrive/Documents/air-quality-forecasting-1/config", config_name="daqff")
def run(cfg):
    model = DAQFFModel(
        cfg.training,
        cfg.data.normalize_mean["PM2.5"], 
        cfg.data.normalize_std["PM2.5"]
    )

    dtm = AirQualityDataModuleV2(
        rootdir="C:/Users/hoang/OneDrive/Documents/air-quality-forecasting-1/data",
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        droprate=0.5,
        fillnan_fn=lambda x : x.interpolate(option="spline").bfill(),
        split_mode="timestamp",
        batch_size=cfg.training.batch_size
    )

    logger = TensorBoardLogger("C:/Users/hoang/OneDrive/Documents/air-quality-forecasting-1/logs", name="dqaff", version="v3")
    trainer = pl.Trainer(logger=logger, accelerator="cpu", max_epochs=10)
    trainer.fit(model, dtm)

    # export(export_dir="/home/hoang/Documents/CodeSpace/air-quality-forecasting/submit", model=model, data=dts)
    

if __name__ == "__main__":
    run()