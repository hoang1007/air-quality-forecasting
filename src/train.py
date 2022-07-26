import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from model import AQFModel
from dataset import AirQualityDataModule


@hydra.main(config_path="../config", config_name="cfg1")
def run(cfg):
    model = AQFModel(
        cfg.training,
        cfg.data.normalize_mean["PM2.5"], 
        cfg.data.normalize_std["PM2.5"]
    )

    dtm = AirQualityDataModule(
        rootdir="/home/hoang/Documents/CodeSpace/air-quality-forecasting/data",
        output_frame_size=cfg.training.output_dim,
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        fillnan_fn=lambda x : x.interpolate(option="spline").bfill(),
        batch_size=cfg.training.batch_size
    )

    logger = TensorBoardLogger("/home/hoang/Documents/CodeSpace/air-quality-forecasting/logs", name="aqf_model")
    trainer = pl.Trainer(logger=logger, accelerator="gpu", min_epochs=2)
    trainer.fit(model, dtm)
    

if __name__ == "__main__":
    run()