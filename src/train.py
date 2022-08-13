from gc import callbacks
from subprocess import call
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from model import DAQFFModel
from dataset import *
from utils.export import export, export_dqaff
import warnings
warnings.filterwarnings('ignore')


@hydra.main(config_path="../config", config_name="daqff-big")
def run(cfg):
    ckpt = "/home/hoang/Documents/CodeSpace/air-quality-forecasting/ckpt/epoch=17-step=162.ckpt"
    model = DAQFFModel(
        cfg.training,
        cfg.data.normalize_mean["PM2.5"], 
        cfg.data.normalize_std["PM2.5"]
    ).load_from_checkpoint(ckpt)

    # dtm = AirQualityDataModuleV3(
    #     rootdir="/home/hoang/Documents/CodeSpace/air-quality-forecasting/data",
    #     normalize_mean=cfg.data.normalize_mean,
    #     normalize_std=cfg.data.normalize_std,
    #     droprate=0.5,
    #     fillnan_fn=lambda x : x.interpolate(option="spline").bfill(),
    #     batch_size=cfg.training.batch_size
    # )

    # logger = TensorBoardLogger("/home/hoang/Documents/CodeSpace/air-quality-forecasting/logs", name="dqaff", version="v2")
    # trainer = pl.Trainer(
    #     logger=logger,
    #     accelerator="cpu",
    #     max_epochs=50,
    #     callbacks=[
    #         ModelCheckpoint(dirpath="/home/hoang/Documents/CodeSpace/air-quality-forecasting/ckpt", filename="trained")
    #     ])

    # trainer.fit(model, dtm)

    test_dts = AirQualityDataset2(
        rootdir="/home/hoang/Documents/CodeSpace/air-quality-forecasting/data",
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        droprate=0.5,
        fillnan_fn=lambda x : x.interpolate(option="spline").bfill(),
        data_set="test"
    )

    torch.save((test_dts.data, 100))

    export_dqaff(export_dir="/home/hoang/Documents/CodeSpace/air-quality-forecasting/submit", model=model, data=test_dts)


if __name__ == "__main__":
    pl.seed_everything(188)
    run()