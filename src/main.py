import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from models import DAQFFModel
from dataset import *
from models.geometric import SpatialCorrelation
from utils.export import export
import warnings
warnings.filterwarnings('ignore')


ROOTDIR = os.getcwd()
DATADIR = os.path.join(ROOTDIR, "data")
CKPT_DIR = os.path.join(ROOTDIR, "ckpt")
CKPT_PATH = os.path.join(CKPT_DIR, "pretrained.ckpt")
LOGDIR = os.path.join(ROOTDIR, "logs")
EXPORT_DIR = os.path.join(ROOTDIR, "submit")


def train(cfg, device):
    model = DAQFFModel(
        cfg.training,
        cfg.data.normalize_mean["PM2.5"],
        cfg.data.normalize_std["PM2.5"]
    )

    dtm = AirQualityDataModule2(
        rootdir=DATADIR,
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        droprate=0.5,
        train_ratio=0.9,
        batch_size=cfg.training.batch_size
    )

    logger = TensorBoardLogger(LOGDIR, name="dqaff", version="v1")

    ckpt = ModelCheckpoint(
        dirpath=CKPT_DIR,
        save_top_k=1,
        monitor="mae"
    )

    trainer = pl.Trainer(
        logger=logger,
        accelerator=device,
        max_epochs=cfg.training.epochs,
        callbacks=[ckpt]
    )

    trainer.fit(model, dtm)
    os.rename(ckpt.best_model_path, CKPT_PATH)

def test(cfg, device):
    if device == "gpu":
        device = "cuda"

    model = DAQFFModel(
        cfg.training,
        cfg.data.normalize_mean["PM2.5"],
        cfg.data.normalize_std["PM2.5"]
    ).load_from_checkpoint(CKPT_PATH, map_location=device)

    geo = SpatialCorrelation(
        11, 4,
        cfg.data.normalize_mean["PM2.5"],
        cfg.data.normalize_std["PM2.5"]
    )

    dtm = AirQualityDataModule(
        rootdir=DATADIR,
        # normalize_mean={"humidity":0, "temperature": 0, "PM2.5": 0},
        # normalize_std={"humidity": 1, "temperature": 1, "PM2.5": 1},
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        droprate=1.0,
        split_mode="timestamp",
        train_ratio=0.9,
        batch_size=1
    )
    dtm.setup()

    geo.fit(
        dtm.train_dataloader(),
        dtm.val_dataloader(),
        n_epochs=5,
        device=device
    )

    test_dts = AirQualityDataset2(
        rootdir=DATADIR,
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        droprate=0.5,
        fillnan_fn=lambda x : x.interpolate(option="spline").bfill(),
        data_set="test"
    )

    export(
        export_dir=EXPORT_DIR,
        model=model,
        data=test_dts,
        correlations=geo.get_correlation()
    )

@hydra.main(config_path="../config", config_name="config", version_base=None)
def run(cfg):
    pl.seed_everything(3107)

    if cfg.mode == "train":
        train(cfg.model, cfg.device)
    elif cfg.mode == "test":
        test(cfg.model, cfg.device)


if __name__ == "__main__":
    # torch.cuda.set_per_process_memory_fraction(0.5)
    run()