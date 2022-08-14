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
LOGDIR = os.path.join(ROOTDIR, "logs")
EXPORT_DIR = os.path.join(ROOTDIR, "submit")

@hydra.main(config_path="../config", config_name="daqff")
def run(cfg):
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

    logger = TensorBoardLogger(LOGDIR, name="dqaff", version="v2")
    callbacks = [
        ModelCheckpoint(dirpath=CKPT_DIR)
    ]

    trainer = pl.Trainer(
        logger=logger,
        accelerator="cpu",
        max_epochs=20,
        callbacks=callbacks
    )

    trainer.fit(model, dtm)
    model.load_from_checkpoint(callbacks[0].best_model_path)

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
        train_ratio=0.75,
        batch_size=1
    )
    dtm.setup()

    geo.fit(
        dtm.train_dataloader(),
        dtm.val_dataloader(),
        n_epochs=5,
        device="cuda"
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


if __name__ == "__main__":
    pl.seed_everything(3107)
    # torch.cuda.set_per_process_memory_fraction(0.5)
    run()