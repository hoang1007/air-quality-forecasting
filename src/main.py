import os
import time
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from models import AQFModel
from dataset import *
from dataset.preprocessing import imputation
from utils.export import batch_export
import warnings
warnings.filterwarnings('ignore')


ROOTDIR = os.getcwd()
DATADIR = os.path.join(ROOTDIR, "private-data")
CKPT_DIR = os.path.join(ROOTDIR, "ckpt")
CKPT_PATH = os.path.join(CKPT_DIR, "pretrained.ckpt")
LOGDIR = os.path.join(ROOTDIR, "logs")
EXPORT_DIR = os.path.join(ROOTDIR, "submit")


def train(cfg, device):
    model = AQFModel(
        cfg.training,
        cfg.data.normalize_mean["PM2.5"],
        cfg.data.normalize_std["PM2.5"]
    )

    dtm = PrivateDataModule(
        rootdir=DATADIR,
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        train_ratio=0.75,
        batch_size=cfg.training.batch_size
    )

    logger = TensorBoardLogger(LOGDIR, name="aqf-base", version="v1")

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

    model = AQFModel(
        cfg.training,
        cfg.data.normalize_mean["PM2.5"],
        cfg.data.normalize_std["PM2.5"]
    )

    state_dict = torch.load(CKPT_PATH, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)

    test_dts = PrivateDataset(
        rootdir=DATADIR,
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        split_ids=None,
        data_set="test"
    )

    batch_export(
        export_dir=EXPORT_DIR,
        model=model,
        data=test_dts
    )

def prepare_data():
    if not path.exists(DATADIR):
        shutil.unpack_archive(os.path.join(ROOTDIR, "private-data.zip"), DATADIR)
        # time.sleep(1) # wait for unzip data

    # impute train data
    imputation(os.path.join(DATADIR, "train/air"), method="idw")

    for dirpath in os.scandir(os.path.join(DATADIR, "test")):
        imputation(dirpath.path, method="idw")

@hydra.main(config_path="../config", config_name="config", version_base=None)
def run(cfg):
    pl.seed_everything(3107)

    if cfg.mode == "train":
        train(cfg.model, cfg.device)
    elif cfg.mode == "test":
        test(cfg.model, cfg.device)
    elif cfg.mode == "makedata":
        prepare_data()


if __name__ == "__main__":
    run()