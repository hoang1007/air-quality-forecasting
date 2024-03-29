import os
from dataset import *
from models import *
from hydra import initialize, compose
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import matplotlib.pyplot as plt


with initialize(version_base=None, config_path="../config/model"):
    cfg = compose(config_name="aqf.yaml")

ROOTDIR = os.getcwd()
DATADIR = os.path.join(ROOTDIR, "data")
CKPT_DIR = os.path.join(ROOTDIR, "ckpt")
CKPT_PATH = os.path.join(CKPT_DIR, "pretrained.ckpt")
LOGDIR = os.path.join(ROOTDIR, "logs")
EXPORT_DIR = os.path.join(ROOTDIR, "submit")

pl.seed_everything(3107)
model = AQFModel(
    cfg.training, cfg.data.normalize_mean["PM2.5"], cfg.data.normalize_std["PM2.5"])


def train(n_epochs: int, batch_size: int):
    dtm = PrivateDataModule(
        "./data-private",
        # normalize_mean={"humidity":0, "temperature": 0, "PM2.5": 0},
        # normalize_std={"humidity": 1, "temperature": 1, "PM2.5": 1},
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        train_ratio=0.75,
        batch_size=batch_size
    )

    ckpt = ModelCheckpoint(CKPT_DIR, filename="aqf", monitor="mae")
    lr_monitor = LearningRateMonitor("epoch")
    logger = TensorBoardLogger(LOGDIR, name="aqf", version="nearest", default_hp_metric=False)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=n_epochs,
        callbacks=[ckpt, lr_monitor],
        accelerator="gpu",
        log_every_n_steps=8
    )

    trainer.fit(model, dtm)
    # lr_finder = trainer.tuner.lr_find(model, dtm)
    # lr_finder.plot(suggest=True)

def test():
    st = torch.load("ckpt/aqf-v1.ckpt")["state_dict"]
    model.load_state_dict(st)
    dtm = PrivateDataModule(
        "./data-private",
        # normalize_mean={"humidity":0, "temperature": 0, "PM2.5": 0},
        # normalize_std={"humidity": 1, "temperature": 1, "PM2.5": 1},
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        train_ratio=0.75,
        batch_size=1
    )

    dtm.setup()
    
    dt = dtm.data_val[10]
    pred = model.predict(dt)
    target = dt["targets"]

    fig, axes = plt.subplots(10, 7, figsize=(20, 20), dpi=100)
    plt.tight_layout()

    for i in range(70):
        row_idx = i // 7
        col_idx = i % 7
        ax = axes[row_idx, col_idx]

        ax.plot(pred[i].cpu())
        ax.plot(target[i].cpu())
        ax.legend(["Predicted", "Actual"])
    plt.show()

    # for i in range(pred.size(0)):
    #     plt.plot(pred[i].cpu())
    # plt.show()

    # model.is_predict = True
    # model.eval()
    # dtm = AirQualityDataModule2(
    #     "./data-full",
    #     # normalize_mean={"humidity":0, "temperature": 0, "PM2.5": 0},
    #     # normalize_std={"humidity": 1, "temperature": 1, "PM2.5": 1},
    #     normalize_mean=cfg.data.normalize_mean,
    #     normalize_std=cfg.data.normalize_std,
    #     train_ratio=0.75,
    #     batch_size=16
    # )
    # dtm.setup()

    # log = []
    # for batch in dtm.val_dataloader():
    #     log.append(model.validation_step(batch, 0))
    
    # print(model.validation_epoch_end(log))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # train(20, 8)
    test()