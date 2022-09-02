import os
from dataset import *
from models import *
from hydra import initialize, compose
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import matplotlib.pyplot as plt


with initialize(version_base=None, config_path="../config/model"):
    cfg = compose(config_name="gagnn.yaml")

ROOTDIR = os.getcwd()
DATADIR = os.path.join(ROOTDIR, "data")
CKPT_DIR = os.path.join(ROOTDIR, "ckpt")
CKPT_PATH = os.path.join(CKPT_DIR, "pretrained.ckpt")
LOGDIR = os.path.join(ROOTDIR, "logs")
EXPORT_DIR = os.path.join(ROOTDIR, "submit")

pl.seed_everything(3107)
model = GAGNNModel(
    cfg.training, cfg.data.normalize_mean["PM2.5"], cfg.data.normalize_std["PM2.5"])


def train(n_epochs: int, batch_size: int):
    dtm = AirQualityDataModule(
        "./data-full",
        # normalize_mean={"humidity":0, "temperature": 0, "PM2.5": 0},
        # normalize_std={"humidity": 1, "temperature": 1, "PM2.5": 1},
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        train_ratio=0.75,
        batch_size=batch_size
    )

    ckpt = ModelCheckpoint(CKPT_DIR, filename="gagnn")
    lr_monitor = LearningRateMonitor("epoch")
    logger = TensorBoardLogger(LOGDIR, name="gagnn", version="time_embed_v2", default_hp_metric=False)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=n_epochs,
        callbacks=[ckpt],
        accelerator="gpu",
        log_every_n_steps=8
    )

    trainer.fit(model, dtm)


def test():
    dts = AirQualityDataset(
        "data-full",
        # normalize_mean={"humidity":0, "temperature": 0, "PM2.5": 0},
        # normalize_std={"humidity": 1, "temperature": 1, "PM2.5": 1},
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        # split_stations=(12, 3),
        data_set="train"
    )

    model.load_from_checkpoint("ckpt/gagnn-v2.ckpt")

    dt = dts[0]

    probs = torch.softmax(model.encoder.weights, dim=-1)
    print(probs)
    station_groups = probs.max(-1).indices.cpu().tolist()
    print(station_groups)

    src_stations = [
        "Tran Quoc Toan",
        "Quan Hoa",
        "DHQG Ha Noi",
        "Hang Trong",
        "Ngoc Khanh",
        "Lomonoxop",
        "GENESIS",
        "Yen So",
        "Ba Trieu",
        "Tran Quang Khai",
        "Ton That Thuyet"
    ]

    offset_x = -0
    offset_y = 0

    for i in range(11):
        plt.text(dt["src_locs"][i, 0].item() + offset_x, dt["src_locs"][i, 1].item() + offset_y, s=src_stations[i], fontsize=10)
    plt.scatter(dt["src_locs"][:, 0], dt["src_locs"][:, 1], c=station_groups, cmap="spring", linewidths=3)
    plt.show()


    out = model.predict(dts[0])

    print(dts[0]["src_nexts"] - out)
    # print(out)

	# print(dts[0]["gt_target"])

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train(20, 16)
    # test()