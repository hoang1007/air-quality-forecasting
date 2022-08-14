# %%
from dataset import AirQualityDataset2, AirQualityDataModule
from hydra import initialize, compose
import os
import torch
from torch import nn
import torch.nn.functional as F

with initialize(version_base=None, config_path="../config"):
  cfg = compose(config_name="daqff.yaml")

# %%
dtm = AirQualityDataModule(
        "data",
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

# %%
from utils.functional import scale_softmax
from tqdm import tqdm


class SpatialModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.linear1 = nn.Linear(2, 3)
        w = torch.zeros(11, 1)
        nn.init.xavier_uniform_(w)

        self.linear = nn.parameter.Parameter(w)
    
    def forward(
        self,
        srcs: torch.Tensor,
        src_locs: torch.Tensor,
        tar_loc: torch.Tensor,
        src_masks: torch.Tensor
    ):
        assert srcs.size(0) == 1
        # srcs.shape == (src_len, 24)
        srcs = srcs.squeeze(0)

        # mean_ = src_locs.mean(-1)
        # std_ = src_locs.std(-1)

        # # norm
        # src_locs = (src_locs - mean_) / std_
        # tar_loc = (tar_loc - mean_) / std_



        weights = self.linear
        weights = scale_softmax(weights, dim=0)
        preds = (srcs * weights).sum(0)

        return preds
    
    def compute_loss(self, input: torch.Tensor, target: torch.Tensor):
        assert input.isnan().sum() == 0
        assert target.isnan().sum() == 0
        loss = torch.pow((input - target), 2).mean().sqrt()

        if loss.isnan().sum() > 0:
            print(loss)
        return loss

    def fit(self, train_dataloader, val_dataloader, n_epochs: int):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        criterion = self.compute_loss

        tar_idx = 2
        metrics = []
        loss_log = []

        for _ in range(n_epochs):
            with tqdm(train_dataloader) as bar:
                rloss = []
                for batch_idx, batch in enumerate(bar):
                    if batch["target_idx"][0] == tar_idx:
                        output = self(batch["src_nexts"], batch["src_locs"], batch["tar_loc"], batch["src_masks"])

                        loss = criterion(output, batch["target"].squeeze(0))

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        rloss.append(loss.item())

                    if len(rloss) == 100:
                        avgl = sum(rloss) / len(rloss)
                        loss_log.append(avgl)
                        bar.set_postfix(loss=avgl)
                        rloss.clear()
                scheduler.step()

            with tqdm(val_dataloader, leave=False) as bar:
                self.eval()
                with torch.no_grad():
                    rmetr = []
                    for batch_idx, batch in enumerate(bar):
                        if batch["target_idx"][0] == tar_idx:
                            output = self(batch["src_nexts"], batch["src_locs"], batch["tar_loc"], batch["src_masks"])

                            output = output * cfg.data.normalize_std["PM2.5"] + cfg.data.normalize_mean["PM2.5"]
                            mae = (output - batch["gt_target"]).abs().mean().item()

                            rmetr.append(mae)
                    metrics.append(sum(rmetr) / len(rmetr))
                    rmetr.clear()

                self.train()

        return loss_log, metrics

if __name__ == "__main__":
    # %%
    torch.manual_seed(3107)
    # torch.autograd.set_detect_anomaly(True)
    model = SpatialModel()

    # %%
    loss_log, metric_log = model.fit(dtm.train_dataloader(), dtm.val_dataloader(), n_epochs=6)

    print(model.linear.data)

    # %%
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2)
    ax[0].plot(loss_log)
    ax[1].plot(metric_log)
    fig.show()
    plt.show()