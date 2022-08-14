import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.functional import scale_softmax
from tqdm import tqdm


class SpatialCorrelation(nn.Module):
    def __init__(
        self,
        n_src_stations: int,
        n_tar_stations: int,
        target_mean: float,
        target_std: float
        # src_locs: torch.Tensor,
        # tar_loc: torch.Tensor
    ):
        super().__init__()

        self.target_mean = target_mean
        self.target_std = target_std

        weights = torch.zeros(n_tar_stations, n_src_stations, 1)
        nn.init.xavier_uniform_(weights)

        self.weights = nn.parameter.Parameter(weights)
        # self.src_locs = src_locs
        # self.tar_loc = tar_loc

        # self.loc_mean_ = src_locs.mean(-1)
        # self.loc_std_ = src_locs.std(-1)

    def forward(
        self,
        srcs: torch.Tensor,
        tar_idx: int
    ):
        assert isinstance(tar_idx, int)
        assert srcs.size(0) == 1, "Only support batch_size = 1"
        # srcs.shape == (n_src, seq_len)
        srcs = srcs.squeeze(0)

        weights = scale_softmax(self.weights[tar_idx], dim=0)

        out = (srcs * weights).sum(0, keepdim=True)

        return out

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor):
        loss = (input - target).pow(2).mean().sqrt()

        return loss

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        n_epochs: int = 5,
        step_log: int = 20,
        device: str = "cpu",
        return_log: bool = False
    ):
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        criterion = self._compute_loss

        train_log = []
        eval_log = []

        print("Finding spatial correlation")
        for epoch in range(1, n_epochs + 1):
            with tqdm(train_dataloader) as train_bar:
                train_bar.set_description(f"Epoch {epoch}")
                rloss = []
                for batch_idx, batch in enumerate(train_bar):
                    src_nexts = batch["src_nexts"].to(device)
                    target_idx = batch["target_idx"].item()
                    target = batch["target"].to(device)

                    pred = self(src_nexts, target_idx)

                    loss = criterion(pred, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    rloss.append(loss.item())

                    if len(rloss) == step_log or batch_idx + 1 == len(train_dataloader):
                        avg_loss = sum(rloss) / len(rloss)
                        train_log.append(avg_loss)
                        train_bar.set_postfix(loss=avg_loss)

                scheduler.step()

                with tqdm(val_dataloader, leave=False) as val_bar:
                    reval = []
                    self.eval()
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(val_bar):
                            src_nexts = batch["src_nexts"].to(device)
                            target_idx = batch["target_idx"].item()
                            target = batch["gt_target"].to(device)

                            if target_idx != 0:
                                continue

                            pred = self(src_nexts, target_idx) * self.target_std + self.target_mean

                            loss = criterion(pred, target)

                            reval.append(loss.item())
                    self.train()
                    avg = sum(reval) / len(reval)
                    eval_log.append(avg)

        if return_log:
            return train_log, eval_log

    def get_correlation(self):
        return self.weights.detach().squeeze(-1)

    def predict(self, srcs, tar_idx):
        out = self(srcs, tar_idx).squeeze(0)

        return out * self.target_std + self.target_mean