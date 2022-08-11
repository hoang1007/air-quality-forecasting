from torch import nn
import torch.nn.functional as F
from .layers import *
from utils.functional import inverse_distance_weighted
from model import BaseAQFModel


class DAQFFModel(BaseAQFModel):
    """
    Args:
        x: Tensor (batch_size, n_stations, seq_len, n_features)
        locs: Tensor (batch_size, n_stations, 2)
        masks: Tensor (batch_size, n_stations)
    Returns:
        outputs: Tensor (batch_size, n_stations, output_len)
    """
    def __init__(self, config, target_normalize_mean: float, target_normalize_std: float):
        super().__init__(config, target_normalize_mean, target_normalize_std)

        self.extractor = Conv1dExtractor(config, config["n_features"]) # spatial feat?
        self.st_layer = SpatialTemporalLayer(config, config["n_stations"])

        self.linear1 = nn.Linear(config["lstm_output_size"], config["n_stations"])
        self.linear2 = nn.Linear(1, config["output_size"])

    def forward(
        self,
        x: torch.Tensor,
        locs: torch.Tensor
    ): 
        # features.shape == (batch_size, extractor_size, n_stations)
        features = self.extractor(x)

        # corr_weights.shape == (batch_size, n_stations, n_stations)
        corr_weigts = inverse_distance_weighted(locs, locs, beta=1.0)
        features = torch.bmm(features, corr_weigts)

        features = self.st_layer(features)

        out = self.linear1(features).unsqueeze(-1)
        out = self.linear2(out)

        return out

    def compute_distances(self, src_locs: torch.Tensor, tar_locs: torch.Tensor):
        tar_length = tar_locs.size(1)
        # expand src_locs to shape (batch_size, src_length, tar_length, 2)
        src_locs = src_locs.unsqueeze(-2)
        new_shape = list(src_locs.shape)
        new_shape[-2] = tar_length
        src_locs = src_locs.expand(new_shape)

        # tar_locs.shape == (batch_size, 1, tar_length, 2)
        tar_locs = tar_locs.unsqueeze(1)

        # dists.shape == (batch_size, src_length, tar_length)
        dists = (src_locs - tar_locs).pow(2).sum(dim=-1).sqrt()

        return dists

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor, masks: torch.Tensor):
        # shape == (batch_size, n_stations, output_size)
        return F.mse_loss(input[:, masks], target[:, masks])

    def training_step(self, batch, batch_idx):
        preds = self(batch["features"], batch["src_locs"])

        loss = self.compute_loss(preds, batch["src_nexts"])

        self.log("loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch["features"], batch["src_locs"])

        preds = preds * self.target_normalize_std + self.target_normalize_mean

        mae = (preds - batch["src_nexts"]).abs().mean()

        return {"mae": mae}