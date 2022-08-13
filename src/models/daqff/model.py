from torch import nn
import torch.nn.functional as F
from .layers import *
from utils.functional import inverse_distance_weighted, scale_softmax
from models import BaseAQFModel


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

        self.ff_size = config["ff_size"]

        self.extractor = Conv1dExtractor(config, config["n_features"])
        self.st_layer = SpatialTemporalLayer(config, self.ff_size)

        self.linear1 = nn.Linear(config["lstm_output_size"], config["n_stations"])
        self.linear2 = nn.Linear(1, config["output_size"])

        self.register_parameter("linear3", self._init_feedforward(config["n_stations"], self.ff_size))
    
    def _init_feedforward(self, input_size, output_size):
        weights = torch.zeros((input_size, output_size))
        nn.init.xavier_uniform_(weights)

        return nn.parameter.Parameter(weights, requires_grad=True)

    def forward(
        self,
        x: torch.Tensor,
        src_locs: torch.Tensor,
        masks: torch.Tensor
    ):

        # features.shape == (batch_size, extractor_size, n_stations)
        features = self.extractor(x)

        batch_size = x.size(0)
        
        _feats = x.new_zeros(batch_size, features.size(1), self.ff_size)
        for i in range(batch_size):
            masks_ = masks[i]
            _feats[i] = torch.matmul(features[i, :, masks_], self.linear3[masks_])
        features = _feats

        # corr_weights.shape == (batch_size, n_stations, n_stations)
        # corr_weigts = inverse_distance_weighted(locs, locs, beta=1.0)
        # corr_weigts = torch.softmax(corr_weigts, dim=-1)
        # features = torch.bmm(features, corr_weigts)

        features = self.st_layer(features)

        outs = self.linear1(features).unsqueeze(-1)
        # (batch_size, 11, 24)
        outs = self.linear2(outs)

        return outs

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor):
        loss = F.mse_loss(input, target, reduction="none")

        loss = loss.mean(-1).sqrt()

        loss = loss.sum() / loss.size(0)

        return loss

    def training_step(self, batch, batch_idx):
        outs = self(batch["features"], batch["src_locs"], batch["src_masks"])

        loss = self.compute_loss(outs, batch["src_nexts"])

        self.log("loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        # pres.shape == (batch_size, n_src_stations, output_size)
        outs = self(batch["features"], batch["src_locs"], batch["src_masks"])

        outs = outs * self.target_normalize_std + self.target_normalize_mean
        # src_outs = src_outs * self.target_normalize_std + self.target_normalize_mean

        # batch_size = src_outs.size(0)
        # preds = src_outs.new_zeros(batch_size, src_outs.size(2))

        # for i in range(batch_size):
        #     masks = batch["src_masks"][i]
        #     src_locs = batch["src_locs"][i, masks].unsqueeze(0)
        #     tar_loc = batch["tar_loc"][i].unsqueeze(0)

        #     weights = inverse_distance_weighted(src_locs, tar_loc, beta=1.0).squeeze(0)
        #     preds[i] = (src_outs[i, masks] * weights).sum(0)

        # return self.metrics(tar_outs, batch["gt_target"])
        mae = (outs - batch["src_nexts"]).abs().mean()

        return {"mae": mae}


    def predict(self, dt):
        st = self.training
        self.train(False)
        with torch.no_grad():
            # add batch dim
            features = dt["features"].unsqueeze(0).to(self.device)
            src_locs = dt["src_locs"].unsqueeze(0).to(self.device)
            # tar_loc = tar_loc.unsqueeze(0).to(self.device)
            src_masks = dt["src_masks"].unsqueeze(0).to(self.device)

            output = self(features, src_locs, src_masks).squeeze(0)
            # inverse transforms
            # output.shape == (n_src_stations, output_size)
            output = output * self.target_normalize_std + self.target_normalize_mean

            # weights.shape == (n_src_st', 1)
            # weights = inverse_distance_weighted(src_locs[:, src_masks[0]], tar_loc, beta=1.0).squeeze(0)

            # output = (output[src_masks[0]] * weights).sum(0)
        self.train(st)
        return output