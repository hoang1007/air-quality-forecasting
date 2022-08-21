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
        self.time_embedding_dim = 16
        self.loc_embedding_dim = 12
        self.n_features = config["n_features"] + 4 * self.time_embedding_dim + self.loc_embedding_dim

        self.loc_embedding = nn.Linear(2, self.loc_embedding_dim)
        self.hour_embedding = nn.Embedding(24, self.time_embedding_dim)
        self.solar_term_embedding = nn.Embedding(24, self.time_embedding_dim)
        self.day_embedding = nn.Embedding(31, self.time_embedding_dim)
        self.month_embedding = nn.Embedding(12, self.time_embedding_dim)

        self.extractor = Conv1dExtractor(config, self.n_features)
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
        masks: torch.Tensor,
        time: torch.Tensor,
    ):  
        batch_size, n_stations = x.size(0), x.size(1)
        # time_mean_ = x.new_tensor([0, 1, 1, 0])
        # time_std_ = x.new_tensor([24, 31, 12, 24])
        # time = (time - time_mean_) / time_std_

        # x = torch.cat((x, time), dim=-1).float()

        time = time.reshape(batch_size * n_stations, -1, time.size(-1))
        hour_embed = self.hour_embedding(time[..., 0]).view(batch_size, n_stations, -1, self.time_embedding_dim)
        day_embed = self.day_embedding(time[..., 1] - 1).view(batch_size, n_stations, -1, self.time_embedding_dim)
        month_embed = self.month_embedding(time[..., 2] - 1).view(batch_size, n_stations, -1, self.time_embedding_dim)
        solar_term_embed = self.solar_term_embedding(time[..., 3]).view(batch_size, n_stations, -1, self.time_embedding_dim)

        x = torch.cat((x, hour_embed, day_embed, month_embed, solar_term_embed), dim=-1).float()

        src_locs = (src_locs - src_locs.mean(1, keepdim=True)) / src_locs.std(1, keepdim=True)
        loc_embed = self.loc_embedding(src_locs).unsqueeze(2).repeat_interleave(x.size(2), 2)

        x = torch.cat((x, loc_embed), dim=-1)
        # features.shape == (batch_size, extractor_size, n_stations)
        features = self.extractor(x)
        
        _feats = x.new_zeros(batch_size, features.size(1), self.ff_size)
        for i in range(batch_size):
            masks_ = masks[i]
            _feats[i] = torch.matmul(features[i, :, masks_], self.linear3[masks_])
        features = _feats

        features = self.st_layer(features)

        outs = self.linear1(features).unsqueeze(-1)
        # (batch_size, 11, 24)
        outs = self.linear2(outs)

        return outs * self.target_normalize_std + self.target_normalize_mean

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
        loss = F.l1_loss(input, target, reduction=reduction)

        # loss = loss.mean(-1).sqrt()

        # loss = loss.sum() / loss.size(0)

        return loss

    def training_step(self, batch, batch_idx):
        outs = self(batch["features"], batch["src_locs"], batch["src_masks"], batch["time"])

        loss = self.compute_loss(outs, batch["src_nexts"].float(), reduction="mean")

        self.log("loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        # pres.shape == (batch_size, n_src_stations, output_size)
        outs = self(batch["features"], batch["src_locs"], batch["src_masks"], batch["time"])

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
        # mae = (outs - batch["src_nexts"].float()).abs().mean()
        mae = self.compute_loss(outs, batch["src_nexts"].float(), reduction="sum")

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