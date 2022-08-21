import torch
from torch import nn
import torch.nn.functional as F
from .utils import inverse_distance_weighting
from .layers import GAGNNDecoder, GAGNNEncoder, GraphNode
from models import BaseAQFModel


class GAGNNModel(BaseAQFModel):
    def __init__(self, config, target_normalize_mean: float, target_normalize_std: float):
        """
        Args:
            inputs: Tensor (batch_size, n_stations, inseq_len, n_features)
            src_locs: Tensor (batch_size, n_stations, 2)
            tar_loc: Tensor (batch_size, 2)
            time: Tensor (batch_size, inseq_len, 4) (hour, day, month, solar_term)
        """
        super().__init__(config, target_normalize_mean, target_normalize_std)

        self.time_embedding_dim = config["time_embedding_dim"]
        self.gnn_dim = config["gnn_dim"]
        self.outseq_len = config["outseq_len"]


        self.hour_embedding = nn.Embedding(24, self.time_embedding_dim)
        self.solar_term_embedding = nn.Embedding(24, self.time_embedding_dim)
        self.day_embedding = nn.Embedding(31, self.time_embedding_dim)
        self.month_embedding = nn.Embedding(12, self.time_embedding_dim)

        self.encoder = GAGNNEncoder(config)
        self.decoder = GAGNNDecoder(config)

        self.predictor = nn.Sequential(
            nn.Linear(self.gnn_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, self.outseq_len),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_locs: torch.Tensor,
        tar_locs: torch.Tensor,
        time: torch.Tensor
    ):  
        # loc_mean_ = x.new_tensor([105.8010,  21.0294])
        # loc_std_ = x.new_tensor([0.0553, 0.0131])
        
        # src_locs = (src_locs - loc_mean_) / loc_std_
        # tar_locs = (tar_locs - loc_mean_) / loc_std_

        batch_size = x.size(0)
        n_stations = x.size(1)

        # time.shape == (batch_size, n_stations, seq_len, n_timefeats)
        time = time.reshape(batch_size * n_stations, -1, time.size(-1))
        hour_embed = self.hour_embedding(time[..., 0]).view(batch_size, n_stations, -1, self.time_embedding_dim)
        day_embed = self.day_embedding(time[..., 1] - 1).view(batch_size, n_stations, -1, self.time_embedding_dim)
        month_embed = self.month_embedding(time[..., 2] - 1).view(batch_size, n_stations, -1, self.time_embedding_dim)
        solar_term_embed = self.solar_term_embedding(time[..., 3]).view(batch_size, n_stations, -1, self.time_embedding_dim)

        x = torch.cat((x, hour_embed, day_embed, month_embed, solar_term_embed), dim=-1).float()

        edge_weights, edge_ids = inverse_distance_weighting(src_locs)

        enc_outs, gr_edge_weights, gr_edge_ids = self.encoder(x, src_locs, edge_weights, edge_ids)
        
        # dec_outs.shape == (batch_size, n_stations, gnn_dim)
        dec_outs = self.decoder(
            enc_outs,
            n_stations,
            self.encoder.weights,
            self.encoder.weights,
            gr_edge_ids,
            gr_edge_weights,
            edge_ids,
            edge_weights
        )

        preds = self.predictor(dec_outs).view(batch_size, -1, self.outseq_len)

        return preds * self.target_normalize_std + self.target_normalize_mean

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor, reduction="mean"):
        return F.l1_loss(input, target, reduction=reduction)

    def training_step(self, batch, batch_idx):
        # outs = self(batch["features"], batch["src_locs"], batch["tar_locs"], None)
        outs = self(batch["features"], batch["src_locs"], None, batch["time"])

        # loss = self.compute_loss(outs, batch["target"])
        loss = self.compute_loss(outs, batch["src_nexts"], reduction="mean")

        self.log("loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        # pres.shape == (batch_size, n_src_stations, output_size)
        # outs = self(batch["features"], batch["src_locs"], batch["tar_locs"], None)
        outs = self(batch["features"], batch["src_locs"], None, batch["time"])

        # mae = self.compute_loss(outs, batch["gt_target"])
        mae = self.compute_loss(outs, batch["src_nexts"], reduction="sum")

        return {"mae": mae}
        # return self.metric_fns(outs.squeeze(), batch["gt_target"].squeeze())

    def predict(self, dt):
        st = self.training
        self.train(False)
        with torch.no_grad():
            # add batch dim
            features = dt["features"].unsqueeze(0).to(self.device)
            src_locs = dt["src_locs"].unsqueeze(0).to(self.device)
            # tar_loc = dt["tar_locs"].unsqueeze(0).to(self.device)
            time = dt["time"].unsqueeze(0).to(self.device)

            # output = self(features, src_locs, tar_loc, None).squeeze(0)
            output = self(features, src_locs, None, time).squeeze(0)
            # inverse transforms
            # output = output * self.target_normalize_std + self.target_normalize_mean

        self.train(st)
        return output