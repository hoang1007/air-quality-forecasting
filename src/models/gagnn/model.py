from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
from .utils import batch_gnn_input, inverse_distance_weighting
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
        time: Dict[str, torch.Tensor]
    ):  
        batch_size = x.size(0)
        n_stations = x.size(1)

        # time_embed.shape == (batch_size, seq_len, n_timefeats)
        time_embed = self.time_embedding(time)

        edge_weights, edge_ids = inverse_distance_weighting(src_locs)

        enc_outs, gr_edge_weights, gr_edge_ids = self.encoder(x, src_locs, time_embed, edge_weights, edge_ids)
        
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

    def time_embedding(self, time: Dict[str, torch.Tensor]):
        # shape (batch_size, seq_len)
        hour_embed = self.hour_embedding(time["hour"][:, 0])
        day_embed = self.day_embedding(time["day"][:, 0] - 1)
        month_embed = self.month_embedding(time["month"][:, 0] - 1)
        solar_term_embed = self.solar_term_embedding(time["solar_term"][:, 0])

        time_embed = torch.cat((hour_embed, day_embed, month_embed, solar_term_embed), dim=-1)

        return time_embed

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor, reduction="mean"):
        return F.l1_loss(input, target, reduction=reduction)

    def training_step(self, batch, batch_idx):
        # outs = self(batch["features"], batch["src_locs"], batch["tar_locs"], None)
        outs = self(batch["metero"], batch["src_locs"], batch["time"])

        # loss = self.compute_loss(outs, batch["target"])
        loss = self.compute_loss(outs, batch["src_nexts"], reduction="mean")

        self.log("loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        # pres.shape == (batch_size, n_src_stations, output_size)
        # outs = self(batch["features"], batch["src_locs"], batch["tar_locs"], None)
        outs = self(batch["metero"], batch["src_locs"], batch["time"])

        return self.metric_fns(outs, batch["src_nexts"])

    def predict(self, dt):
        st = self.training
        self.train(False)
        with torch.no_grad():
            # add batch dim
            metero = dt["metero"].to(self.device).unsqueeze(0)
            src_locs = dt["src_locs"].to(self.device).unsqueeze(0)

            for k in dt["time"]:
                dt["time"][k] = dt["time"][k].to(self.device).unsqueeze(0)

            # output = self(features, src_locs, tar_loc, None).squeeze(0)
            output = self(metero, src_locs, dt["time"]).squeeze(0)

        self.train(st)
        return output


class AQFBaseGAGNN(BaseAQFModel):
    def __init__(self, config, target_mean, target_std):
        super().__init__(config, target_mean, target_std)

        self.num_stations = config["num_stations"]
        self.n_features = config["n_features"]
        self.input_embedding_dim = config["input_embedding_dim"]
        self.loc_embedding_dim = config["loc_embedding_dim"]
        self.time_embedding_dim = config["time_embedding_dim"]
        self.gnn_dim = config["gnn_dim"]
        self.outseq_len = config["outseq_len"]
        self.num_gnn_layers = config["num_enc_gnn_layers"]
        self.edge_dim = config["edge_dim"]
        self.gnn_dim = config["gnn_dim"]
        self.num_groups = config["num_groups"]
        self.dropout = config["dropout"]


        self.hour_embedding = nn.Embedding(24, self.time_embedding_dim)
        self.solar_term_embedding = nn.Embedding(24, self.time_embedding_dim)
        self.day_embedding = nn.Embedding(31, self.time_embedding_dim)
        self.month_embedding = nn.Embedding(12, self.time_embedding_dim)

        self.encoder = GAGNNEncoder(config)
        self.decoder_embedding = nn.Linear(self.gnn_dim, self.input_embedding_dim)
        self.loc_embedding = nn.Linear(2, self.loc_embedding_dim)
        self.loc_norm = nn.BatchNorm1d(self.loc_embedding_dim)

        self.group_gnn = nn.ModuleList([GraphNode(self.input_embedding_dim, self.edge_dim, self.gnn_dim)])
        for _ in range(self.num_gnn_layers - 1):
            self.group_gnn.append(GraphNode(self.gnn_dim, self.edge_dim, self.gnn_dim))

        self.global_gnn = nn.ModuleList([GraphNode(self.input_embedding_dim + self.gnn_dim, 1, self.gnn_dim)])
        for _ in range(self.num_gnn_layers - 1):
            self.global_gnn.append(GraphNode(self.gnn_dim, 1, self.gnn_dim))

        # self.fc1 = nn.Linear(self.gnn_dim + self.loc_embedding_dim, 32)
        # self.fc2 = nn.Linear(32 + self.input_embedding_dim, 16)
        # self.fc3 = nn.Linear(16, self.outseq_len)
        self.fc = nn.Sequential(
            nn.Linear(self.gnn_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, self.outseq_len)
        )

    def forward(
        self,
        x: torch.Tensor,
        src_locs: torch.Tensor,
        time: Dict[str, torch.Tensor],
        time_next: Dict[str, torch.Tensor],
        tar_locs: torch.Tensor = None,
    ):
        batch_size = x.size(0)
        n_stations = x.size(1)

        time_embed = self.time_embedding(time)
        # time_embed = time_embed.unsqueeze(dim=1).repeat_interleave(n_stations, dim=1)
        # x = torch.cat((x.float(), time_embed), dim=-1)

        edge_weights, edge_ids = inverse_distance_weighting(src_locs)

        # enc_outs.shape == (batch_size, num_groups, gnn_dim)
        enc_outs, gr_edge_weights, gr_edge_ids = self.encoder(x, src_locs, time_embed, edge_weights, edge_ids)

        # DECODING
        dec_in = self.decoder_embedding(enc_outs)

        w1 = torch.softmax(self.encoder.weights, dim=-1).t()
        w1 = w1.unsqueeze(dim=0).repeat_interleave(batch_size, dim=0)

        gr_feats = torch.bmm(w1, dec_in).view(-1, self.input_embedding_dim)
        
        for layer in self.group_gnn:
            gr_feats = layer(gr_feats, gr_edge_ids, gr_edge_weights)

        gr_feats = gr_feats.view(batch_size, -1, self.gnn_dim)

        # outs.shape == (batch_size, num_stations, gnn_dim)
        w2 = w1.transpose(1, 2)
        outs = torch.bmm(w2, gr_feats)

        outs = torch.cat((dec_in, outs), dim=-1)
        outs, edge_weights, edge_ids = batch_gnn_input(outs, edge_weights.unsqueeze(-1), edge_ids)

        for layer in self.global_gnn:
            outs = layer(outs, edge_ids, edge_weights)

        preds = self.fc(outs).view(batch_size, n_stations, self.outseq_len)

        # outs = outs.unsqueeze(dim=2).repeat_interleave(self.outseq_len, dim=2)
        # dec_time = self.time_embedding(time_next)
        # dec_time = dec_time.unsqueeze(dim=1).repeat_interleave(n_stations, dim=1)

        # outs = torch.cat((outs, dec_time), dim=-1)
        # loc_embed = F.relu(self.loc_embedding(src_locs))
        # loc_embed = self.loc_norm(loc_embed.view(-1, self.loc_embedding_dim))
        # loc_embed = loc_embed.view(batch_size, -1, self.loc_embedding_dim)

        
        # outs = torch.cat((outs, loc_embed), dim=-1)
        # outs = F.relu(self.fc1(outs))

        # outs = torch.cat((outs, dec_in), dim=-1)
        # outs = F.relu(self.fc2(outs))

        # preds = self.fc3(outs)

        return preds * self.target_normalize_std + self.target_normalize_mean

    def time_embedding(self, time: Dict[str, torch.Tensor]):
        # shape (batch_size, seq_len)
        hour_embed = self.hour_embedding(time["hour"][:, 0])
        day_embed = self.day_embedding(time["day"][:, 0] - 1)
        month_embed = self.month_embedding(time["month"][:, 0] - 1)
        solar_term_embed = self.solar_term_embedding(time["solar_term"][:, 0])

        time_embed = torch.cat((hour_embed, day_embed, month_embed, solar_term_embed), dim=-1)

        return time_embed

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor):
        return F.mse_loss(input, target)

    def training_step(self, batch, batch_idx):
        outs = self(
            batch["metero"],
            batch["src_locs"],
            batch["time"],
            None,
            None
        )

        loss = self.compute_loss(outs, batch["src_nexts"].float())

        self.log("loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(
            batch["metero"],
            batch["src_locs"],
            batch["time"],
            None,
            None
        )

        return self.metric_fns(outs, batch["src_nexts"])

    def predict(self, dt):
        st = self.training
        self.train(False)

        with torch.no_grad():
            metero = dt["metero"].to(self.device).unsqueeze(0)
            src_locs = dt["src_locs"].to(self.device).unsqueeze(0)

            for k in dt["time"]:
                dt["time"][k] = dt["time"][k].to(self.device).unsqueeze(0)

            # for k in dt["time_next"]:
            #     dt["time_next"][k] = dt["time_next"][k].to(self.device).unsqueeze(0)

            preds = self(metero, src_locs, dt["time"], None, None)

        self.train(st)
        return preds