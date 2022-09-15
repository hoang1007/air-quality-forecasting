# from typing import Dict
# from torch import nn
# import torch.nn.functional as F
# from .layers import *
# from utils.functional import inverse_distance_weighted
# from models import BaseAQFModel


# class DAQFFModel(BaseAQFModel):
#     """
#     Args:
#         x: Tensor (batch_size, n_stations, seq_len, n_features)
#         locs: Tensor (batch_size, n_stations, 2)
#         masks: Tensor (batch_size, n_stations)
#     Returns:
#         outputs: Tensor (batch_size, n_stations, output_len)
#     """
#     def __init__(self, config, target_normalize_mean: float, target_normalize_std: float):
#         super().__init__(config, target_normalize_mean, target_normalize_std)

#         self.ff_size = config["ff_size"]
#         self.n_features = config["n_features"]
#         # self.time_embedding_dim = 16
#         # self.loc_embedding_dim = 12
#         # self.n_features = config["n_features"] + 3 * self.time_embedding_dim + self.loc_embedding_dim

#         # self.loc_embedding = nn.Linear(2, self.loc_embedding_dim)
#         # self.hour_embedding = nn.Embedding(24, self.time_embedding_dim)
#         # self.solar_term_embedding = nn.Embedding(24, self.time_embedding_dim)
#         # self.weekday_embedding = nn.Embedding(31, self.time_embedding_dim)

#         # self.loc_norm = nn.BatchNorm1d(self.loc_embedding_dim)

#         self.extractor = Conv1dExtractor(config, self.n_features)
#         self.st_layer = SpatialTemporalLayer(config, config["n_stations"])

#         self.linear1 = nn.Linear(config["lstm_output_size"], config["n_stations"])
#         self.linear2 = nn.Linear(1, config["output_size"])

#     def forward(
#         self,
#         air: torch.Tensor,
#         meteo: torch.Tensor,
#         air_locs: torch.Tensor,

#     ):
#         batch_size, n_stations = x.size(0), x.size(1)

#         # time_embed.shape == (batch_size, seq_len, n_timefeats)
#         time_embed = self.time_embedding(time)
#         time_embed = time_embed.unsqueeze(1).repeat_interleave(n_stations, 1)

#         x = torch.cat((x, time_embed), dim=-1).float()

#         # src_locs = (src_locs - src_locs.mean(1, keepdim=True)) / src_locs.std(1, keepdim=True)
#         loc_embed = self.loc_embedding(src_locs).view(-1, self.loc_embedding_dim)
#         loc_embed = self.loc_norm(loc_embed)
#         loc_embed = loc_embed.view(batch_size, n_stations, -1).unsqueeze(2).repeat_interleave(x.size(2), 2)

#         x = torch.cat((x, loc_embed), dim=-1)
#         # features.shape == (batch_size, extractor_size, n_stations)
#         features = self.extractor(x)

#         features = self.st_layer(features)

#         outs = self.linear1(features).unsqueeze(-1)
#         # (batch_size, 11, 24)
#         outs = self.linear2(outs)

#         return outs * self.target_normalize_std + self.target_normalize_mean

#     # def time_embedding(self, time: Dict[str, torch.Tensor]):
#     #     # shape (batch_size, seq_len)
#     #     hour_embed = self.hour_embedding(time["hour"])
#     #     weekday_embed = self.weekday_embedding(time["weekday"] - 1)
#     #     # month_embed = self.month_embedding(time["month"] - 1)
#     #     solar_term_embed = self.solar_term_embedding(time["solar_term"])

#     #     time_embed = torch.cat((hour_embed, weekday_embed, solar_term_embed), dim=-1)

#     #     return time_embed

#     def compute_loss(self, input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
#         loss = F.l1_loss(input, target, reduction=reduction)

#         return loss

#     def training_step(self, batch, batch_idx):
#         outs = self(batch["metero"], batch["src_locs"], batch["time"])

#         loss = self.compute_loss(outs, batch["src_nexts"].float(), reduction="mean")

#         self.log("loss", loss.item())

#         return loss

#     def validation_step(self, batch, batch_idx):
#         # pres.shape == (batch_size, n_src_stations, output_size)
#         outs = self(batch["metero"], batch["src_locs"], batch["time"])

#         # mae = self.compute_loss(outs, batch["src_nexts"].float(), reduction="sum")
#         # mae = (outs - batch["src_nexts"]).abs().sum(-1)
#         # mae = mae.mean()

#         return self.metric_fns(outs, batch["src_nexts"])


#     def predict(self, dt):
#         st = self.training
#         self.train(False)
#         with torch.no_grad():
#             metero = dt["metero"].to(self.device).unsqueeze(0)
#             src_locs = dt["src_locs"].to(self.device).unsqueeze(0)

#             for k in dt["time"]:
#                 dt["time"][k] = dt["time"][k].to(self.device).unsqueeze(0)
            
#             preds = self(metero, src_locs, dt["time"])
#         self.train(st)

#         return preds