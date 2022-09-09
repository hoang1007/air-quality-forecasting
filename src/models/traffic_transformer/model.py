from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
from models import BaseAQFModel
from .layers import (
    TrafficTransformerEncoder,
    TrafficTransformerDecoder,
    positional_encoding
)


class TrafficTransformer(BaseAQFModel):
    def __init__(self, config, target_normalize_mean: float, target_normalize_std: float):
        """
        Args:
            metero: Tensor (batch_size, num_stations, inseq_len, n_features)
            locs: Tensor (batch_size, num_stations, 2)
            time: Dict[str, Tensor] [weekday, timestamp] for both encode and decode phase
            metero_next: Tensor (batch_size, num_stations, outseq_len, n_features)

        Returns:
            predicted: Tensor (batch_size, num_stations, n_features)
        """
        super().__init__(config, target_normalize_mean, target_normalize_std)
        self.is_predict = True
        self.inseq_len = config["inseq_len"]
        self.outseq_len = config["outseq_len"]
        self.pos_embedding_dim = config["pos_embedding_dim"]

        self.encoder = TrafficTransformerEncoder(config)
        self.decoder = TrafficTransformerDecoder(config)

        self.fc = nn.Linear(config["gcn_dim"] + config["n_features"], config["n_features"])

        # self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.uniform_(param.data)

    def forward(
        self,
        meteo: torch.Tensor,
        locs: torch.Tensor,
        time: Dict[str, torch.Tensor],
        targets: torch.Tensor = None
    ):
        pos_embed, periodic_embed = self.hybrid_pos_embedding(time)
        enc_pos_em = pos_embed[:, :self.inseq_len]
        enc_periodic_em = periodic_embed[:, :self.inseq_len]

        enc_out = self.encoder(meteo, locs, enc_pos_em, enc_periodic_em)

        shifted_targets = torch.cat((meteo[:, :, -1].unsqueeze(2), targets[:, :, :-1]), dim=2)
        if self.training:
            dec_start_pos = self.inseq_len - 1
            dec_end_pos = dec_start_pos + self.outseq_len

            dec_pos_em = pos_embed[:, dec_start_pos : dec_end_pos]
            dec_periodic_em = periodic_embed[:, dec_start_pos : dec_end_pos]
            outputs = self.decoder(shifted_targets, enc_out, locs, dec_pos_em, enc_periodic_em, dec_periodic_em)
            predicted = self.fc(outputs)

            return predicted
        else:
            # eval phase
            if self.is_predict:
                dec_in = meteo[:, :, -1].unsqueeze(2)
                dec_start_pos = self.inseq_len - 1
                for i in range(1, self.outseq_len + 1):
                    dec_pos_em = pos_embed[:, dec_start_pos : dec_start_pos + i]
                    dec_periodic_em = periodic_embed[:, dec_start_pos : dec_start_pos + i]
                    dec_out = self.decoder(dec_in, enc_out, locs, dec_pos_em, enc_periodic_em, dec_periodic_em)[:, :, -1]
                    dec_out = self.fc(dec_out).unsqueeze(2)

                    dec_in = torch.cat((dec_in, dec_out), dim=2)
                
                return dec_in[:, :, 1:]
            else:
                dec_start_pos = self.inseq_len - 1
                dec_end_pos = dec_start_pos + self.outseq_len

                dec_pos_em = pos_embed[:, dec_start_pos : dec_end_pos]
                dec_periodic_em = periodic_embed[:, dec_start_pos : dec_end_pos]
                outputs = self.decoder(shifted_targets, enc_out, locs, dec_pos_em, enc_periodic_em, dec_periodic_em)
                predicted = self.fc(outputs)

                return predicted


    def hybrid_pos_embedding(
        self,
        time: Dict[str, torch.Tensor]
    ):
        weekly_ids = time["weekday"]
        hour = time["hour"]
        global_ids = time["timestamp"]

        assert weekly_ids.size(1) == global_ids.size(1), "positional indices must be the same size"
        relative_ids = torch.arange(weekly_ids.size(1), device=self.device)
        relative_ids = relative_ids.unsqueeze(0).repeat_interleave(weekly_ids.size(0), 0)

        weekly_embed = positional_encoding(weekly_ids, 8)
        hour_embed = positional_encoding(hour, 24)
        global_embed = positional_encoding(global_ids, self.pos_embedding_dim)
        relative_embed = positional_encoding(relative_ids, self.pos_embedding_dim)

        pos_embed = relative_embed + global_embed
        periodic_embed = torch.cat((weekly_embed, hour_embed), dim=-1)

        return pos_embed, periodic_embed

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
        return F.l1_loss(input, target, reduction=reduction)

    def training_step(self, batch, batch_idx):
        outs = self(batch["metero"], batch["src_locs"], batch["time"], batch["metero_next"])

        loss = self.compute_loss(outs, batch["metero_next"], reduction="mean")
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch["metero"], batch["src_locs"], batch["time"], batch["metero_next"])
        preds = outs[..., -1] * self.target_normalize_std + self.target_normalize_mean
        targets = batch["metero_next"][..., -1] * self.target_normalize_std + self.target_normalize_mean
        
        return self.metric_fns(preds, targets)

    def predict(self, dt):
        st = self.training
        self.eval()
        with torch.no_grad():
            dt["metero"] = dt["metero"].to(self.device).unsqueeze(0)
            dt["src_locs"] = dt["src_locs"].to(self.device).unsqueeze(0)
            dt["metero_next"] = dt["metero_next"].to(self.device).unsqueeze(0)

            for k in dt["time"]:
                dt["time"][k] = dt["time"][k].to(self.device).unsqueeze(0)

            outs = self(dt["metero"], dt["src_locs"], dt["time"], dt["metero_next"])
            preds = outs[..., -1] * self.target_normalize_std + self.target_normalize_mean
            targets = dt["metero_next"][..., -1] * self.target_normalize_std + self.target_normalize_mean
        self.train(st)
        return preds.squeeze(0), targets.squeeze(0)