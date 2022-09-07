from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
from models import BaseAQFModel
from .layers import *


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

        self.inseq_len = config["inseq_len"]
        self.outseq_len = config["outseq_len"]

        self.pos_embedding = PositionalEmbedding(config["pos_embedding_dim"])
        self.encoder = TrafficTransformerEncoder(config)
        self.decoder = TrafficTransformerDecoder(config)

        self.fc = nn.Linear(config["gcn_dim"], config["n_features"])

        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.uniform_(param.data)

    def forward(
        self,
        metero: torch.Tensor,
        locs: torch.Tensor,
        time: Dict[str, torch.Tensor],
        metero_next: torch.Tensor = None
    ):
        hpe = self.hybrid_pos_embedding(time)
        enc_pos_em = hpe[:, :self.inseq_len]

        enc_out = self.encoder(metero, locs, enc_pos_em)

        if self.training:
            dec_start_pos = self.inseq_len - 1
            dec_end_pos = dec_start_pos + self.outseq_len

            dec_pos_em = hpe[:, dec_start_pos : dec_end_pos]
            outputs = self.decoder(metero_next, enc_out, locs, enc_pos_em, dec_pos_em)
            predicted = self.fc(outputs)

            return predicted
        else:
            # eval phase
            # dec_in = metero[:, :, -1].unsqueeze(2)
            # dec_start_pos = self.inseq_len - 1
            # for i in range(1, self.outseq_len + 1):
            #     dec_pos_em = hpe[:, dec_start_pos : dec_start_pos + i]
            #     dec_out = self.decoder(dec_in, enc_out, locs, enc_pos_em, dec_pos_em)[:, :, -1]
            #     dec_out = self.fc(dec_out).unsqueeze(2)

            #     dec_in = torch.cat((dec_in, dec_out), dim=2)
            
            # return dec_in[:, :, 1:]
            dec_start_pos = self.inseq_len - 1
            dec_end_pos = dec_start_pos + self.outseq_len

            dec_pos_em = hpe[:, dec_start_pos : dec_end_pos]
            outputs = self.decoder(metero_next, enc_out, locs, enc_pos_em, dec_pos_em)
            predicted = self.fc(outputs)

            return predicted


    def hybrid_pos_embedding(
        self,
        time: Dict[str, torch.Tensor]
    ):
        weekly_ids = time["weekday"]
        global_ids = time["timestamp"]

        assert weekly_ids.size(1) == global_ids.size(1), "positional indices must be the same size"
        relative_ids = torch.arange(weekly_ids.size(1), device=self.device)
        relative_ids = relative_ids.unsqueeze(0).repeat_interleave(weekly_ids.size(0), 0)

        weekly_embed = self.pos_embedding(weekly_ids)
        global_embed = self.pos_embedding(global_ids)
        relative_embed = self.pos_embedding(relative_ids)

        hybrid_embed = weekly_embed + global_embed + relative_embed

        return hybrid_embed

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
        return F.l1_loss(input, target, reduction=reduction)

    def training_step(self, batch, batch_idx):
        metero_next = torch.cat((
            batch["metero"][:, :, -1].unsqueeze(2),
            batch["metero_next"][:, :, :-1]
        ), dim=2)

        outs = self(batch["metero"], batch["src_locs"], batch["time"], metero_next)

        loss = self.compute_loss(outs, batch["metero_next"], reduction="mean")
        return loss

    def validation_step(self, batch, batch_idx):
        metero_next = torch.cat((
            batch["metero"][:, :, -1].unsqueeze(2),
            batch["metero_next"][:, :, :-1]
        ), dim=2)

        outs = self(batch["metero"], batch["src_locs"], batch["time"], metero_next)
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
            
            metero_next = torch.cat((
                dt["metero"][:, :, -1].unsqueeze(2),
                dt["metero_next"][:, :, :-1]
            ), dim=2)

            outs = self(dt["metero"], dt["src_locs"], dt["time"], metero_next)
            preds = outs[..., -1] * self.target_normalize_std + self.target_normalize_mean
            targets = dt["metero_next"][..., -1] * self.target_normalize_std + self.target_normalize_mean
        self.train(st)
        return preds.squeeze(0), targets.squeeze(0)