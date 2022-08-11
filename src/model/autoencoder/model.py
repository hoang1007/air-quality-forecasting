import torch
from torch import nn
from utils.functional import dilate_loss
from model import BaseAQFModel


class AutoEncoder(BaseAQFModel):
    def __init__(self, config, target_normalize_mean: float, target_normalize_std: float):
        """
        Args:
            x: Tensor (1, n_stations1, seq_len, n_features)
            src_locs: Tensor (1, n_stations1, 2)
            tar_loc: Tensor (1, 2)
            src_masks: Tensor (1, n_stations1)

        Returns:
            outputs: Tensor (1, output_len)

        Note: Only support batch size 1
        """
        super().__init__(config, target_normalize_mean, target_normalize_std)

        self.hidden_size = config["hidden_size"]
        self.output_size = config["output_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.ff_dim = config["ff_dim"]
        self.beta = config["id_beta"]

        self.encoder = nn.LSTM(
            input_size=config["n_features"],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )

        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        src_locs: torch.Tensor,
        tar_loc: torch.Tensor,
        src_masks: torch.Tensor
    ):  
        assert x.size(0) == 1, "Only support one batch size"
        # Remove batch dim
        tar_loc = tar_loc.squeeze(0)
        src_masks = src_masks.squeeze(0)
        src_locs = src_locs.squeeze(0)[src_masks]
        x = x.squeeze(0)[src_masks]

        ht, ct = self._init_state(x.size(0), self.hidden_size, self.num_layers, x.device)
        idws = self.compute_invdist_scores(src_locs, tar_loc)

        # enc_outputs.shape == (n_stations, seq_len, enc_hidden_size)
        enc_outputs, (ht, ct) = self.encoder(x)

        # ht.shape == (num_layers, 1, hidden_size)
        ht = torch.sum(ht * idws[None, :, None], dim=1, keepdim=True)
        ct = torch.sum(ct * idws[None, :, None], dim=1, keepdim=True)
        enc_outputs = torch.sum(enc_outputs[:, -1] * idws[:, None], dim=0, keepdim=True).unsqueeze(1)

        dec_outputs = self.decoder(enc_outputs, (ht, ct))[0]

        for i in range(1, self.output_size):
            self.decoder.flatten_parameters()
            # out.shape == (1, dec_hidden_size)
            out = self.decoder(dec_outputs, (ht, ct))[0][:, -1]

            dec_outputs = torch.cat((dec_outputs, out.unsqueeze(1)), dim=1)

        outputs = self.linear(dec_outputs)

        return outputs.squeeze(-1)

    def _init_state(
            self,
            batch_size: int,
            hidden_size: int,
            num_layers: int,
            device: int):

        ht = torch.zeros(2 * num_layers, batch_size,
                         hidden_size, device=device)
        ct = torch.zeros(2 * num_layers, batch_size,
                         hidden_size, device=device)

        nn.init.xavier_uniform_(ht)
        nn.init.xavier_uniform_(ct)

        return ht, ct

    def compute_invdist_scores(self, src_locs: torch.Tensor, tar_loc: torch.Tensor):
        '''
        Compute inverse distance weights

        Args:
            src_locs: Tensor (src_len, 2)
            tar_loc: Tensor (2,)

        Returns:
            scores: Tensor (src_len,)
        '''
        tar_loc = tar_loc.unsqueeze(0)
        # dists.shape == (src_len, 1)
        dists = (src_locs - tar_loc).pow(2).sum(dim=-1).sqrt()
        
        scores = torch.pow(dists, -self.beta)
        scores = scores / scores.sum()

        return scores

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor):
        loss, _, _ = dilate_loss(input.unsqueeze(-1), target.unsqueeze(-1), alpha=0.2, gamma=0.001)

        return loss