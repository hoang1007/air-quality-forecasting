from models import BaseAQFModel
import torch
from torch import nn


class EncoderDecoder(BaseAQFModel):
    def __init__(self, config, target_normalize_mean, target_normalize_std):
        """
        Args:
            x: Tensor (batch_size, n_stations, seq_len, n_features)
            locs: Tensor (batch_size, n_stations, 2)
            masks: Tensor (batch_size, n_stations)
        Returns:
            outputs: Tensor (batch_size, n_stations, output_len)
        """
        super().__init__(config, target_normalize_mean, target_normalize_std)

        self.input_size = config["n_features"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.output_size = config["output_size"]
        self.out_seq_len = config["out_seq_len"]

        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config["dropout"]
        )

        self.decoder = nn.LSTM(
            input_size=self.output_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config["dropout"]
        )

        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: torch.Tensor, locs: torch.Tensor, masks: torch.Tensor):
        # x.shape == (batch_size, src_len, n_seq, n_features)
        enc_out, enc_state = self.encoder()