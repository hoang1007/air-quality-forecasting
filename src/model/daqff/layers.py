from typing import List
import torch
from torch import nn
from utils.functional import scale_softmax


class Conv1dExtractor(nn.Module):
    def __init__(self, config):
        """
        Args:
            x: Tensor (batch_size, src_len, seq_len, n_features)

        Returns:
            outputs: Tensor (batch_size, extractor_size, src_len)
        """
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=config["n_features"],
            out_channels=config["channels1"],
            kernel_size=config["kernel_size1"],
            padding=2
        )

        self.conv2 = nn.Conv1d(
            in_channels=config["channels1"],
            out_channels=config["channels2"],
            kernel_size=config["kernel_size2"],
            padding=1
        )

        self.conv3 = nn.Conv1d(
            in_channels=config["channels2"],
            out_channels=config["channels3"],
            kernel_size=config["kernel_size3"],
            padding=0
        )

        self.bn1 = nn.BatchNorm1d(config["channels1"])
        self.bn2 = nn.BatchNorm1d(config["channels2"])
        self.bn3 = nn.BatchNorm1d(config["channels3"])

        self.linear = nn.Linear(
            config["channels3"] * config["seq_len"], config["extractor_size"])
        self.dropout = nn.Dropout(p=config["extractor_dropout"])

    def forward(self, x: torch.Tensor):
        src_len = x.size(1)

        extracted = []
        for i in range(src_len):
            # xi.shape == (batch_size, n_features, seq_len)
            xi = x[:, i].transpose(1, 2)

            out1 = torch.relu(self.bn1(self.conv1(xi)))
            out2 = torch.relu(self.bn2(self.conv2(out1)))
            out3 = torch.relu(self.bn3(self.conv3(out2)))

            out_flatten = out3.flatten(start_dim=1)

            out = torch.relu(self.dropout(
                self.linear(out_flatten))).unsqueeze(-1)

            extracted.append(out)

        extracted = torch.cat(extracted, dim=2)

        return extracted


class AIDWLayer(nn.Module):
    def __init__(self, config):
        """
        Args:
            features: Tensor (batch_size, seq_len, src_len)
            src_locs: Tensor (batch_size, src_len, 2)
            tar_loc: Tensor (batch_size, 2)
            src_masks: Tensor (batch_size, src_len)

        Returns:
            outputs: Tensor (batch_size, seq_len, n_features)
        """
        super().__init__()

        self.output_size = config["aidw_output_size"]
        self.src_len = config["n_stations"]
        self.beta = config["idw_beta"]

        self.register_parameter(
            "linear", self._init_feedforward(self.src_len, self.output_size))

    def forward(
        self,
        features: torch.Tensor,
        src_locs: torch.Tensor,
        tar_loc: torch.Tensor,
        src_masks: torch.Tensor
    ):
        batch_size = features.size(0)
        tar_locs = tar_loc.unsqueeze(1)  # only support 1 target location

        out = features.new_zeros(batch_size, features.size(1), self.output_size)

        for i in range(batch_size):
            __src_masks = src_masks[i]
            __src_locs = src_locs[i, __src_masks].unsqueeze(0)
            __tar_loc = tar_locs[i].unsqueeze(0)

            # inv_dists.shape == (src_len')
            id_weights = self.compute_invdist_scores(
                __src_locs, __tar_loc).squeeze()

            # feat.shape == (seq_len, src_len')
            feat = features[i, :, __src_masks] * id_weights
            out[i] = torch.matmul(feat, self.linear[__src_masks])

        return out

    def _init_feedforward(self, input_size: int, output_size: int):
        weights = torch.zeros(input_size, output_size, requires_grad=True)
        nn.init.xavier_uniform_(weights)

        return nn.parameter.Parameter(weights)

    def compute_invdist_scores(self, src_locs: torch.Tensor, tar_locs: torch.Tensor):
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
        
        scores = torch.pow(dists, -self.beta)
        scores = scores / scores.sum(dim=1, keepdim=True)

        return scores

    def locs_to_grid(self, src_locs: torch.Tensor, tar_locs: torch.Tensor):
        num_src_locs = src_locs.size(1)
        # (batch_size, n_locs, 2)
        locs = torch.cat((src_locs, tar_locs), dim=1)

        x_sorted = torch.sort(locs[:, :, 0], dim=-1).values
        y_sorted = torch.sort(locs[:, :, 1], dim=-1).values

        x_min = x_sorted[:, :1]
        y_min = y_sorted[:, :1]

        x_min_diff = torch.diff(x_sorted)
        x_min_diff[x_min_diff.eq(0)] = 1e9
        x_min_diff = x_min_diff.min(dim=-1, keepdim=True).values

        y_min_diff = torch.diff(y_sorted)
        y_min_diff[y_min_diff.eq(0)] = 1e9
        y_min_diff = y_min_diff.min(dim=-1, keepdim=True).values

        locs[:, :, 0] = torch.div(
            locs[:, :, 0] - x_min, x_min_diff, rounding_mode="trunc")
        locs[:, :, 1] = torch.div(
            locs[:, :, 1] - y_min, y_min_diff, rounding_mode="trunc")

        return locs[:, :num_src_locs], locs[:, num_src_locs:]


class SpatialTemporalLayer(nn.Module):
    """
    Args:
        x: Tensor (batch_size, seq_len, src_len)

    Returns:
        outputs: Tensor (batch_size, output_size)
    """

    def __init__(self, config, input_size: int):
        """
        Args:
            x: Tensor (batch_size, seq_len, src_len)

        Returns:
            outputs: Tensor (batch_size, output_size)
        """
        super().__init__()

        self.hidden_size = config["lstm_hidden_size"]
        self.num_layers = config["lstm_num_layers"]
        self.input_size = input_size
        self.lookup_size = config["lstm_lookup_size"]
        self.output_size = config["lstm_output_size"]

        self.bi_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

        lstm_flatten_size = (self.lookup_size + 1) * self.hidden_size * 2
        self.fusion = nn.Linear(lstm_flatten_size, self.output_size)

    def forward(self, x: torch.Tensor):
        state = self._init_state(
            x.size(0), self.hidden_size, self.num_layers, x.device)
        # outputs.shape == (batch_size, seq_len, hidden_size)
        outputs, _ = self.bi_lstm(x, state)

        O = outputs[:, -self.lookup_size - 1:]

        out = torch.flatten(O, start_dim=1)
        out = self.fusion(out)

        return out

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


if __name__ == "__main__":
    convex = Conv1dExtractor({
        "n_features": 11,
        "channels1": 32,
        "channels2": 16,
        "channels3": 8,
        "kernel_size1": 5,
        "kernel_size2": 3,
        "kernel_size3": 1,
        "extractor_size": 64,
        "extractor_dropout": 0.1
    })

    print(convex(torch.rand((2, 6, 5, 11))).shape)

    # st_layer = SpatialTemporalLayer({
    #     "n_stations": 10,
    #     "lstm_hidden_size": 5,
    #     "lstm_num_layers": 2,
    #     "lstm_output_size": 20
    # })

    # print(st_layer(torch.rand((3, 12, 10))))
