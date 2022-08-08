import torch
from torch import nn
import random
from utils.functional import scale_softmax

class InverseDistanceAttention(nn.Module):
    '''
    Args:
        features: Tensor (batch_size, src_length, n_features)
        src_locs: Tensor (batch_size, src_length, 2)
        tar_locs: Tensor (batch_size, 2)
        src_masks: Tensor (batch_size, src_length)

    Returns:
        outputs: Tensor (batch_size, n_features)
    '''
    def __init__(
        self,
        config,
        n_features: int):
        super().__init__()

        self.ff_dim = config["attn_ff_dim"]
        self.n_features = n_features

        self.attention = nn.Sequential(
            nn.Linear(n_features, self.ff_dim),
            nn.Sigmoid(),
            nn.Linear(self.ff_dim, 1),
            nn.Sigmoid()
        )
        
        # self.linear = nn.Linear(n_features, n_features)
        # self.dropout = nn.Dropout(config["attn_dropout"])
    
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
        inv_dists = torch.div(1, dists + 1e-8).float()

        return inv_dists

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

        locs[:, :, 0] = torch.div(locs[:, :, 0] - x_min, x_min_diff, rounding_mode="trunc")
        locs[:, :, 1] = torch.div(locs[:, :, 1] - y_min, y_min_diff, rounding_mode="trunc")

        return locs[:, :num_src_locs], locs[:, num_src_locs:]

    def forward(
        self,
        features: torch.Tensor,
        src_locs: torch.Tensor,
        tar_locs: torch.Tensor,
        src_masks: torch.Tensor
    ):
        batch_size = features.size(0)
        tar_locs = tar_locs.unsqueeze(1) # only support 1 target location

        attn = self.attention(features) # shape == (batch_size, src_length, 1)
        
        outputs = features.new_zeros(batch_size, self.n_features)
        for i in range(batch_size):
            __src_masks = src_masks[i]
            __src_locs = src_locs[i, __src_masks].unsqueeze(0)
            __tar_loc = tar_locs[i].unsqueeze(0)

            # grid_src_locs, grid_tar_locs = self.locs_to_grid(__src_locs, __tar_loc)
            # inv_dists.shape == (1, src_len', 1)
            inv_dists = self.compute_invdist_scores(__src_locs, __tar_loc).squeeze(0)

            # __attn.shape == (src_len', 1)
            __attn = attn[i, __src_masks] * inv_dists
            outputs[i] = (features[i, __src_masks] * torch.softmax(__attn, dim=0)).sum(0)

        # outputs = self.linear((attn_scores * features).sum(1))
        # outputs = self.dropout(outputs)
        return outputs


class InverseDistancePooling(nn.Module):
    def __init__(self):
        super().__init__()

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
        inv_dists = torch.div(1, dists + 1e-8).float()

        return inv_dists

    def forward(self, inputs: torch.Tensor, src_locs: torch.Tensor, tar_locs: torch.Tensor):
        # inputs.shape == (batch_size, src_len, n_features)

        # inv_dists.shape == (batch_size, src_len, 1)
        inv_dists = self.compute_invdist_scores(src_locs, tar_locs.unsqueeze(1))

        pooled = (inputs * inv_dists).sum(1) / inv_dists.sum(1)

        return pooled


class LSTMAutoEncoder(nn.Module):
    def __init__(self, config, n_features: int):
        super().__init__()
        self.output_dim = config["output_dim"]

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=config["hidden_size_autoencoder"],
            batch_first=True,
            num_layers=config["num_enc_layers"]
        )

        self.decoder = nn.LSTM(
            input_size=config["hidden_size_autoencoder"],
            hidden_size=config["hidden_size_autoencoder"],
            batch_first=True,
            num_layers=config["num_dec_layers"]
        )

        self.linear2 = nn.Linear(config["hidden_size_autoencoder"], 1)

    def forward(self, x: torch.Tensor):
        # x.shape == (batch_size, n_features)
        encoder_output, enc_state = self.encoder(x.unsqueeze(1))
        encoder_output = torch.relu(encoder_output)

        decoder_in, _ = self.decoder(encoder_output, enc_state)

        outputs = [decoder_in]

        for t in range(self.output_dim - 1):
            pred, _ = self.decoder(decoder_in, enc_state)

            outputs.append(pred)
            decoder_in = pred
            # pred = pred[:, -1].unsqueeze(1)
            # decoder_in = torch.cat((decoder_in, pred), dim=1)
        
        # outputs = decoder_in
        outputs = torch.cat(outputs, dim=1)
        outputs = self.linear2(outputs)

        return outputs


class HybridPredictor(nn.Module):
    '''
    Args:
        inputs: Tensor (batch_size, n_features)
        src_mask: Tensor (batch_size, src_length)
    '''
    def __init__(self, config, n_features: int):
        super().__init__()
        self.output_dim = config["output_dim"]

        self.linear1 = nn.Linear(n_features, n_features)

        self.indep_predictor = nn.Linear(n_features, config["output_dim"])

        self.dep_predictor = LSTMAutoEncoder(config, n_features)

        self.alpha = torch.tensor(0.5, requires_grad=True)
        
    def forward(self, inputs):
        inputs = torch.relu(self.linear1(inputs))

        indep_output = self.indep_predictor(inputs)

        dep_output = self.dep_predictor(inputs).squeeze(-1)

        hybrid_out = indep_output * self.alpha + dep_output * (1 - self.alpha)
        
        return hybrid_out


if __name__ == "__main__":
    # ida = InverseDistanceAttention(3)

    feats = torch.rand((2, 3, 10, 3))
    src_locs = torch.rand((2, 3, 2))
    tar_locs = torch.rand((2, 2))

    # print(ida(feats, src_locs, tar_locs).shape)
    idl = InverseDistancePooling()

    print(idl(feats, src_locs, tar_locs).shape) 