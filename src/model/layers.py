import torch
from torch import nn

class InverseDistanceAttention(nn.Module):
    '''
    Args:
        features: Tensor (batch_size, n_sequence, src_length, n_features)
        src_locs: Tensor (batch_size, src_length, 2)
        tar_locs: Tensor (batch_size, 2)

    Returns:
        outputs: Tensor (batch_size, n_sequence, n_features)
    '''
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

        self.proj1 = nn.Linear(n_features, n_features, bias=True)
        self.proj2 = nn.Linear(n_features, 1, bias=True)
    
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
        inv_dists = torch.div(1, dists).float()

        return inv_dists

    def forward(self, features: torch.Tensor, src_locs: torch.Tensor, tar_locs: torch.Tensor):
        attn = self.proj1(features)
        attn = torch.relu(attn)

        attn = self.proj2(attn)
        attn = torch.relu(attn) # shape == (batch_size, n_sequence, src_length, 1)
        
        # inv_dists.shape == (batch_size, src_length, 1)
        inv_dists = self.compute_invdist_scores(src_locs, tar_locs.unsqueeze(1))

        # attn_scores.shape == (batch_size, n_sequence, src_length, 1)
        attn_scores = torch.softmax(attn * inv_dists.unsqueeze(1), dim=2)

        # outputs.shape == (batch_size, n_sequence, tar_length, n_features)
        # outputs = torch.einsum("abcd,abce->abed", features, attn_scores)
        outputs = (features * attn_scores).sum(2)

        return outputs


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

        self.linear = nn.Linear(config["hidden_size_autoencoder"], 1)

    def forward(self, x: torch.Tensor):
        # x.shape == (batch_size, seq_len, n_features)
        encoder_output, enc_state = self.encoder(x)
        encoder_output = torch.relu(encoder_output[:, -1, :]).unsqueeze(1)

        output, _ = self.decoder(encoder_output, enc_state)

        for t in range(self.output_dim - 1):
            pred, _ = self.decoder(output, enc_state)
            pred = pred[:, -1, :].unsqueeze(1)

            output = torch.cat((output, pred), dim=1)
        
        output = self.linear(output)

        return output


class HybridPredictor(nn.Module):
    '''
    Args:
        inputs: Tensor (batch_size, n_sequence, n_features)
    '''
    def __init__(self, config, n_features: int):
        super().__init__()

        self.linear1 = nn.Linear(n_features, n_features)

        self.indep_predictor = nn.Linear(config["n_sequence"] * n_features, config["output_dim"])

        self.dep_predictor = LSTMAutoEncoder(config, n_features)

        self.weight = nn.Linear(2, 1)
        
    def forward(self, inputs):
        batch_size = inputs.size(0)

        # output.shape == (batch, output_dim, 1)
        indep_output = self.indep_predictor(inputs.view(batch_size, -1)).unsqueeze(-1)

        dep_output = self.dep_predictor(inputs)

        hybrid = torch.cat((indep_output, dep_output), dim=-1)
        
        output = self.weight(hybrid).squeeze(-1)

        return output


if __name__ == "__main__":
    ida = InverseDistanceAttention(3)

    feats = torch.rand((1, 10, 3, 3))
    src_locs = torch.rand((1, 3, 2))
    tar_locs = torch.rand((1, 2))

    print(ida(feats, src_locs, tar_locs).shape)