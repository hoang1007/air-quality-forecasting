import math
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from .utils import inverse_distance_weighting
from .transformerv1 import TransformerEncoderLayer, TransformerDecoderLayer, PositionalEmbedding


class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Args:
            input: Tensor (batch_size, num_vertices, in_features)
            edge_ids: List of Tensor (2, npairs)
            edge_weights: List of Tensor (npairs)

        Returns:
            output: Tensor (batch_size, num_vertices, out_features)
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self._weights, self._bias = self._init_params(bias)

    def _init_params(self, use_bias: bool):
        weights = torch.zeros(self.in_features, self.out_features)
        nn.init.xavier_uniform_(weights)
        weights = nn.parameter.Parameter(weights)

        if use_bias:
            bias = torch.zeros(self.out_features)
            nn.init.uniform_(bias)
            bias = nn.parameter.Parameter(bias)
        else:
            bias = None

        return weights, bias

    def forward(self, input: torch.Tensor, edge_ids: List[torch.Tensor], edge_weights: List[torch.Tensor]):
        # support.shape == (batch_size, V, out_features)
        support = torch.matmul(input, self._weights)
        outputs = []

        for i in range(len(edge_ids)):
            adj = torch.sparse_coo_tensor(
                indices=edge_ids[i],
                values=edge_weights[i],
            )

            out = torch.sparse.mm(adj, support[i])
            outputs.append(out)
        outputs = torch.stack(outputs, dim=0)

        if self._bias is not None:
            outputs = outputs + self._bias
        
        return outputs

class GCN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.gc1 = GraphConv(in_features, hidden_size)
        self.gc2 = GraphConv(hidden_size, out_features)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_ids, edge_weights):
        x = F.relu(self.gc1(x, edge_ids, edge_weights))
        x = self.dropout(x)
        x = self.gc2(x, edge_ids, edge_weights)

        # return F.log_softmax(x, dim=1)
        return x

class TrafficTransformerEncoder(nn.Module):
    def __init__(self, config):
        """
        Args:
            x: Tensor (batch_size, num_stations, inseq_len, n_features)
            locs: Tensor (batch_size, num_stations, 2)
            pos_embedding: Tensor (batch_size, inseq_len, embedding_dim)
        """
        super().__init__()

        self.n_features = config["n_features"]
        self.embedding_dim = config["pos_embedding_dim"]
        self.gcn_dim = config["gcn_dim"]
        self.gcn_hidden_dim = config["gcn_hidden_dim"]
        self.num_attn_heads = config["num_attn_heads"]

        self.pos_encoder = PositionalEmbedding(self.embedding_dim)
        self.graph_conv = GCN(self.n_features, self.gcn_dim, self.gcn_hidden_dim, config["dropout"])
        self.fc1 = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.dropout = nn.Dropout(p=config["dropout"])

        self.transformer_enc = TransformerEncoderLayer(
            dmodel=self.gcn_dim,
            nhead=self.num_attn_heads,
            expansion_factor=config["expansion_factor"],
            dropout=config["dropout"]
        )

    def forward(
        self,
        x: torch.Tensor,
        locs: torch.Tensor,
        pos_embedding: torch.Tensor
    ):
        # x.shape == (batch_size, n_stations, seq_len, n_features)
        batch_size, n_stations, inseq_len, _ = x.shape
        edge_weights, edge_ids = inverse_distance_weighting(locs)
        pos_embedding = pos_embedding.repeat_interleave(n_stations, 0)
        gcn_outs = x.new_empty(batch_size, n_stations, inseq_len, self.gcn_dim)

        for i in range(inseq_len):
            gcn_outs[:, :, i] = torch.relu(self.graph_conv(x[:, :, i], edge_ids, edge_weights))
        gcn_outs = self.fc1(gcn_outs)

        # gcn_outs = self.dropout(gcn_outs + pos_embedding)
        gcn_outs = gcn_outs.view(-1, inseq_len, self.gcn_dim)

        outs = self.transformer_enc(gcn_outs, pos_embedding).view(batch_size, n_stations, inseq_len, self.gcn_dim)

        return outs


class TrafficTransformerDecoder(nn.Module):
    def __init__(self, config):
        """
        Args:
            dec_in: Tensor (batch_size, num_stations, outseq_len, n_features)
            enc_out: Tensor (batch_size, num_stations, inseq_len, d_model)
            locs: Tensor (batch_size, num_stations, 2)
            pos_embedding: Tensor (batch_size, outseq_len, embedding_dim)

        Returns:
            outputs: Tensor (batch_size, num_stations, outseq_len, d_model)
        """
        super().__init__()

        self.n_features = config["n_features"] 
        self.embedding_dim = config["pos_embedding_dim"]
        self.gcn_dim = config["gcn_dim"]
        self.gcn_hidden_dim = config["gcn_hidden_dim"]
        self.num_attn_heads = config["num_attn_heads"]

        self.pos_encoder = PositionalEmbedding(self.embedding_dim)
        self.graph_conv = GraphConv(self.n_features, self.gcn_dim, self.gcn_hidden_dim)
        self.fc1 = nn.Linear(self.gcn_dim, self.gcn_dim)

        self.transformer_dec = TransformerDecoderLayer(
            dmodel=self.gcn_dim,
            nhead=self.num_attn_heads,
            expansion_factor=config["expansion_factor"],
            dropout=config["dropout"]
        )

    def forward(
        self,
        dec_in: torch.Tensor,
        enc_out: torch.Tensor,
        locs: torch.Tensor,
        enc_pos_embedding: torch.Tensor,
        dec_pos_embedding: torch.Tensor
    ):
        batch_size, n_tar_stations, outseq_len, _ = dec_in.shape
        n_src_stations, inseq_len = enc_out.size(1), enc_out.size(2)

        enc_pos_embedding = enc_pos_embedding.repeat_interleave(n_src_stations, 0)
        dec_pos_embedding = dec_pos_embedding.repeat_interleave(n_tar_stations, 0)

        edge_weights, edge_ids = inverse_distance_weighting(locs)
        gcn_outs = dec_in.new_empty(batch_size, n_tar_stations, outseq_len, self.gcn_dim)

        for i in range(outseq_len):
            gcn_outs[:, :, i] = torch.relu(self.graph_conv(dec_in[:, :, i], edge_ids, edge_weights))
        gcn_outs = self.fc1(gcn_outs)

        # gcn_outs = self.dropout(gcn_outs + pos_embedding)
        dec_in = gcn_outs.view(-1, outseq_len, self.gcn_dim)
        enc_out = enc_out.view(-1, inseq_len, self.gcn_dim)

        outs = self.transformer_dec(dec_in, enc_out, enc_pos_embedding, dec_pos_embedding)
        outs = outs.view(batch_size, n_tar_stations, outseq_len, self.gcn_dim)
        
        return outs


if __name__ == '__main__':
    # pos_encoder = PositionalEncoding(512)

    # pos_enc = pos_encoder(torch.arange(9999, 12345).unsqueeze(0))

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(20, 10))
    # cax = plt.matshow(pos_enc[0], fignum=1, aspect="auto")
    # fig = plt.gcf().colorbar(cax)
    # plt.show()
    self = GraphConv(64, 32)
    x = torch.rand((1, 3, 64))

    output = self(x, [torch.tensor([[0, 0, 1], [1, 2, 2]])], [torch.tensor([0.1, 0.2, 0.3])])

    print(output.shape)