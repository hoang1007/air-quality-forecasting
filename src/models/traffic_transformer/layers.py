import math
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from .utils import inverse_distance_weighting
from .transformerv1 import TransformerEncoderLayer, TransformerDecoderLayer, positional_encoding
from torch_geometric.nn import GCNConv


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

        self.graph_conv = GCNConv(self.n_features, self.gcn_dim)
        self.fc1 = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.dropout = nn.Dropout(config["dropout"])
        self.transformer_enc = TransformerEncoderLayer(
            dmodel=self.gcn_dim + self.n_features,
            nhead=self.num_attn_heads,
            expansion_factor=config["expansion_factor"],
            dropout=config["dropout"]
        )

        # self.transformer_enc = nn.TransformerEncoderLayer(
        #     d_model=self.gcn_dim + self.n_features,
        #     nhead=self.num_attn_heads,
        #     dim_feedforward=self.gcn_dim * config["expansion_factor"],
        #     dropout=config["dropout"],
        #     batch_first=True
        # )

    def forward(
        self,
        x: torch.Tensor,
        locs: torch.Tensor,
        pos_em: torch.Tensor,
        periodic_em: torch.Tensor
    ):
        # x.shape == (batch_size, n_stations, seq_len, n_features)
        batch_size, n_stations, inseq_len, _ = x.shape
        edge_weights, edge_ids = inverse_distance_weighting(locs, norm=True)
        periodic_em = periodic_em.repeat_interleave(n_stations, 0)
        pos_em = pos_em.repeat_interleave(n_stations, 0)
        gcn_outs = x.new_empty(batch_size, n_stations, inseq_len, self.gcn_dim)
        
        for batch_idx in range(batch_size):
            for seq_idx in range(inseq_len):
                gcn_outs[batch_idx, :, seq_idx] = torch.relu(self.graph_conv(x[batch_idx, :, seq_idx], edge_ids[batch_idx], edge_weights[batch_idx]))
        gcn_outs = self.fc1(gcn_outs)
    
        gcn_outs = torch.cat((x, gcn_outs), dim=-1).view(-1, inseq_len, self.gcn_dim + self.n_features)

        gcn_outs = self.dropout(gcn_outs + pos_em)
        outs = self.transformer_enc(gcn_outs, periodic_em)

        outs = outs.view(batch_size, n_stations, inseq_len, self.gcn_dim + self.n_features)
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

        self.graph_conv = GCNConv(self.n_features, self.gcn_dim)
        self.fc1 = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.dropout = nn.Dropout(config["dropout"])
        # self.transformer_dec = nn.TransformerDecoderLayer(
        #     d_model=self.gcn_dim + self.n_features,
        #     nhead=self.num_attn_heads,
        #     dim_feedforward=self.gcn_dim * config["expansion_factor"],
        #     dropout=config["dropout"],
        #     batch_first=True
        # )
        self.transformer_dec = TransformerDecoderLayer(
            dmodel=self.gcn_dim + self.n_features,
            nhead=self.num_attn_heads,
            expansion_factor=config["expansion_factor"],
            dropout=config["dropout"]
        )

    def forward(
        self,
        dec_in: torch.Tensor,
        enc_out: torch.Tensor,
        locs: torch.Tensor,
        dec_pos_em: torch.Tensor,
        enc_periodic_em: torch.Tensor,
        dec_periodic_em: torch.Tensor
    ):
        batch_size, n_tar_stations, outseq_len, _ = dec_in.shape
        n_src_stations, inseq_len = enc_out.size(1), enc_out.size(2)

        dec_periodic_em = dec_periodic_em.repeat_interleave(n_tar_stations, 0)
        enc_periodic_em = enc_periodic_em.repeat_interleave(n_src_stations, 0)
        dec_pos_em = dec_pos_em.repeat_interleave(n_tar_stations, 0)

        edge_weights, edge_ids = inverse_distance_weighting(locs, norm=True)
        gcn_outs = dec_in.new_empty(batch_size, n_tar_stations, outseq_len, self.gcn_dim)

        for batch_idx in range(batch_size):
            for seq_idx in range(outseq_len):
                gcn_outs[batch_idx, :, seq_idx] = torch.relu(self.graph_conv(dec_in[batch_idx, :, seq_idx], edge_ids[batch_idx], edge_weights[batch_idx]))
        gcn_outs = self.fc1(gcn_outs)

        dec_in = torch.cat((dec_in, gcn_outs), dim=-1).view(-1, outseq_len, self.gcn_dim + self.n_features)
        enc_out = enc_out.view(-1, inseq_len, self.gcn_dim + self.n_features)

        dec_in = self.dropout(dec_in + dec_pos_em)
        outs = self.transformer_dec(dec_in, enc_out, enc_periodic_em, dec_periodic_em)
        outs = outs.view(batch_size, n_tar_stations, outseq_len, self.gcn_dim + self.n_features)
        
        return outs

    # def _create_subsequent_mask(self, x: torch.Tensor):
    #     batch_size, seq_len, _ = x.shape

    #     # mask = torch.triu(-1 * torch.ones(
    #     #     seq_len, seq_len, device=x.device, dtype=torch.int), diagonal=1) + 1
    #     mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)

    #     return mask.unsqueeze(0).repeat_interleave(batch_size * self.num_attn_heads, 0)