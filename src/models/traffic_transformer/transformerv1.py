import math
import torch
from torch import nn
import torch.nn.functional as F


class PositionalEmbedding:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

    def __call__(self, pos_ids: torch.Tensor):
        """
        Args:
            pos_ids: Tensor (batch_size, seq_len)

        Returns:
            pos_enc: Tensor (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = pos_ids.shape
        device = pos_ids.device
        N = 1e4

        pos_ids = pos_ids.unsqueeze(-1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=device) *
            (-math.log(N) / self.embedding_dim)
        )

        pe = torch.zeros(batch_size, seq_len,
                         self.embedding_dim, device=device)
        pe[..., 0::2] = torch.sin(pos_ids * div_term)
        pe[..., 1::2] = torch.cos(pos_ids * div_term)

        return pe

class MultiHeadAttention(nn.Module):
    def __init__(self, dmodel: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert dmodel % nhead == 0, "embed_dim must be divisible by num_heads"

        self.add_bias = False
        self.head_dim = dmodel // nhead
        self.dmodel = dmodel
        self.nhead = nhead
        self.dropout = nn.Dropout(p=dropout)

        self.proj_q = nn.Linear(dmodel, dmodel, bias=self.add_bias)
        self.proj_k = nn.Linear(dmodel, dmodel, bias=self.add_bias)
        self.proj_v = nn.Linear(dmodel, dmodel, bias=self.add_bias)

        self.linear = nn.Linear(dmodel, dmodel, bias=self.add_bias)

        self.layer_norm = nn.LayerNorm(dmodel)

    def dot_product_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ):
        # q, k, v have shape (batch_size, num_heads, seq_length, head_dim)
        # pos_embedding has shape (batch_size, seq_length, head_dim)

        prod = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = (mask - 1) * 1e8
            if len(mask.shape) == 2:
                mask = mask[:, None, None, :]
            else:
                mask = mask.unsqueeze(1)

            prod = prod + mask

        # attn_logits.shape == (batch_size, num_heads, seq_len, seq_len)
        attn_logits = F.softmax(prod, dim=-1)
        # (batch_size, num_heads, seq_len, head_dim)
        attn_values = torch.matmul(attn_logits, v)

        return attn_values

    def split_heads(self, x: torch.Tensor):
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, head_dim)
        x = x.view(x.size(0), x.size(1), self.nhead, self.head_dim)

        return x.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ):
        """
        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)
            value: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len) | (batch_size, seq_len, seq_len)
        """
        residual = query

        query = self.proj_q(query)
        key = self.proj_k(key)
        value = self.proj_v(value)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        attn_values = self.dot_product_attn(
            query, key, value, mask)
        attn_values = attn_values.transpose(1, 2).contiguous()\
            .view(attn_values.size(0), -1, self.dmodel)

        output = self.linear(attn_values)
        output = self.dropout(output)

        return self.layer_norm(residual + output)


class SimilarityAttention(nn.Module):
    def __init__(self, dmodel: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert dmodel % nhead == 0, "embed_dim must be divisible by num_heads"

        self.add_bias = False
        self.head_dim = dmodel // nhead
        self.dmodel = dmodel
        self.nhead = nhead
        self.dropout = nn.Dropout(p=dropout)

        self.proj_q = nn.Linear(dmodel, dmodel, bias=self.add_bias)
        self.proj_k = nn.Linear(dmodel, dmodel, bias=self.add_bias)
        self.proj_v = nn.Linear(dmodel, dmodel, bias=self.add_bias)

        self.linear = nn.Linear(dmodel, dmodel, bias=self.add_bias)

        self.layer_norm = nn.LayerNorm(dmodel)

    def dot_product_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_pos_em: torch.Tensor,
        k_pos_em: torch.Tensor,
        mask: torch.Tensor = None
    ):
        # q, k, v have shape (batch_size, num_heads, seq_length, head_dim)
        # pos_embedding has shape (batch_size, seq_length, head_dim)

        simi = torch.softmax(torch.matmul(
            q_pos_em, k_pos_em.transpose(1, 2)), dim=-1)

        prod = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        prod = prod * simi.unsqueeze(1).repeat_interleave(self.nhead, 1)

        if mask is not None:
            mask = (mask - 1) * 1e8
            if len(mask.shape) == 2:
                mask = mask[:, None, None, :]
            else:
                mask = mask.unsqueeze(1)

            prod = prod + mask

        # attn_logits.shape == (batch_size, num_heads, seq_len, seq_len)
        attn_logits = F.softmax(prod, dim=-1)
        # (batch_size, num_heads, seq_len, head_dim)
        attn_values = torch.matmul(attn_logits, v)

        return attn_values

    def split_heads(self, x: torch.Tensor):
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, head_dim)
        x = x.view(x.size(0), x.size(1), self.nhead, self.head_dim)

        return x.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos_em: torch.Tensor,
        key_pos_em: torch.Tensor,
        mask: torch.Tensor = None
    ):
        """
        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)
            value: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len) | (batch_size, seq_len, seq_len)
        """
        residual = query

        query = self.proj_q(query)
        key = self.proj_k(key)
        value = self.proj_v(value)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        attn_values = self.dot_product_attn(
            query, key, value, query_pos_em, key_pos_em, mask)
        attn_values = attn_values.transpose(1, 2).contiguous()\
            .view(attn_values.size(0), -1, self.dmodel)

        output = self.linear(attn_values)
        output = self.dropout(output)

        return self.layer_norm(residual + output)


class PositionWiseFFN(nn.Module):
    def __init__(self, input_size: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()

        hidden_size = expansion_factor * input_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(input_size)

    def forward(self, x: torch.Tensor):
        out = self.fc1(x)
        out = F.relu(out)

        out = self.fc2(out)
        out = self.dropout(out)

        return self.norm(x + out)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dmodel: int,
        nhead: int,
        expansion_factor: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            x: Tensor (batch_size, inseq_len, d_model)
            pos_em: Tensor (batch_size, inseq_len, embedding_dim)
            mask: Optional

        Returns:
            output: Tensor (batch_size, inseq_len, d_model)
        """
        super().__init__()

        # self.mha = SimilarityAttention(dmodel, nhead, dropout)
        self.mha = MultiHeadAttention(dmodel, nhead, dropout)
        self.ffn = PositionWiseFFN(
            dmodel, expansion_factor=expansion_factor, dropout=dropout)

    def forward(self, x: torch.Tensor, pos_em: torch.Tensor, mask: torch.Tensor = None):
        # mha_out = self.mha(x, x, x, pos_em, pos_em, mask)

        x = x + pos_em
        mha_out = self.mha(x, x, x, mask)

        ffn_out = self.ffn(mha_out)

        return ffn_out


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dmodel: int,
        nhead: int,
        expansion_factor: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            dec_in: Tensor (batch_size, inseq_len, d_model)
            enc_out: Tensor (batch_size, outseq_len, d_model)
            enc_pos_embedding: Tensor (batch_size, inseq_len, embedding_dim)
            dec_pos_embedding: Tensor (batch_size, outseq_len, embedding_dim)
            enc_mask: Optional
            dec_mask: Optional

        Returns:
            outputs: Tensor (batch_size, outseq_len, d_model)
        """
        super().__init__()

        # self.self_mha = SimilarityAttention(dmodel, nhead, dropout)
        # self.enc_mha = SimilarityAttention(dmodel, nhead, dropout)
        self.self_mha = MultiHeadAttention(dmodel, nhead, dropout)
        self.enc_mha = MultiHeadAttention(dmodel, nhead, dropout)

        self.ffn = PositionWiseFFN(
            dmodel, expansion_factor=expansion_factor, dropout=dropout)

    def _create_subsequent_mask(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        # mask = torch.triu(-1 * torch.ones(
        #     seq_len, seq_len, device=x.device, dtype=torch.int), diagonal=1) + 1
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.int)).T

        return mask.unsqueeze(0).repeat_interleave(batch_size, 0)

    def forward(
        self,
        dec_in: torch.Tensor,
        enc_out: torch.Tensor,
        enc_pos_embedding: torch.Tensor,
        dec_pos_embedding: torch.Tensor,
        enc_mask: torch.Tensor = None,
        dec_mask: torch.Tensor = None
    ):
        subseq_mask = self._create_subsequent_mask(dec_in)

        if dec_mask is not None:
            if len(dec_mask.shape) == 2:
                dec_mask = dec_mask.unsqueeze(
                    1).repeat_interleave(dec_mask.size(1), 1)

            dec_mask = torch.logical_and(dec_mask, subseq_mask)
        else:
            dec_mask = subseq_mask

        # self_mha_out = self.self_mha(
        #     dec_in, dec_in, dec_in, dec_pos_embedding, dec_pos_embedding, dec_mask)

        # dec_enc_out = self.enc_mha(
        #     self_mha_out, enc_out, enc_out, dec_pos_embedding, enc_pos_embedding, enc_mask)

        dec_in = dec_in + dec_pos_embedding
        self_mha_out = self.self_mha(dec_in, dec_in ,dec_in, dec_mask)
        dec_enc_out = self.enc_mha(self_mha_out, enc_out, enc_out, enc_mask)

        ffn_out = self.ffn(dec_enc_out)

        return ffn_out
