import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .layers import *
from utils.metrics import MultiMetrics
from model import BaseAQFModel


class AQFModel(BaseAQFModel):
    '''
    Dự đoán PM2.5 trong 24h kế tiếp ở những địa điểm không biết trước.

    Args:
        features: Tính năng time series của trạm đo. Tensor (n_batches, n_stations1, n_timesteps, n_features)
        src_locs: Vị trí kinh độ và vĩ độ của các trạm đo cho trước. Tensor (n_batches, n_stations1, 2)
        tar_locs: Vị trí kinh độ và vĩ độ của trạm đo cần dự đoán. Tensor (n_batches, 2)
        src_masks: Đánh dấu những trạm được dùng để dự đoán. Tensor (n_batches, n_stations1)
    Returns:
        outputs: PM2.5 của các trạm đo cần dự đoán. Tensor (n_batches, n_stations2)
    '''

    def __init__(self, config, target_normalize_mean: float, target_normalize_std: float):
        super().__init__(config, target_normalize_mean, target_normalize_std)

        self.extractor_size = config["extractor_size"]

        self.feat_extractor = nn.LSTM(
            input_size=config["n_features"],
            hidden_size=self.extractor_size,
            batch_first=True,
            num_layers=config["num_extractor_layers"]
        )

        self.invdist_attention = InverseDistanceAttention(
            config,
            self.extractor_size
        )

        self.hybrid_predictor = HybridPredictor(config, self.extractor_size)

    def forward(self, features: torch.Tensor, src_locs: torch.Tensor, tar_loc: torch.Tensor, src_masks: torch.Tensor):
        batch_size, n_stations1, n_timesteps, n_features = features.size()

        # feat_extracted.shape == (batch_size, n_stations, extractor_size)
        feat_extracted = features.new_zeros((batch_size, n_stations1, self.extractor_size))
        for s in range(n_stations1):
            self.feat_extractor.flatten_parameters()
            feat_extracted[:, s, :] = self.feat_extractor(features[:, s, :, :])[0][:, -1]
        
        output = self.invdist_attention(feat_extracted, src_locs, tar_loc, src_masks)
        output = self.hybrid_predictor(output)

        return output