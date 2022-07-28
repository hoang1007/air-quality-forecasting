import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .layers import InverseDistanceAttention, HybridPredictor, LSTMAutoEncoder
from utils.metrics import MultiMetrics


class AQFModel(pl.LightningModule):
    '''
    Dự đoán PM2.5 trong 24h kế tiếp ở những địa điểm không biết trước.

    Args:
        features: Tính năng time series của trạm đo. Tensor (n_batches, n_stations1, n_timesteps, n_features)
        src_locs: Vị trí kinh độ và vĩ độ của các trạm đo cho trước. Tensor (n_batches, n_stations1, 2)
        tar_locs: Vị trí kinh độ và vĩ độ của trạm đo cần dự đoán. Tensor (n_batches, 2)

    Returns:
        outputs: PM2.5 của các trạm đo cần dự đoán. Tensor (n_batches, n_stations2)
    '''
    def __init__(self, config, target_normalize_mean: float, target_normalize_std: float):
        super().__init__()

        self.feat_extractor = nn.LSTM(
            input_size=config["n_features"],
            hidden_size=config["extractor_size"],
            batch_first=True,
            num_layers=config["num_extractor_layers"]
        )

        self.invdist_attention = InverseDistanceAttention(
            config["extractor_size"], 
            config["attention_dropout"]
        )

        self.autoencoder = LSTMAutoEncoder(config, config["extractor_size"])
        # self.hybrid_predictor = HybridPredictor(config, config["extractor_size"])

        self.target_normalize_mean = target_normalize_mean
        self.target_normalize_std = target_normalize_std

        self.metrics = MultiMetrics()

        self.optim_config = config["optim"]

    def forward(self, features: torch.Tensor, src_locs: torch.Tensor, tar_loc: torch.Tensor):
        batch_size, n_stations1, n_timesteps, n_features = features.size()

        # features.shape == (batch_size * n_stations1, n_timesteps, n_features)
        features = features.view(batch_size * n_stations1, n_timesteps, n_features)

        features = self.feat_extractor(features)[0]

        # features.shape == (batch_size, n_timesteps, n_stations1, n_features)
        features = features.view(batch_size, n_stations1, n_timesteps, -1)\
                        .permute(0, 2, 1, 3)

        output = self.invdist_attention(features, src_locs, tar_loc)
        # output = self.hybrid_predictor(output)
        output = self.autoencoder(output).squeeze(-1)

        return output

    def predict(self,
        features: torch.Tensor,
        src_locs: torch.Tensor,
        tar_loc: torch.Tensor):
        '''
        Args:
            features: Tensor (n_stations, n_timesteps, n_features)
            src_locs: Tensor (n_stations, 2)
            tar_loc: Tensor (2)

        Returns:
            output: Tensor (n_output_timesteps,)
        '''
        self.eval()
        with torch.no_grad():
            # add batch dim
            features = features.unsqueeze(0).to(self.device)
            src_locs = src_locs.unsqueeze(0).to(self.device)
            tar_loc = tar_loc.unsqueeze(0).to(self.device)

            output = self(features, src_locs, tar_loc).squeeze(0)
            # inverse transforms
            output = output * self.target_normalize_std + self.target_normalize_mean
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch["features"], batch["src_locs"], batch["tar_loc"])

        loss = F.mse_loss(outputs, batch["target"])

        self.log("loss", loss.item())

        return loss
    
    def validation_step(self, batch, batch_idx):
        # outputs.shape == (batch_size, n_pred_steps)
        outputs = self(batch["features"], batch["src_locs"], batch["tar_loc"])

        # inverse transform
        preds = outputs * self.target_normalize_std + self.target_normalize_mean

        return self.metrics(preds, batch["gt_target"])

    def validation_epoch_end(self, val_outputs):
        metrics = {}

        for batch in val_outputs:
            for metric in batch:
                if metric in metrics:
                    metrics[metric].append(batch[metric])
                else:
                    metrics[metric] = [batch[metric]]

        for metric in metrics:
            metrics[metric] = sum(metrics[metric]) / len(metrics[metric])

        self.log_dict(metrics)
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optim_config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.optim_config["step_size"])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            }
        }