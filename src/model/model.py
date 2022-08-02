import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .layers import *
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
        # self.invdist_pooling = InverseDistancePooling()

        self.hybrid_predictor = HybridPredictor(config, self.extractor_size)

        self.target_normalize_mean = target_normalize_mean
        self.target_normalize_std = target_normalize_std

        self.metrics = MultiMetrics()

        self.optim_config = config["optim"]

    def forward(self, features: torch.Tensor, src_locs: torch.Tensor, tar_loc: torch.Tensor, src_mask: torch.Tensor):
        batch_size, n_stations1, n_timesteps, n_features = features.size()

        feat_extracted = features.new_zeros((batch_size, n_stations1, self.extractor_size))
        for s in range(n_stations1):
            self.feat_extractor.flatten_parameters()
            feat_extracted[:, s, :] = self.feat_extractor(features[:, s, :, :])[0][:, -1]

        output = self.invdist_attention(feat_extracted, src_locs, tar_loc)
        output = self.hybrid_predictor(output, src_mask)

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
        outputs = self(batch["features"], batch["src_locs"], batch["tar_loc"], batch["mask"])

        loss = F.mse_loss(outputs, batch["target"])

        self.log("loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        # outputs.shape == (batch_size, n_pred_steps)
        outputs = self(batch["features"], batch["src_locs"], batch["tar_loc"], batch["mask"])

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
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.optim_config["lr"])

        print("Using", self.optim_config["scheduler"])
        
        if self.optim_config["scheduler"] == "plateau":
            scheduler_cfg = self.optim_config["plateau"]
            scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=scheduler_cfg["patience"],
                eps=scheduler_cfg["eps"]
            ),
                "monitor": scheduler_cfg["monitor"]}
        else:
            scheduler_cfg = self.optim_config["steplr"]
            scheduler = {"scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=scheduler_cfg["step_size"])}
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
