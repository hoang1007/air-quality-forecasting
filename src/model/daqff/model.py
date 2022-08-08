from torch import nn
from .layers import *
from pytorch_lightning import LightningModule
from utils.metrics import MultiMetrics


class DAQFFModel(LightningModule):
    """
    Args:
        x: Tensor (batch_size, n_stations1, seq_len, n_features)
    
    Returns:
        outputs: Tensor (batch_size, output_len)
    """
    def __init__(self, config, target_normalize_mean: float, target_normalize_std: float):
        super().__init__()

        self.extractor = Conv1dExtractor(config)
        self.st_layer = SpatialTemporalLayer(config, config["extractor_size"])

        self.linear1 = nn.Linear(config["lstm_output_size"], config["output_size"])

        self.target_normalize_mean = target_normalize_mean
        self.target_normalize_std = target_normalize_std
        self.metrics = MultiMetrics()
        self.optim_config = config["optim"]
        self.save_hyperparameters()

    def forward(
        self,
        x: torch.Tensor,
        src_locs: torch.Tensor,
        tar_loc: torch.Tensor,
        src_masks: torch.Tensor
    ):  
        features = self.extractor(x)
        features = self.st_layer(features, src_locs, tar_loc, src_masks)

        out = self.linear1(features)

        return out

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor):
        mape = (input - target).abs().sum(-1)

        return mape.mean()

    def training_step(self, batch, batch_idx):
        outputs = self(batch["features"], batch["src_locs"], batch["tar_loc"], batch["src_masks"])

        loss = self._compute_loss(outputs, batch["target"])

        return loss

    def training_epoch_end(self, outputs):
        loss = 0

        for step in outputs:
            loss += step["loss"].item()

        loss /= len(outputs)

        self.log("loss", loss)

    def validation_step(self, batch, batch_idx):
        # outputs.shape == (batch_size, n_pred_steps)
        outputs = self(batch["features"], batch["src_locs"], batch["tar_loc"], batch["src_masks"])

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