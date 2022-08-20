import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from utils.metrics import MultiMetrics


class BaseAQFModel(LightningModule):
    def __init__(self, config, target_normalize_mean: float, target_normalize_std: float):
        super().__init__()

        self.target_normalize_mean = target_normalize_mean
        self.target_normalize_std = target_normalize_std
        self.optim_config = config["optim"]
        self.metric_fns = MultiMetrics()
        self.save_hyperparameters()

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor):
        loss = F.mse_loss(input, target, reduction="none")

        loss = loss.mean(-1).sqrt()

        loss = loss.sum() / loss.size(0)

        return loss

    def predict(self, dt):
        st = self.training
        self.train(False)
        with torch.no_grad():
            # add batch dim
            features = dt["features"].unsqueeze(0).to(self.device)
            src_locs = dt["src_locs"].unsqueeze(0).to(self.device)
            src_masks = dt["src_masks"].unsqueeze(0).to(self.device)

            output = self(features, src_locs, src_masks).squeeze(0)
            # inverse transforms
            output = output * self.target_normalize_std + self.target_normalize_mean

        self.train(st)
        return output

    def training_step(self, batch, batch_idx):
        outs = self(batch["features"], batch["src_locs"], batch["src_masks"])

        loss = self.compute_loss(outs, batch["src_nexts"])

        self.log("loss", loss.item())

        return loss

    def training_epoch_end(self, outputs):
        loss = 0

        for step in outputs:
            loss += step["loss"].item()

        loss /= len(outputs)

        self.log("epoch_loss", loss)

    def validation_step(self, batch, batch_idx):
        # pres.shape == (batch_size, n_src_stations, output_size)
        outs = self(batch["features"], batch["src_locs"], batch["src_masks"])

        outs = outs * self.target_normalize_std + self.target_normalize_mean

        mae = (outs - batch["src_nexts"]).abs().mean()

        return {"mae": mae}

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
        if self.optim_config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.optim_config["lr"],
                momentum=self.optim_config.sgd.momentum,
                weight_decay=self.optim_config.sgd["weight_decay"])
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optim_config["lr"],
                weight_decay=self.optim_config.adam["weight_decay"])
        
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