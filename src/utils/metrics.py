from typing import Dict
import torch
from pytorch_forecasting.metrics import MAE, MAPE, RMSE
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric


class Metrics:
    '''
    Tính các điểm đánh giá cho mô hình.

    Args:
        y_pred (Tensor): Giá trị dự đoán
        y_true (Tensor): Giá trị thật
        weight (Tensor or None): mảng các giá trị 1, 0. `1` đại diện cho giá trị hợp lệ

    Returns:
        outputs (Dict[float]): Các điểm `mae`, `mape`, `rmse`, `r2`, `mdape`
    '''

    def __init__(self):
        self.mae = MAE(reduction="mean")
        self.mape = MAPE(reduction="mean")
        self.rmse = RMSE(reduction="sqrt-mean")
        self.r2 = R2(reduction="mean")
        self.mdape = MDAPE(reduction="none")

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor = None):
        if weight is not None:
            y_pred = y_pred[weight]
            y_true = y_true[weight]

        return {
            "mdape": self.mdape(y_pred, y_true).item(),
            "mae": self.mae(y_pred, y_true).item(),
            "mape": self.mape(y_pred, y_true).item(),
            "rmse": self.rmse(y_pred, y_true).item(),
            "r2": self.r2(y_pred, y_true).item(),
        }


class R2(MultiHorizonMetric):
    def __init__(self, reduction="mean", **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def loss(self, y_pred: Dict[str, torch.Tensor], target):
        y_pred = self.to_prediction(y_pred)

        rss = torch.pow(target - y_pred, 2)
        tss = torch.pow(target - target.mean(dim=-1), 2)

        return 1 - rss / (tss + 1e-8)


class MDAPE(MultiHorizonMetric):
    def loss(self, y_pred: Dict[str, torch.Tensor], target):
        loss = (self.to_prediction(y_pred) - target).abs() / \
            (target + 1e-8).abs()

        return loss.median()