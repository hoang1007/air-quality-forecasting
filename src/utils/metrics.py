import torch


class MultiMetrics:
    '''
    Tính các điểm đánh giá cho mô hình.

    Args:
        y_pred (Tensor): Giá trị dự đoán
        y_true (Tensor): Giá trị thật
        weight (Tensor or None): mảng các giá trị 1, 0. `1` đại diện cho giá trị hợp lệ

    Returns:
        outputs (Dict[float]): Các điểm `mae`, `mape`, `rmse`, `r2`, `mdape`
    '''
    def __init__(self, flattened: bool = True):
        self._flattened = flattened

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor,):
        if self._flattened:
            assert len(y_pred.shape)

        return {
            "mdape": mdape(y_pred, y_true),
            "mae": mae(y_pred, y_true),
            "mape": mape(y_pred, y_true),
            "rmse": rmse(y_pred, y_true),
            "r2": r2(y_pred, y_true),
        }


def r2(y_pred: torch.Tensor, y: torch.Tensor):
    nom = (y - y_pred).pow(2).sum().item()
    denom = (y - y.mean()).pow(2).sum().item()

    return nom / denom

def mdape(y_pred: torch.Tensor, y: torch.Tensor):
    return ((y - y_pred) / y).abs().median()

def mape(y_pred: torch.Tensor, y: torch.Tensor):
    v = ((y_pred - y) / y).abs()

    v = v.sum() / v.size(0)

    return v.item()

def mae(y_pred: torch.Tensor, y: torch.Tensor):
    err = (y_pred - y).abs()

    err = err.sum() / err.size(0)

    return err.item()

def rmse(y_pred: torch.Tensor, y: torch.Tensor):
    err = (y_pred - y).pow(2)
    err = err.sum() / err.size(0)
    err = err.sqrt()

    return err.item()