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
    def __init__(self, flattened: bool = False):
        self._flattened = flattened

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if not self._flattened:
            y_pred = y_pred.reshape(y_pred.size(0), -1)
            y_true = y_true.reshape_as(y_pred)
        else:
            assert len(y_pred.shape) == 1, "not flattened"
        
        return {
            "mdape": mdape(y_pred, y_true).item(),
            "mae": mae(y_pred, y_true).item(),
            "mape": mape(y_pred, y_true).item(),
            "rmse": rmse(y_pred, y_true).item(),
            "r2": r2(y_pred, y_true).item(),
        }


def r2(y_pred: torch.Tensor, y: torch.Tensor):
    y_avg = y.sum(-1, keepdim=True) / y.size(-1)

    nom = (y - y_pred).pow(2).sum(-1)
    denom = (y - y_avg).pow(2).sum(-1)

    loss = 1 - nom / denom

    return loss.mean()

def mdape(y_pred: torch.Tensor, y: torch.Tensor):
    loss = ((y - y_pred) / y).abs().median(dim=-1).values

    return loss.mean()

def mape(y_pred: torch.Tensor, y: torch.Tensor):
    loss = ((y_pred - y) / y).abs()

    loss = loss.sum(-1) / loss.size(-1)

    return loss.mean()

def mae(y_pred: torch.Tensor, y: torch.Tensor):
    loss = (y_pred - y).abs()

    loss = loss.sum(-1) / loss.size(-1)

    return loss.mean()
    
def rmse(y_pred: torch.Tensor, y: torch.Tensor):
    err = (y_pred - y).pow(2)
    err = err.sum(-1) / err.size(-1)
    err = err.sqrt()

    return err.mean()