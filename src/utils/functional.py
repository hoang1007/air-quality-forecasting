import torch

def scale_softmax(x: torch.Tensor, dim: int = 0):
    EPS = 1e-8
    min_ = x.min(dim=dim, keepdim=True).values
    max_ = x.max(dim=dim, keepdim=True).values

    z = (x - min_) / (max_ - min_ + EPS)

    numerator = torch.exp(z)
    denominator = numerator.sum(dim=dim, keepdim=True)
    return numerator / denominator