import torch


def scale_softmax(x: torch.Tensor, dim: int = 0):
    EPS = 1e-8
    min_ = x.min(dim=dim, keepdim=True).values
    max_ = x.max(dim=dim, keepdim=True).values

    z = (x - min_) / (max_ - min_ + EPS)

    numerator = torch.exp(z)
    denominator = numerator.sum(dim=dim, keepdim=True)
    return numerator / denominator

def euclidean_distances(
    src_locs: torch.Tensor,
    tar_locs: torch.Tensor):

    assert len(src_locs.shape) == 3 and len(tar_locs.shape) == 3,\
        "locs must be of shape [batch_size, N, 2]"

    deltas = src_locs.unsqueeze(2) - tar_locs.unsqueeze(1)
    dists = deltas.pow(2).sum(dim=-1).sqrt()

    return dists

def inverse_distance_weighted(
    src_locs: torch.Tensor,
    tar_locs: torch.Tensor,
    beta: float,
    eps: float = 1e-8):
    """
    Compute inverse distance weighted

    Args:
        src_locs: Tensor (batch_size, N, 2)
        tar_locs: Tensor (batch_size, M, 2)
        beta: Controlling factor. Float
        eps: smoothing computation factor

    Returns:
        weights: Tensor (batch_size, N, M)
    """

    assert len(src_locs.shape) == 3 and len(tar_locs.shape) == 3,\
        "locs must be of shape [batch_size, N, 2]"
    
    # dists.shape == (batch_size, N, M)
    dists = euclidean_distances(src_locs, tar_locs)

    weights = torch.pow(dists + eps, -beta)
    weights = weights / weights.sum(1, keepdim=True)

    return weights