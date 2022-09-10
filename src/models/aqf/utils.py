import torch


def inverse_distance_weighting(
    loc1: torch.Tensor, loc2: torch.Tensor = None,
    dist_thresh: float = None, beta: float = 1,
    norm: bool = False
):
    """
    Args:
        loc1: (num_loc1, 2)
        loc2: (num_loc2, 2)

    Returns:
        weights: (E)
        ids: (2, E)
    """

    if loc2 is None:
        weights, ids = _self_weighting(
            loc1, beta=beta, dist_thresh=dist_thresh, norm=norm)
    else:
        weights, ids = _pairwise_weighting(
            loc1, loc2, beta=beta, dist_thresh=dist_thresh, norm=norm)

    return weights, ids


def _self_weighting(locs: torch.Tensor, beta: float = 1, dist_thresh: float = None, norm: bool = False):
    weights, ids = locs.new_tensor([]), []
    num_locs = locs.size(0)

    for i in range(num_locs):
        row_w = []
        for j in range(num_locs):
            if i != j:
                dist = (locs[i] - locs[j]).pow(2).sum(-1).sqrt()

                if dist_thresh is not None and dist > dist_thresh:
                    continue

                row_w.append(dist.pow(-beta))
                ids.append(locs.new_tensor([i, j], dtype=torch.long))
        row_w = locs.new_tensor(row_w, dtype=torch.float32) # shape (V)
        if norm:
            row_w = row_w / row_w.sum()
        
        weights = torch.hstack((weights, row_w))
        
    ids = torch.stack(ids, dim=-1)

    return weights, ids


def _pairwise_weighting(loc1: torch.Tensor, loc2: torch.Tensor, beta: float = 1, dist_thresh: float = None, norm: bool = False):
    nloc1 = loc1.size(0)
    nloc2 = loc2.size(0)
    weights, ids = loc1.new_tensor([]), []

    for i in range(nloc1):
        row_w = []
        for j in range(nloc2):
            dist = (loc1[i] - loc2[j]).pow(2).sum(-1).sqrt()

            if dist_thresh is not None and dist > dist_thresh:
                continue
    
            row_w.append(dist.pow(-beta))
            ids.append(loc1.new_tensor([i, j], dtype=torch.long))
        row_w = loc1.new_tensor(row_w, dtype=torch.float32) # shape (V)
        if norm:
            row_w = row_w / row_w.sum()
        
        weights = torch.hstack((weights, row_w))
    ids = torch.stack(ids, dim=-1)

    return weights, ids
