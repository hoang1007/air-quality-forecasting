import torch
import math
from utils.functional import euclidean_distance, haversine_distance


def inverse_distance_weighting(
    loc1: torch.Tensor, loc2: torch.Tensor = None,
    dist_thresh: float = None,
    dist_type: str = "euclidean",
    beta: float = 1,
    norm: bool = False
):
    """
    Args:
        loc1: (num_loc1, 2)
        loc2: (num_loc2, 2)
        dist_thresh: if the distance between two locations is greater than the threshold, it will be ignored.
        dist_type: Euclidean distance or Haversine distance
        beta: Inverse Distance weighting factor
        norm: If True, the normalized distance is used.

    Returns:
        weights: (E)
        ids: (2, E)
    """

    if loc2 is None:
        weights, ids = _self_weighting(
            loc1, beta=beta, dist_thresh=dist_thresh, norm=norm, dist_type=dist_type)
    else:
        weights, ids = _pairwise_weighting(
            loc1, loc2, beta=beta, dist_thresh=dist_thresh, norm=norm, dist_type=dist_type)

    return weights, ids


def _self_weighting(locs: torch.Tensor, beta: float, dist_thresh: float, dist_type: str, norm: bool):
    weights, ids = locs.new_tensor([]), []
    num_locs = locs.size(0)

    for i in range(num_locs):
        row_w = []
        for j in range(num_locs):
            if i != j:
                if dist_type == 'haversine':
                    dist = haversine_distance(locs[i], locs[j])
                elif dist_type == 'euclidean':
                    dist = euclidean_distance(locs[i], locs[j])
                else:
                    raise NotImplementedError(
                        f"Don't have {dist_type} distance")

                if dist_thresh is not None and dist > dist_thresh:
                    continue

                row_w.append(math.pow(dist, -beta))
                ids.append(locs.new_tensor([i, j], dtype=torch.long))
        row_w = locs.new_tensor(row_w, dtype=torch.float32)  # shape (V)
        if norm:
            row_w = row_w / row_w.sum()

        weights = torch.hstack((weights, row_w))

    ids = torch.stack(ids, dim=-1)

    return weights, ids


def _pairwise_weighting(loc1: torch.Tensor, loc2: torch.Tensor, beta: float, dist_thresh: float, dist_type: str, norm: bool):
    nloc1 = loc1.size(0)
    nloc2 = loc2.size(0)
    weights, ids = loc1.new_tensor([]), []

    for i in range(nloc1):
        row_w = []
        for j in range(nloc2):
            if dist_type == 'haversine':
                dist = haversine_distance(loc1[i], loc2[j])
            elif dist_type == 'euclidean':
                dist = euclidean_distance(loc1[i], loc2[j])
            else:
                raise NotImplementedError(f"Don't have {dist_type} distance")

            if dist_thresh is not None and dist > dist_thresh:
                continue

            row_w.append(math.pow(dist, -beta))
            ids.append(loc1.new_tensor([i, j], dtype=torch.long))
        row_w = loc1.new_tensor(row_w, dtype=torch.float32)  # shape (V)
        if norm:
            row_w = row_w / row_w.sum()

        weights = torch.hstack((weights, row_w))
    ids = torch.stack(ids, dim=-1)

    return weights, ids
