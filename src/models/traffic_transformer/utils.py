import torch


def inverse_distance_weighting(loc1: torch.Tensor, loc2: torch.Tensor = None, beta: float = 1, norm: bool = True):
        """
        Args:
            loc1: (batch_size, num_loc1, 2)
            loc2: (batch_size, num_loc2, 2) if None compute loc1 - loc1
            beta: float, control weighting factor
            norm: bool

        Returns:
            weights: (batch_size, num_corr)
            ids: (batch_size, 2, num_corr)
        """

        if loc2 is None:
            return _self_weighting(loc1, beta=beta, norm=norm)
        else:
            return _pairwise_weighting(loc1, loc2, beta=beta, norm=norm)
        

def _self_weighting(locs: torch.Tensor, beta: float = 1, norm: bool = True):
    weights, ids = [], []
    num_locs = locs.size(1)

    for i in range(num_locs):
        rw = []
        for j in range(num_locs):
            if i != j:
                # dist.shape == (batch_size)
                dist = (locs[:, i] - locs[:, j]).pow(2).sum(-1).sqrt()

                rw.append(dist.pow(-beta))
                ids.append(locs.new_tensor([i, j], dtype=torch.long))
        
        rw = torch.stack(rw, dim=-1)
        if norm:
            rw = rw / rw.sum(-1, keepdim=True)
        weights.append(rw)

    weights = torch.stack(weights, dim=1).flatten(start_dim=1)
    ids = torch.stack(ids, dim=-1).unsqueeze(0).repeat_interleave(locs.size(0), dim=0)

    return weights, ids


def _pairwise_weighting(loc1: torch.Tensor, loc2: torch.Tensor, beta: float = 1, norm: bool = True):
    nloc1 = loc1.size(1)
    nloc2 = loc2.size(1)
    weights, ids = [], []

    for i in range(nloc1):
        rw = []
        for j in range(nloc2):
                dist = (loc1[:, i] - loc2[:, j]).pow(2).sum(-1).sqrt()

                rw.append(dist.pow(-beta))
                ids.append(loc1.new_tensor([i, j], dtype=torch.long))
        
        rw = torch.tensor(rw, dtype=torch.float32)
        if norm:
            rw = rw / rw.sum(-1, keepdim=True)
        weights.append(rw)

    weights = torch.stack(weights, dim=1).flatten(start_dim=1)
    ids = torch.stack(ids, dim=-1).unsqueeze(0).repeat_interleave(loc1.size(0), dim=0)

    return weights, ids