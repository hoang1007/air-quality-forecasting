import torch

def batch_gnn_input(x: torch.Tensor, edge_weights: torch.Tensor, edge_ids: torch.Tensor):
        num_groups = x.size(1)
        x = x.reshape(-1, x.size(-1))
        edge_weights = edge_weights.reshape(-1, edge_weights.size(-1))

        batch_ids = torch.arange(edge_ids.size(0)).type_as(edge_ids) * num_groups

        edge_ids = edge_ids + batch_ids[:, None, None]
        
        edge_ids = edge_ids.transpose(0, 1).reshape(2, -1)

        return x, edge_weights, edge_ids


def inverse_distance_weighting(loc1: torch.Tensor, loc2: torch.Tensor = None, beta: float = 1):
        """
        Args:
            loc1: (batch_size, num_loc1, 2)
            loc2: (batch_size, num_loc2, 2)
        
        Returns:
            weights: (batch_size, num_corr)
            ids: (batch_size, 2, num_corr)
        """

        if loc2 is None:
            return _self_weighting(loc1, beta=beta)
        else:
            return _pairwise_weighting(loc1, loc2, beta=beta)
        

def _self_weighting(locs: torch.Tensor, beta: float = 1):
    weights, ids = [], []
    num_locs = locs.size(1)

    for i in range(num_locs):
        for j in range(num_locs):
            if i != j:
                dist = (locs[:, i] - locs[:, j]).pow(2).sum(-1).sqrt()

                weights.append(dist.pow(-beta))
                ids.append(locs.new_tensor([i, j], dtype=torch.long))

    weights = torch.stack(weights, dim=-1)
    ids = torch.stack(ids, dim=-1).unsqueeze(0).repeat_interleave(locs.size(0), dim=0)

    return weights, ids


def _pairwise_weighting(loc1: torch.Tensor, loc2: torch.Tensor, beta: float = 1):
    nloc1 = loc1.size(1)
    nloc2 = loc2.size(1)
    weights, ids = [], []

    for i in range(nloc1):
        for j in range(nloc2):
                dist = (loc1[:, i] - loc2[:, j]).pow(2).sum(-1).sqrt()

                weights.append(dist.pow(-beta))
                ids.append(loc1.new_tensor([i, j], dtype=torch.long))

    weights = torch.stack(weights, dim=-1)
    ids = torch.stack(ids, dim=-1).unsqueeze(0).repeat_interleave(loc1.size(0), dim=0)

    return weights, ids