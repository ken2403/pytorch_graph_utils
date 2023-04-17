from __future__ import annotations

import torch

if torch.cuda.is_available():
    import torch_graph_utils._distance_cuda as distance
else:
    import torch_graph_utils._distance_cpu as distance


def euclidean(
    pos: torch.Tensor,
    index_i: torch.Tensor,
    index_j: torch.Tensor,
) -> torch.Tensor:
    r"""
    calculate euclidean distance between two nodes.

    Args:
        pos (torch.Tensor): Positions of nodes shape of (n_node, dim).
        index_i (torch.Tensor): First node index shape of (n_edge).
        index_j (torch.Tensor): Second node index shape of (n_edge).

    Returns:
        distances (torch.Tensor): Euclidean distances between two nodes shape of (n_edge).
    """
    edge_vec = pos[index_j] - pos[index_i]
    return torch.norm(edge_vec, dim=1)


def euclidean_distance_pbc(
    pos: torch.Tensor,
    index_i: torch.Tensor,
    index_j: torch.Tensor,
    lattice: torch.Tensor,
    shifts: torch.Tensor,
    batch_index: torch.Tensor,
) -> torch.Tensor:
    r"""
    calculate euclidean distance between two nodes with periodic boundary condition.

    Args:
        pos (torch.Tensor): Positions of nodes shape of (n_node, dim).
        index_i (torch.Tensor): First node index shape of (n_edge).
        index_j (torch.Tensor): Second node index shape of (n_edge).
        lattice (torch.Tensor): Lattice vectors shape of (n_batch, dim, dim).
        shifts (torch.Tensor): Lattice shifts of node of `index_j` shape of (n_edge, dim).
        batch_index (torch.Tensor): Batch index of nodes shape of (n_node).

    Returns:
        distances (torch.Tensor): Euclidean distances between two nodes shape of (n_edge).
    """
    return distance.euclidian_pbc(pos, index_i, index_j, lattice, shifts, batch_index)
