from __future__ import annotations

import torch


def get_neighbor_list_pbc(
    pos: torch.Tensor,
    cutoff: float,
    lattice: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """_summary_

    Get neighbor list within the cutoff sphere with periodic boundary condition.

    Args:
        pos (torch.Tensor): Positions of nodes shape of (n_node, dim).
        cutoff (float): The cutoff radius.
        lattice (torch.Tensor): Lattice vectors shape of (dim, dim).

    Returns:
        index_i (torch.Tensor): First node index shape of (n_edge).
        index_j (torch.Tensor): Second node index shape of (n_edge).
        shifts (torch.Tensor): Lattice shifts of node of `index_j` shape of (n_edge, dim).
    """
    pass
