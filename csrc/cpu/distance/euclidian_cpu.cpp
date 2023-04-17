#include "euclidian_cpu.h"

#include "../utils.h"

using namespace torch::indexing;

torch::autograd::tensor_list
euclidian_cpu(torch::Tensor x, torch::Tensor y, torch::Tensor lattice, torch::Tensor shifts)
{
    CHECK_CPU(x);
    CHECK_CPU(y);
    CHECK_CPU(lattice);
    CHECK_CPU(shifts);

    x = x.contiguous();
    y = y.contiguous();
    lattice = lattice.contiguous();
    shifts = shifts.contiguous();

    auto vec = torch::empty({x.size(0), x.size(1)}, x.options());
    vec = vec.contiguous();
    auto dist = torch::empty({x.size(0)}, x.options());
    dist = dist.contiguous();
    for (auto n = 0; n < shifts.size(0); n++)
    {
        for (auto i = 0; i < lattice.size(1); i++)
        {
            vec[n][i] = y[n][i] - x[n][i] + (shifts[n] * lattice.index({n, Slice(), i})).sum();
            dist[n] += vec[n][i] * vec[n][i];
        }
    }
    dist = dist.sqrt();

    return {vec, dist};
}
