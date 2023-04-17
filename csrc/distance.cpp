#include <torch/extension.h>
#include <vector>
#include "cpu/distance/euclidian_cpu.h"

#ifdef WITH_CUDA
#include "cuda/distance/euclidian_cuda.h"
#endif

#ifdef WITH_PYTHON
#include <Python.h>
#endif

using namespace torch::indexing;
using namespace torch::autograd;

tensor_list euclidian_pbc_fw(torch::Tensor x, torch::Tensor y, torch::Tensor lattice, torch::Tensor shifts)
{
    if (x.device().is_cuda())
    {
#ifdef WITH_CUDA
        return euclidian_cuda(x, y, lattice, shifts);
#else
        AT_ERROR("Not compiled with CUDA support");
#endif
    }
    else
    {
        return euclidian_cpu(x, y, lattice, shifts);
    }
}

class EuclidianPBC : public Function<EuclidianPBC>
{
public:
    static std::vector<at::Tensor> forward(
        AutogradContext *ctx, torch::Tensor pos, torch::Tensor ind_i, torch::Tensor ind_j,
        torch::Tensor lattice, torch::Tensor shifts, torch::Tensor batch_ind)
    {
        ctx->save_for_backward({pos, ind_i, ind_j});
        auto out = euclidian_pbc_fw(
            pos.index_select(0, ind_i), pos.index_select(0, ind_j),
            lattice.index_select(0, batch_ind.index_select(0, ind_i)), shifts);
        ctx->save_for_backward({out[0], out[1]});
        return {out[1]};
    }

    static std::vector<torch::Tensor> backward(
        AutogradContext *ctx, tensor_list grad_output)
    {
        auto saved = ctx->get_saved_variables();
        auto pos = saved[0];
        auto ind_i = saved[1];
        auto ind_j = saved[2];
        auto vec = saved[3];
        auto dist = saved[4];

        auto grad_out = grad_output[0];
        auto grad_vec = grad_out.unsqueeze_(-1) * vec / dist.unsqueeze_(-1);
        auto grad_pos = torch::zeros_like(pos).index_add(0, ind_j, grad_vec) + torch::zeros_like(pos).index_add(0, ind_i, -grad_vec);

        return {grad_pos, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

torch::Tensor euclidian_pbc(torch::Tensor pos, torch::Tensor ind_i, torch::Tensor ind_j,
                            torch::Tensor lattice, torch::Tensor shifts, torch::Tensor batch_ind)
{
    // return EuclidianPBC::apply(pos, ind_i, ind_j, lattice, shifts, batch_ind)[0];
    auto vec = pos.index_select(0, ind_j) - pos.index_select(0, ind_j) + torch::einsum("ni,nij->nj", {shifts, lattice.index_select(0, batch_ind.index_select(0, ind_i))});
    auto dist = torch::norm(vec, 2, 1);
    return dist;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Calculation of distances in systems with periodic boundary conditions.";
    m.def("euclidian_pbc", &euclidian_pbc, "euclidian distance with pbc");
}
