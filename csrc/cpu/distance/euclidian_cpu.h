#pragma once

#include "../../extensions.h"

torch::autograd::tensor_list euclidian_cpu(
    torch::Tensor x, torch::Tensor y,
    torch::Tensor lattice, torch::Tensor shifts);