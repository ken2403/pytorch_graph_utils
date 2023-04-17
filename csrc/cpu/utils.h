#pragma once

#include "../extensions.h"

#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")