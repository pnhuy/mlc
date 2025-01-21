#ifndef LINEAR_MODELS_H
#define LINEAR_MODELS_H

#include "tensor.h"
#include "la.h"

Tensor *solve_linear_regression(const Tensor *X, const Tensor *y);

#endif // LINEAR_MODELS_H