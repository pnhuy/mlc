#include "utils.h"
#include <math.h>
#include <stddef.h>

bool float_equal(float a, float b) {
    return fabs(a - b) < 1e-5;
}

// Helper function to calculate linear index from multi-dimensional indices
size_t tensor_get_linear_index(const size_t *shape, const size_t *idx, size_t ndim) {
    size_t linear_idx = 0;
    size_t stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        linear_idx += idx[i] * stride;
        stride *= shape[i];
    }
    return linear_idx;
}
