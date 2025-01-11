#include <stdbool.h>
#include <stddef.h>

bool float_equal(float a, float b);
size_t tensor_get_linear_index(const size_t *shape, const size_t *idx, size_t ndim);
