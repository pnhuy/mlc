#include "tensor.h"

// Element-wise Operations
// These operate on tensors of the same shape
Tensor* tensor_add(const Tensor* a, const Tensor* b);
Tensor* tensor_subtract(const Tensor* a, const Tensor* b);
Tensor* tensor_multiply(const Tensor* a, const Tensor* b);  // Hadamard product
Tensor *tensor_divide(const Tensor *a, const Tensor *b);

// Scalar Operations
// These operate on a tensor and a single number
Tensor* tensor_add_scalar(const Tensor* a, float scalar);
Tensor* tensor_subtract_scalar(const Tensor* a, float scalar);
Tensor* tensor_multiply_scalar(const Tensor* a, float scalar);
Tensor *tensor_divide_scalar(const Tensor *a, float scalar);

// Linear Algebra Operations
Tensor* tensor_matmul(const Tensor* a, const Tensor* b);  // Matrix multiplication
float tensor_dot(const Tensor* a, const Tensor* b);     // Dot product
Tensor* tensor_cross(const Tensor* a, const Tensor* b);   // Cross product (3D vectors only)


// Reduction Operations
float tensor_sum(const Tensor* tensor);
float tensor_mean(const Tensor* tensor);
Tensor* tensor_sum_axis(const Tensor* tensor, size_t axis);
Tensor* tensor_mean_axis(const Tensor* tensor, size_t axis);
