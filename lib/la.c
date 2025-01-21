#include "la.h"
#include "tensor.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

// Scalar Operations
// These operate on a tensor and a single number
Tensor *tensor_add_scalar(const Tensor *a, float scalar) {
    Tensor *result = tensor_copy(a);

    for (size_t i = 0; i < result->size; i++) {
        result->data[i] += scalar;
    }

    return result;
}

Tensor *tensor_subtract_scalar(const Tensor *a, float scalar) {
    return tensor_add_scalar(a, -scalar);
}

Tensor *tensor_multiply_scalar(const Tensor *a, float scalar) {
    Tensor *result = tensor_copy(a);

    for (size_t i = 0; i < result->size; i++) {
        result->data[i] *= scalar;
    }

    return result;
}

Tensor *tensor_divide_scalar(const Tensor *a, float scalar) {
    return tensor_multiply_scalar(a, 1.0 / scalar);
}

// Add two tensors
Tensor *tensor_add(const Tensor *t1, const Tensor *t2) {
    // Check if the tensors have the same number of dimensions
    if (t1->ndim != t2->ndim) {
        fprintf(stderr,
                "Error: Tensors must have the same number of dimensions.\n");
        return NULL;
    }

    // Check if the shapes match
    for (size_t i = 0; i < t1->ndim; i++) {
        if (t1->shape[i] != t2->shape[i]) {
            fprintf(stderr, "Error: Tensors must have the same shape.\n");
            return NULL;
        }
    }

    // Create a new tensor to store the result
    Tensor *result = tensor_create_from_shape(t1->ndim, t1->shape);
    // Perform element-wise addition
    for (size_t i = 0; i < result->size; i++) {
        result->data[i] = t1->data[i] + t2->data[i];
    }

    return result;
}

Tensor *tensor_subtract(const Tensor *t1, const Tensor *t2) {
    return tensor_add(t1, tensor_multiply_scalar(t2, -1));
}

Tensor *tensor_multiply(const Tensor *t1, const Tensor *t2) {
    // Check if the tensors have the same number of dimensions
    if (t1->ndim != t2->ndim) {
        fprintf(stderr,
                "Error: Tensors must have the same number of dimensions.\n");
        return NULL;
    }

    // Check if the shapes match
    for (size_t i = 0; i < t1->ndim; i++) {
        if (t1->shape[i] != t2->shape[i]) {
            fprintf(stderr, "Error: Tensors must have the same shape.\n");
            return NULL;
        }
    }

    // Create a new tensor to store the result
    Tensor *result = tensor_create_from_shape(t1->ndim, t1->shape);
    // Perform element-wise multiplication
    for (size_t i = 0; i < result->size; i++) {
        result->data[i] = t1->data[i] * t2->data[i];
    }

    return result;
}

Tensor *tensor_divide(const Tensor *t1, const Tensor *t2) {
    // Check if the tensors have the same number of dimensions
    if (t1->ndim != t2->ndim) {
        fprintf(stderr,
                "Error: Tensors must have the same number of dimensions.\n");
        return NULL;
    }

    // Check if the shapes match
    for (size_t i = 0; i < t1->ndim; i++) {
        if (t1->shape[i] != t2->shape[i]) {
            fprintf(stderr, "Error: Tensors must have the same shape.\n");
            return NULL;
        }
    }

    // Create a new tensor to store the result
    Tensor *result = tensor_create_from_shape(t1->ndim, t1->shape);
    // Perform element-wise division
    for (size_t i = 0; i < result->size; i++) {
        result->data[i] = t1->data[i] / t2->data[i];
    }

    return result;
}

// Matrix Multiplication
Tensor *tensor_matmul(const Tensor *t1, const Tensor *t2) {
    // Check if the tensors have the correct number of dimensions
    if (t1->ndim > 2 || t2->ndim > 2) {
        fprintf(stderr, "Error: Tensors must have 2 dimensions for matrix "
                        "multiplication.\n");
        return NULL;
    }

    // Check if the shapes are compatible for matrix multiplication
    if (t1->shape[1] != t2->shape[0]) {
        fprintf(stderr,
                "Error: Incompatible shapes for matrix multiplication.\n");
        return NULL;
    }

    // Create a new tensor to store the result
    size_t shape[] = {t1->shape[0], t2->shape[1]};
    Tensor *result = tensor_create_from_shape(2, shape);

    // Perform matrix multiplication
    for (size_t i = 0; i < t1->shape[0]; i++) {
        for (size_t j = 0; j < t2->shape[1]; j++) {
            result->data[i * result->shape[1] + j] = 0;
            for (size_t k = 0; k < t1->shape[1]; k++) {
                result->data[i * result->shape[1] + j] +=
                    t1->data[i * t1->shape[1] + k] *
                    t2->data[k * t2->shape[1] + j];
            }
        }
    }

    return result;
}

// Dot Product
float tensor_dot(const Tensor *t1, const Tensor *t2) {
    // Check if the tensors have the correct number of dimensions
    if (t1->ndim != 1 || t2->ndim != 1) {
        fprintf(stderr, "Error: Tensors must have 1 dimension for dot "
                        "product.\n");
        exit(EXIT_FAILURE);
    }

    // Check if the shapes are compatible for dot product
    if (t1->shape[0] != t2->shape[0]) {
        fprintf(stderr, "Error: Incompatible shapes for dot product.\n");
        exit(EXIT_FAILURE);
    }

    // Perform dot product
    float dot_product = 0;
    for (size_t i = 0; i < t1->shape[0]; i++) {
        dot_product += t1->data[i] * t2->data[i];
    }

    return dot_product;
}

Tensor *tensor_cross(const Tensor *t1, const Tensor *t2) {
    // Check if the tensors have the correct number of dimensions
    if (t1->ndim != 1 || t2->ndim != 1) {
        fprintf(stderr, "Error: Tensors must have 1 dimension for cross "
                        "product.\n");
        return NULL;
    }

    // Check if the shapes are compatible for cross product
    if (t1->shape[0] != 3 || t2->shape[0] != 3) {
        fprintf(stderr, "Error: Tensors must have shape (3,) for cross "
                        "product.\n");
        return NULL;
    }

    // Create a new tensor to store the result
    size_t shape[] = {3};
    Tensor *result = tensor_create_from_shape(1, shape);

    // Perform cross product
    result->data[0] = t1->data[1] * t2->data[2] - t1->data[2] * t2->data[1];
    result->data[1] = t1->data[2] * t2->data[0] - t1->data[0] * t2->data[2];
    result->data[2] = t1->data[0] * t2->data[1] - t1->data[1] * t2->data[0];

    return result;
}

Tensor *tensor_inverse(const Tensor *t) {
    // Check for valid input (must be 2D square matrix)
    if (!t || t->ndim != 2 || t->shape[0] != t->shape[1]) {
        return NULL;
    }

    size_t n = t->shape[0]; // Matrix dimension

    // Create augmented matrix [A|I] with shape [n, 2n]
    size_t aug_shape[] = {n, 2 * n};
    Tensor *augmented = tensor_create_from_shape(2, aug_shape);
    if (!augmented)
        return NULL;

    // Initialize augmented matrix [A|I]
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            // Copy original matrix to left half
            augmented->data[i * (2 * n) + j] = t->data[i * n + j];
            // Put identity matrix in right half
            augmented->data[i * (2 * n) + (j + n)] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Gauss-Jordan elimination
    for (size_t i = 0; i < n; i++) {
        // Find pivot
        float pivot = augmented->data[i * (2 * n) + i];

        // Check if matrix is singular
        if (fabsf(pivot) < 1e-10f) {
            tensor_free(augmented);
            return NULL;
        }

        // Scale pivot row
        for (size_t j = 0; j < 2 * n; j++) {
            augmented->data[i * (2 * n) + j] /= pivot;
        }

        // Eliminate column
        for (size_t k = 0; k < n; k++) {
            if (k != i) {
                float factor = augmented->data[k * (2 * n) + i];
                for (size_t j = 0; j < 2 * n; j++) {
                    augmented->data[k * (2 * n) + j] -=
                        factor * augmented->data[i * (2 * n) + j];
                }
            }
        }
    }

    // Create result tensor for inverse matrix
    size_t inv_shape[] = {n, n};
    Tensor *inverse = tensor_create_from_shape(2, inv_shape);
    if (!inverse) {
        tensor_free(augmented);
        return NULL;
    }

    // Extract inverse from right half of augmented matrix
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            inverse->data[i * n + j] = augmented->data[i * (2 * n) + (j + n)];
        }
    }

    // Clean up
    tensor_free(augmented);
    return inverse;
}

// Helper function to check if a matrix is invertible
bool tensor_is_invertible(const Tensor *t) {
    if (!t || t->ndim != 2 || t->shape[0] != t->shape[1]) {
        return false;
    }

    // For 2x2 matrices, check determinant
    if (t->shape[0] == 2) {
        float det = t->data[0] * t->data[3] - t->data[1] * t->data[2];
        return fabsf(det) > 1e-10f;
    }

    // For larger matrices, attempt inversion
    Tensor *inv = tensor_inverse(t);
    if (!inv)
        return false;

    tensor_free(inv);
    return true;
}

// Sum of all elements in a tensor
Dtype tensor_sum(const Tensor *t) {
    Dtype sum = 0;
    for (size_t i = 0; i < t->size; i++) {
        sum += t->data[i];
    }
    return sum;
}

// Mean of all elements in a tensor
Dtype tensor_mean(const Tensor *t) {
    return tensor_sum(t) / t->size;
}

// Max element in a tensor
Dtype tensor_max(const Tensor *t) {
    Dtype max = t->data[0];
    for (size_t i = 1; i < t->size; i++) {
        if (t->data[i] > max) {
            max = t->data[i];
        }
    }
    return max;
}

// Min element in a tensor
Dtype tensor_min(const Tensor *t) {
    Dtype min = t->data[0];
    for (size_t i = 1; i < t->size; i++) {
        if (t->data[i] < min) {
            min = t->data[i];
        }
    }
    return min;
}

// Argmax of a tensor
size_t tensor_argmax(const Tensor *t) {
    Dtype max = t->data[0];
    size_t argmax = 0;
    for (size_t i = 1; i < t->size; i++) {
        if (t->data[i] > max) {
            max = t->data[i];
            argmax = i;
        }
    }
    return argmax;
}

// Argmin of a tensor
size_t tensor_argmin(const Tensor *t) {
    Dtype min = t->data[0];
    size_t argmin = 0;
    for (size_t i = 1; i < t->size; i++) {
        if (t->data[i] < min) {
            min = t->data[i];
            argmin = i;
        }
    }
    return argmin;
}

Tensor *tensor_sum_axis(const Tensor *tensor, size_t axis) {
    if (axis >= tensor->ndim) {
        fprintf(stderr, "Error: Axis out of bounds.\n");
        return NULL;
    }

    // Calculate the shape of the result tensor
    size_t *result_shape = malloc((tensor->ndim - 1) * sizeof(size_t));
    if (!result_shape) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return NULL;
    }

    for (size_t i = 0, j = 0; i < tensor->ndim; i++) {
        if (i != axis) {
            result_shape[j++] = tensor->shape[i];
        }
    }

    // Create the result tensor
    Tensor *result = tensor_create_from_shape(tensor->ndim - 1, result_shape);
    free(result_shape);
    if (!result) {
        return NULL;
    }

    // Initialize the result tensor to zero
    for (size_t i = 0; i < result->size; i++) {
        result->data[i] = 0;
    }

    // Sum along the specified axis
    size_t stride = 1;
    for (size_t i = axis + 1; i < tensor->ndim; i++) {
        stride *= tensor->shape[i];
    }

    for (size_t i = 0; i < result->size; i++) {
        size_t offset = (i / stride) * tensor->shape[axis] * stride + (i % stride);
        for (size_t j = 0; j < tensor->shape[axis]; j++) {
            result->data[i] += tensor->data[offset + j * stride];
        }
    }

    return result;
}

Tensor *tensor_mean_axis(const Tensor *tensor, size_t axis) {
    Tensor *sum = tensor_sum_axis(tensor, axis);
    if (!sum) {
        return NULL;
    }

    Tensor *mean = tensor_multiply_scalar(sum, 1.0 / tensor->shape[axis]);
    tensor_free(sum);

    return mean;
}
