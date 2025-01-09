#include "la.h"
#include "tensor.h"
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
    tensor_print(t1);
    tensor_print(t2);
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
