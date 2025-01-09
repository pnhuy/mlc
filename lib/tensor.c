#include "tensor.h"
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Function to create a tensor with arbitrary shape
Tensor *tensor_create(size_t ndim, ...) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = (size_t *)malloc(ndim * sizeof(size_t));

    va_list args;
    va_start(args, ndim);
    t->size = 1;
    for (size_t i = 0; i < ndim; i++) {
        t->shape[i] = va_arg(args, size_t);
        t->size *= t->shape[i];
    }
    va_end(args);

    t->data = (Dtype *)malloc(t->size * sizeof(Dtype));
    return t;
}

// Create a tensor from shape
Tensor *tensor_create_from_shape(size_t ndim, size_t shape[]) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = (size_t *)malloc(ndim * sizeof(size_t));
    t->size = 1;

    for (size_t i = 0; i < (t->ndim); i++) {
        t->shape[i] = *(shape + i);
        t->size *= *(shape + i);
    }

    t->data = (Dtype *)malloc(t->size * sizeof(Dtype));
    return t;
}

// Copy a tensor
Tensor *tensor_copy(const Tensor *t) {
    Tensor *copy = (Tensor *)malloc(sizeof(Tensor));
    copy->ndim = t->ndim;
    copy->shape = (size_t *)malloc(t->ndim * sizeof(size_t));
    copy->size = t->size;

    for (size_t i = 0; i < t->ndim; i++) {
        copy->shape[i] = t->shape[i];
    }

    copy->data = (Dtype *)malloc(t->size * sizeof(Dtype));
    for (size_t i = 0; i < t->size; i++) {
        copy->data[i] = t->data[i];
    }

    return copy;
}

// Populate a tensor with data from an array
void tensor_populate_array(Tensor *t, Dtype array[]) {
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = array[i];
    }
}

// Reshape a tensor
Tensor *tensor_reshape(const Tensor *t, size_t ndim, size_t shape[]) {
    Tensor *reshaped = tensor_copy(t);
    size_t new_size = 1;
    for (size_t i = 0; i < ndim; i++) {
        new_size *= shape[i];
    }

    if (new_size != reshaped->size) {
        fprintf(stderr,
                "Error: New shape must have the same number of elements.\n");

        tensor_free(reshaped);
        return NULL;
    }

    reshaped->ndim = ndim;
    free(reshaped->shape);
    reshaped->shape = (size_t *)malloc(ndim * sizeof(size_t));
    for (size_t i = 0; i < ndim; i++) {
        reshaped->shape[i] = shape[i];
    }

    return reshaped;
}

// Transpose a tensor
Tensor *tensor_transpose(const Tensor *tensor) {
    if (!tensor || tensor->ndim < 2) {
        return NULL; // Handle invalid input
    }

    // Create new dimensions array with reversed order
    size_t *new_dims = malloc(tensor->ndim * sizeof(size_t));
    if (!new_dims)
        return NULL;

    // Reverse the dimensions
    for (size_t i = 0; i < tensor->ndim; i++) {
        new_dims[i] = tensor->shape[tensor->ndim - 1 - i];
    }

    // Create new tensor with transposed dimensions
    Tensor *transposed = tensor_create_from_shape(tensor->ndim, new_dims);
    free(new_dims); // Clean up temporary array

    if (!transposed)
        return NULL;

    // Calculate strides for both tensors
    size_t *original_strides = malloc(tensor->ndim * sizeof(size_t));
    size_t *new_strides = malloc(tensor->ndim * sizeof(size_t));

    if (!original_strides || !new_strides) {
        free(original_strides);
        free(new_strides);
        tensor_free(transposed);
        return NULL;
    }

    // Calculate strides for original tensor
    original_strides[tensor->ndim - 1] = 1;
    for (int i = tensor->ndim - 2; i >= 0; i--) {
        original_strides[i] = original_strides[i + 1] * tensor->shape[i + 1];
    }

    // Calculate strides for transposed tensor
    new_strides[tensor->ndim - 1] = 1;
    for (int i = tensor->ndim - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * transposed->shape[i + 1];
    }

    // Iterate through all elements using a counter
    size_t total_elements = tensor->size;
    for (size_t count = 0; count < total_elements; count++) {
        // Convert linear index to multi-dimensional indices for original tensor
        size_t remaining = count;
        size_t *indices = malloc(tensor->ndim * sizeof(size_t));

        if (!indices) {
            free(original_strides);
            free(new_strides);
            tensor_free(transposed);
            return NULL;
        }

        for (size_t i = 0; i < tensor->ndim; i++) {
            indices[i] = remaining / original_strides[i];
            remaining %= original_strides[i];
        }

        // Calculate transposed position
        size_t transposed_pos = 0;
        for (size_t i = 0; i < tensor->ndim; i++) {
            // Use reversed indices for transposition
            transposed_pos += indices[tensor->ndim - 1 - i] * new_strides[i];
        }

        // Copy the data to its transposed position
        transposed->data[transposed_pos] = tensor->data[count];
        free(indices);
    }

    free(original_strides);
    free(new_strides);
    return transposed;
}

// Copy data from array to tensor
void copy_data(Dtype array[], Tensor *t) {
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = *(array + i);
    }
}

// Function to free the tensor
void tensor_free(Tensor *t) {
    free(t->data);
    free(t->shape);
    free(t);
}

// Function to print the tensor (for debugging purposes)
void tensor_print(const Tensor *t) {
    printf("===\nTensor with %zu dimensions:\n", t->ndim);
    for (size_t i = 0; i < t->ndim; i++) {
        printf("Dimension %zu: %zu\n", i, t->shape[i]);
    }
    printf("Data:\n");
    for (size_t i = 0; i < (t->size > 10 ? 10 : t->size); i++) {
        printf("%f ", t->data[i]);
    }
    printf("\n===\n");
}
