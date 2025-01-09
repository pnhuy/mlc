// tensor.h - Header file for general tensor operations
#ifndef TENSOR_H
#define TENSOR_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

// General Tensor structure
typedef float Dtype;

typedef struct {
    Dtype *data;  // Pointer to flattened data
    size_t *shape; // Array storing dimensions
    size_t ndim;   // Number of dimensions
    size_t size;   // Total number of elements (product of shape)
} Tensor;

// Function prototypes
Tensor *tensor_create(size_t ndim, ...);
Tensor *tensor_create_from_shape(size_t ndim, size_t shape[]);
Tensor *tensor_copy(const Tensor *t);
void tensor_populate_array(Tensor *t, Dtype array[]);
Tensor *tensor_reshape(const Tensor *t, size_t ndim, size_t shape[]);
Tensor *tensor_transpose(const Tensor *tensor);

void tensor_free(Tensor *t);
void tensor_print(const Tensor *t);


void copy_data(float array[], Tensor *t);

#endif // TENSOR_H
