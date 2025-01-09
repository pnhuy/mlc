/* Linear Algebra C Library with Tensor Data Structure */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Define a structure for tensors */
typedef struct {
    size_t rows;
    size_t cols;
    double *data;
} Tensor;

double rand_double() { return (double)rand() / RAND_MAX; }

/* Function to create a tensor */
Tensor *create_tensor(size_t rows, size_t cols) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) {
        fprintf(stderr, "Memory allocation failed for tensor struct\n");
        exit(EXIT_FAILURE);
    }
    t->rows = rows;
    t->cols = cols;
    t->data = (double *)malloc(rows * cols * sizeof(double));
    if (!t->data) {
        fprintf(stderr, "Memory allocation failed for tensor data\n");
        free(t);
        exit(EXIT_FAILURE);
    }
    return t;
}

Tensor *rand_tensor(size_t rows, size_t cols) {
    Tensor *rt = create_tensor(rows, cols);
    size_t n = rows * cols;
    for (size_t i = 0; i < n; i++) {
        rt->data[i] = rand_double();
    }
    return rt;
}

/* Function to free a tensor */
void free_tensor(Tensor *t) {
    if (t) {
        free(t->data);
        free(t);
    }
}

void populate_array(double *array, Tensor *t) {
    size_t n = t->cols * t->rows;
    for (size_t i = 0; i < n; i++) {
        t->data[i] = *(array + i);
    }
}

/* Function to print a tensor */
void print_tensor(Tensor *t) {
    printf("[\n");
    for (size_t i = 0; i < t->rows; i++) {
        printf("\t");
        for (size_t j = 0; j < t->cols; j++) {
            printf("%lf ", t->data[i * t->cols + j]);
        }
        printf("\n");
    }
    printf("]\n");
}

/* Function to add two tensors, returns a new tensor */
Tensor *add_tensors(Tensor *a, Tensor *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Tensor dimensions do not match for addition\n");
        exit(EXIT_FAILURE);
    }
    Tensor *result = create_tensor(a->rows, a->cols);
    for (size_t i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

/* Function to multiply two tensors (matrix multiplication), returns a new tensor */
Tensor *multiply_tensors(Tensor *a, Tensor *b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "Tensor dimensions do not match for multiplication\n");
        exit(EXIT_FAILURE);
    }
    Tensor *result = create_tensor(a->rows, b->cols);
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            result->data[i * b->cols + j] = 0.0;
            for (size_t k = 0; k < a->cols; k++) {
                result->data[i * b->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
        }
    }
    return result;
}

/* Function to compute the dot product of two vectors (tensors with 1 row or column) */
double dot_product(Tensor *a, Tensor *b) {
    if (a->rows != 1 || b->rows != 1 || a->cols != b->cols) {
        fprintf(stderr, "Tensor sizes do not match for dot product\n");
        exit(EXIT_FAILURE);
    }
    double result = 0.0;
    for (size_t i = 0; i < a->cols; i++) {
        result += a->data[i] * b->data[i];
    }
    return result;
}

/* Function to transpose a tensor, returns a new tensor */
Tensor *transpose_tensor(Tensor *t) {
    Tensor *result = create_tensor(t->cols, t->rows);
    for (size_t i = 0; i < t->rows; i++) {
        for (size_t j = 0; j < t->cols; j++) {
            result->data[j * t->rows + i] = t->data[i * t->cols + j];
        }
    }
    return result;
}

Tensor *invert_tensor(Tensor *t) {
    if (t->rows != t->cols) {
        fprintf(stderr, "Matrix inversion is only defined for square matrices\n");
        exit(EXIT_FAILURE);
    }

    size_t n = t->rows;
    Tensor *augmented = create_tensor(n, 2 * n);
    
    // Initialize augmented matrix with [t | I]
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            augmented->data[i * 2 * n + j] = t->data[i * n + j]; // Copy original matrix
            augmented->data[i * 2 * n + (j + n)] = (i == j) ? 1.0 : 0.0; // Identity matrix
        }
    }

    // Perform Gauss-Jordan elimination
    for (size_t i = 0; i < n; i++) {
        // Find the pivot element
        double pivot = augmented->data[i * 2 * n + i];
        if (fabs(pivot) < 1e-9) {
            fprintf(stderr, "Matrix is singular and cannot be inverted\n");
            free(augmented->data);
            free(augmented);
            exit(EXIT_FAILURE);
        }

        // Normalize the pivot row
        for (size_t j = 0; j < 2 * n; j++) {
            augmented->data[i * 2 * n + j] /= pivot;
        }

        // Eliminate other rows
        for (size_t k = 0; k < n; k++) {
            if (k == i) continue;
            double factor = augmented->data[k * 2 * n + i];
            for (size_t j = 0; j < 2 * n; j++) {
                augmented->data[k * 2 * n + j] -= factor * augmented->data[i * 2 * n + j];
            }
        }
    }

    // Extract the inverse matrix
    Tensor *inverse = create_tensor(n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            inverse->data[i * n + j] = augmented->data[i * 2 * n + (j + n)];
        }
    }

    free(augmented->data);
    free(augmented);
    return inverse;
}
