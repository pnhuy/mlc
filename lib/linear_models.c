#include "linear_models.h"

Tensor *solve_linear_regression(const Tensor *X, const Tensor *y) {
    // Check if X and Y have the same number of samples
    if (X->shape[0] != y->shape[0]) {
        fprintf(stderr, "Number of samples in X and Y do not match\n");
        return NULL;
    }

    // Perform linear regression
    Tensor *Xt = tensor_transpose(X);
    Tensor *XtX = tensor_matmul(Xt, X);
    Tensor *XtX_inv = tensor_inverse(XtX);
    Tensor *XtY = tensor_matmul(Xt, y);
    Tensor *W = tensor_matmul(XtX_inv, XtY);

    // Clean up
    tensor_free(Xt);
    tensor_free(XtX);
    tensor_free(XtX_inv);
    tensor_free(XtY);

    return W;
}