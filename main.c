#include "la.h"
#include "tensor.h"
#include "linear_models.h"

int main() {
    // Create input tensor X (4 samples, 2 features)
    Tensor *X = tensor_create(2, 4, 2);
    copy_data((Dtype[]){1, 1, 2, 1, 3, 1, 4, 1}, X);
    tensor_print(X);

    // Create output tensor Y (4 samples, 1 feature)
    Tensor *Y = tensor_create(2, 4, 1);
    copy_data((Dtype[]){2, 3, 4, 5}, Y);
    tensor_print(Y);

    // Perform linear regression
    Tensor *W = solve_linear_regression(X, Y);

    // Print the weights
    tensor_print(W);

    // Clean up
    tensor_free(X);
    tensor_free(Y);
    tensor_free(W);

    return EXIT_SUCCESS;
}
