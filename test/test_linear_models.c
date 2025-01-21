#include <tensor.h>
#include <linear_models.h>
#include <utils.h>
#include <assert.h>

void test_solve_linear_regression() {
    // Create input tensor X (4 samples, 2 features)
    Tensor *X = tensor_create(2, 4, 2);
    copy_data((Dtype[]){1, 1, 2, 1, 3, 1, 4, 1}, X);

    // Create output tensor Y (4 samples, 1 feature)
    Tensor *Y = tensor_create(2, 4, 1);
    copy_data((Dtype[]){2, 3, 4, 5}, Y);

    // Perform linear regression
    Tensor *W = solve_linear_regression(X, Y);

    assert(float_equal(W->data[0], 1.0));
    assert(float_equal(W->data[1], 1.0));

    // Clean up
    tensor_free(X);
    tensor_free(Y);
    tensor_free(W);

    printf("Linear regression test passed\n");
}

int main() {
    test_solve_linear_regression();
    return 0;
}