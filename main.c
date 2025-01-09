#include "la.h"

int main() {
    Tensor *X = create_tensor(5, 2);
    Tensor *y = create_tensor(5, 1);
    populate_array((double[10]){1, 2, 1, 4, 1, 6, 1, 8, 1, 10}, X);
    populate_array((double[5]){5, 9, 13, 17, 21}, y);
    Tensor *w = rand_tensor(2, 1);

    print_tensor(X);
    print_tensor(y);
    print_tensor(w);

    // solve lr
    Tensor *X_T = transpose_tensor(X);
    Tensor *w_hat = multiply_tensors(invert_tensor(multiply_tensors(X_T, X)),
                                     multiply_tensors(X_T, y));
    print_tensor(w_hat);

    return EXIT_SUCCESS;
}
