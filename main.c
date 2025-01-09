#include "tensor.h"
#include "la.h"

int main() {
    printf("Hello\n");
    Tensor *X = tensor_create(2, 3, 2);
    copy_data((double[]){1, 2, 3, 4, 5, 6}, X);
    tensor_print(X);
    
    Tensor *Y = tensor_transpose(X);
    tensor_print(Y);

    return EXIT_SUCCESS;
}
