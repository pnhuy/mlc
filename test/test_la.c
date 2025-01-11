#include "la.h"
#include "tensor.h"
#include "utils.h"
#include <assert.h>

void test_element_wise_operations() {
    // Create two 2x2 tensors
    Tensor *a = tensor_create(2, 2, 2);
    Tensor *b = tensor_create(2, 2, 2);
    float a_data[] = {1.0, 2.0, 3.0, 4.0};
    float b_data[] = {2.0, 3.0, 4.0, 5.0};
    tensor_populate_array(a, a_data);
    tensor_populate_array(b, b_data);

    // Test addition
    Tensor *sum = tensor_add(a, b);
    assert(float_equal(sum->data[0], 3.0)); // 1 + 2
    assert(float_equal(sum->data[1], 5.0)); // 2 + 3
    assert(float_equal(sum->data[2], 7.0)); // 3 + 4
    assert(float_equal(sum->data[3], 9.0)); // 4 + 5

    // Test subtraction
    Tensor *diff = tensor_subtract(a, b);
    assert(float_equal(diff->data[0], -1.0)); // 1 - 2
    assert(float_equal(diff->data[1], -1.0)); // 2 - 3
    assert(float_equal(diff->data[2], -1.0)); // 3 - 4
    assert(float_equal(diff->data[3], -1.0)); // 4 - 5

    // Test Hadamard product
    Tensor *prod = tensor_multiply(a, b);
    assert(float_equal(prod->data[0], 2.0));  // 1 * 2
    assert(float_equal(prod->data[1], 6.0));  // 2 * 3
    assert(float_equal(prod->data[2], 12.0)); // 3 * 4
    assert(float_equal(prod->data[3], 20.0)); // 4 * 5

    // Test division
    Tensor *div = tensor_divide(a, b);
    assert(float_equal(div->data[0], 0.5));       // 1 / 2
    assert(float_equal(div->data[1], 2.0 / 3.0)); // 2 / 3
    assert(float_equal(div->data[2], 0.75));      // 3 / 4
    assert(float_equal(div->data[3], 0.8));       // 4 / 5

    tensor_free(a);
    tensor_free(b);
    tensor_free(sum);
    tensor_free(diff);
    tensor_free(prod);
    tensor_free(div);
}

void test_scalar_operations() {
    // Create a 2x2 tensor
    Tensor *a = tensor_create(2, 2, 2);
    float a_data[] = {1.0, 2.0, 3.0, 4.0};
    tensor_populate_array(a, a_data);
    float scalar = 2.0;

    // Test scalar addition
    Tensor *sum = tensor_add_scalar(a, scalar);
    assert(float_equal(sum->data[0], 3.0)); // 1 + 2
    assert(float_equal(sum->data[1], 4.0)); // 2 + 2
    assert(float_equal(sum->data[2], 5.0)); // 3 + 2
    assert(float_equal(sum->data[3], 6.0)); // 4 + 2

    // Test scalar subtraction
    Tensor *diff = tensor_subtract_scalar(a, scalar);
    assert(float_equal(diff->data[0], -1.0)); // 1 - 2
    assert(float_equal(diff->data[1], 0.0));  // 2 - 2
    assert(float_equal(diff->data[2], 1.0));  // 3 - 2
    assert(float_equal(diff->data[3], 2.0));  // 4 - 2

    // Test scalar multiplication
    Tensor *prod = tensor_multiply_scalar(a, scalar);
    assert(float_equal(prod->data[0], 2.0)); // 1 * 2
    assert(float_equal(prod->data[1], 4.0)); // 2 * 2
    assert(float_equal(prod->data[2], 6.0)); // 3 * 2
    assert(float_equal(prod->data[3], 8.0)); // 4 * 2

    // Test scalar division
    Tensor *div = tensor_divide_scalar(a, scalar);
    assert(float_equal(div->data[0], 0.5)); // 1 / 2
    assert(float_equal(div->data[1], 1.0)); // 2 / 2
    assert(float_equal(div->data[2], 1.5)); // 3 / 2
    assert(float_equal(div->data[3], 2.0)); // 4 / 2

    tensor_free(a);
    tensor_free(sum);
    tensor_free(diff);
    tensor_free(prod);
    tensor_free(div);
}

void test_linear_algebra_operations() {
    // Test matrix multiplication
    Tensor *a = tensor_create(2, 2, 3); // 2x3 matrix
    Tensor *b = tensor_create(2, 3, 2); // 3x2 matrix
    float a_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float b_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    tensor_populate_array(a, a_data);
    tensor_populate_array(b, b_data);

    Tensor *matmul = tensor_matmul(a, b);
    // First row: 1*1 + 2*3 + 3*5 = 22, 1*2 + 2*4 + 3*6 = 28
    // Second row: 4*1 + 5*3 + 6*5 = 49, 4*2 + 5*4 + 6*6 = 64
    assert(float_equal(matmul->data[0], 22.0));
    assert(float_equal(matmul->data[1], 28.0));
    assert(float_equal(matmul->data[2], 49.0));
    assert(float_equal(matmul->data[3], 64.0));

    // Test dot product
    Tensor *vec1 = tensor_create(1, 3); // 3D vector
    Tensor *vec2 = tensor_create(1, 3);
    float vec1_data[] = {1.0, 2.0, 3.0};
    float vec2_data[] = {4.0, 5.0, 6.0};
    tensor_populate_array(vec1, vec1_data);
    tensor_populate_array(vec2, vec2_data);

    float dot = tensor_dot(vec1, vec2); // 1*4 + 2*5 + 3*6 = 32
    assert(float_equal(dot, 32.0));

    // Test cross product
    Tensor *cross = tensor_cross(vec1, vec2);
    // Cross product: [2*6-3*5, 3*4-1*6, 1*5-2*4]
    assert(float_equal(cross->data[0], -3.0)); // 2*6 - 3*5 = 12 - 15 = -3
    assert(float_equal(cross->data[1], 6.0));  // 3*4 - 1*6 = 12 - 6 = 6
    assert(float_equal(cross->data[2], -3.0)); // 1*5 - 2*4 = 5 - 8 = -3

    tensor_free(a);
    tensor_free(b);
    tensor_free(matmul);
    tensor_free(vec1);
    tensor_free(vec2);
    tensor_free(cross);
}

void test_reduction_operations() {
    // Create a 2x3 tensor
    Tensor *t = tensor_create(2, 2, 3);
    float t_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    tensor_populate_array(t, t_data);

    // Test sum
    assert(float_equal(tensor_sum(t), 21.0)); // 1+2+3+4+5+6

    // Test mean
    assert(float_equal(tensor_mean(t), 3.5)); // 21/6

    // Test min/max
    assert(float_equal(tensor_min(t), 1.0));
    assert(float_equal(tensor_max(t), 6.0));

    // Test argmin/argmax
    assert(tensor_argmin(t) == 0); // Index of minimum value (1.0)
    assert(tensor_argmax(t) == 5); // Index of maximum value (6.0)

    // Test sum along axis
    Tensor *sum_axis0 = tensor_sum_axis(t, 0);    // Sum along rows
    assert(float_equal(sum_axis0->data[0], 5.0)); // 1+4
    assert(float_equal(sum_axis0->data[1], 7.0)); // 2+5
    assert(float_equal(sum_axis0->data[2], 9.0)); // 3+6

    Tensor *sum_axis1 = tensor_sum_axis(t, 1);     // Sum along columns
    assert(float_equal(sum_axis1->data[0], 6.0));  // 1+2+3
    assert(float_equal(sum_axis1->data[1], 15.0)); // 4+5+6

    // Test mean along axis
    Tensor *mean_axis0 = tensor_mean_axis(t, 0);   // Mean along rows
    assert(float_equal(mean_axis0->data[0], 2.5)); // (1+4)/2
    assert(float_equal(mean_axis0->data[1], 3.5)); // (2+5)/2
    assert(float_equal(mean_axis0->data[2], 4.5)); // (3+6)/2

    Tensor *mean_axis1 = tensor_mean_axis(t, 1);   // Mean along columns
    assert(float_equal(mean_axis1->data[0], 2.0)); // (1+2+3)/3
    assert(float_equal(mean_axis1->data[1], 5.0)); // (4+5+6)/3

    tensor_free(t);
    tensor_free(sum_axis0);
    tensor_free(sum_axis1);
    tensor_free(mean_axis0);
    tensor_free(mean_axis1);
}

int main() {
    test_element_wise_operations();
    test_scalar_operations();
    test_linear_algebra_operations();
    test_reduction_operations();
    printf("All tests passed!\n");
    return 0;
}
