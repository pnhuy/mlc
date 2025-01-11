#include <assert.h>
#include <math.h>
#include <string.h>
#include "tensor.h"
#include "utils.h"

// Test creation functions
void test_tensor_create() {
    printf("\nTesting tensor_create...\n");
    
    // Test 1: Create 2D tensor
    Tensor *t1 = tensor_create(2, 2, 3);
    assert(t1 != NULL);
    assert(t1->ndim == 2);
    assert(t1->shape[0] == 2);
    assert(t1->shape[1] == 3);
    assert(t1->size == 6);
    tensor_free(t1);
    printf("2D tensor creation passed\n");
    
    // Test 2: Create 3D tensor
    Tensor *t2 = tensor_create(3, 2, 3, 4);
    assert(t2 != NULL);
    assert(t2->ndim == 3);
    assert(t2->size == 24);
    tensor_free(t2);
    printf("3D tensor creation passed\n");
    
    // Test 3: Create 1D tensor
    Tensor *t3 = tensor_create(1, 5);
    assert(t3 != NULL);
    assert(t3->ndim == 1);
    assert(t3->size == 5);
    tensor_free(t3);
    printf("1D tensor creation passed\n");
}

// Test data population and copying
void test_tensor_data_operations() {
    printf("\nTesting tensor data operations...\n");
    
    // Test 1: Populate and copy 2D tensor
    size_t shape[] = {2, 3};
    Tensor *t1 = tensor_create_from_shape(2, shape);
    float data[] = {1, 2, 3, 4, 5, 6};
    tensor_populate_array(t1, data);
    
    Tensor *t2 = tensor_copy(t1);
    assert(tensor_equal(t1, t2));
    
    tensor_free(t1);
    tensor_free(t2);
    printf("Data population and copying passed\n");
    
    // Test 2: Check data integrity
    t1 = tensor_create_from_shape(2, shape);
    tensor_populate_array(t1, data);
    for (size_t i = 0; i < t1->size; i++) {
        assert(float_equal(t1->data[i], data[i]));
    }
    tensor_free(t1);
    printf("Data integrity passed\n");
}

// Test reshape operation
void test_tensor_reshape() {
    printf("\nTesting tensor_reshape...\n");
    
    // Test 1: Basic reshape
    size_t shape1[] = {2, 3};
    Tensor *t1 = tensor_create_from_shape(2, shape1);
    float data[] = {1, 2, 3, 4, 5, 6};
    tensor_populate_array(t1, data);
    
    size_t new_shape[] = {3, 2};
    Tensor *t2 = tensor_reshape(t1, 2, new_shape);
    assert(t2 != NULL);
    assert(t2->size == t1->size);
    
    tensor_free(t2);
    printf("Basic reshape passed\n");
    
    // Test 2: Invalid reshape
    size_t invalid_shape[] = {4, 2};  // Different total size
    t2 = tensor_reshape(t1, 2, invalid_shape);
    assert(t2 == NULL);
    printf("Invalid reshape handling passed\n");
    tensor_free(t1);
}

// Test transpose operation
void test_tensor_transpose() {
    printf("\nTesting tensor_transpose...\n");
    
    // Test 1: 2D tensor transpose
    size_t shape[] = {2, 3};
    Tensor *t1 = tensor_create_from_shape(2, shape);
    float data[] = {1, 2, 3, 4, 5, 6};
    tensor_populate_array(t1, data);
    
    Tensor *t2 = tensor_transpose(t1);
    assert(t2 != NULL);
    assert(t2->shape[0] == t1->shape[1]);
    assert(t2->shape[1] == t1->shape[0]);
    assert(float_equal(t2->data[0], 1));
    assert(float_equal(t2->data[1], 4));
    
    tensor_free(t1);
    tensor_free(t2);
    printf("2D tensor transpose passed\n");
}

// Test concatenation
void test_tensor_concatenate() {
    printf("\nTesting tensor_concatenate...\n");
    
    // Test 1: Concatenate along axis 0
    size_t shape[] = {2, 3};
    Tensor *t1 = tensor_create_from_shape(2, shape);
    Tensor *t2 = tensor_create_from_shape(2, shape);
    
    float data1[] = {1, 2, 3, 4, 5, 6};
    float data2[] = {7, 8, 9, 10, 11, 12};
    tensor_populate_array(t1, data1);
    tensor_populate_array(t2, data2);
    
    Tensor *t3 = tensor_concatenate(t1, t2, 0);
    assert(t3 != NULL);
    assert(t3->shape[0] == t1->shape[0] + t2->shape[0]);
    assert(t3->shape[1] == t1->shape[1]);
    
    tensor_free(t1);
    tensor_free(t2);
    tensor_free(t3);
    printf("Concatenation along axis 0 passed\n");
}

// Test random tensor creation
void test_tensor_rand() {
    printf("\nTesting tensor_rand...\n");
    
    // Test 1: Create random tensor
    Tensor *t1 = tensor_rand(2, 3, 4);
    assert(t1 != NULL);
    assert(t1->ndim == 2);
    assert(t1->shape[0] == 3);
    assert(t1->shape[1] == 4);
    
    // Check if values are within expected range
    for (size_t i = 0; i < t1->size; i++) {
        assert(t1->data[i] >= 0.0f && t1->data[i] <= 1.0f);
    }
    
    tensor_free(t1);
    printf("Random tensor creation passed\n");
}

int main() {
    printf("Running tensor operations tests...\n");
    
    test_tensor_create();
    test_tensor_data_operations();
    test_tensor_reshape();
    test_tensor_transpose();
    test_tensor_concatenate();
    test_tensor_rand();
    
    printf("\nAll tests passed successfully!\n");
    return 0;
}
