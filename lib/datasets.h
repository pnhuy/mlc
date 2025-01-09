#ifndef DATASET_H
#define DATASET_H

#include "la.h"

typedef struct {
    Tensor *X;
    Tensor *y;
} Dataset;

Dataset *make_regression(int n_samples, int n_features, Tensor *weight, double bias) {
    Tensor *X = rand_tensor(n_samples, n_features);
    
}



#endif
