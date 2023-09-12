#pragma once

#include <cstdint>
#include <string>
#include <fstream>
#include <byteswap.h>
#include <memory>
#include <iomanip>
#include <iostream>
#include "cnn/tensor_t.h"
#include "cuda.h"
#include <cuda_runtime.h>

// Some constants:
constexpr size_t img_dim = 28;      // All training images in the MNIST data set are 28x28.
constexpr size_t num_digits = 10;

// case including input data and label
struct case_t {
  tensor_t<float> data;
  tensor_t<float> out;
};

/// @brief Read the MNIST training data from
void readTrainingData();

/**
 * @brief Train the CNN with a whole dataset of images
 * @return          The total error after training the whole data set
 */
float trainCNN();

void init_dptrs();
__global__ void activate_conv_reduction_kernel();
__global__ void activate_fc_reduction_kernel();