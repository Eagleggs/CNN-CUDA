#include "global_ptr_cuda.cuh"
#include <cuda_runtime.h>
#include <cuda.h>

unsigned *cuda_ptr_iterator = nullptr;
float *cuda_ptr_img_array = nullptr;
float *cuda_ptr_lbl_array = nullptr;
float *cuda_ptr_err = nullptr;
float *cuda_ptr_tot_err = nullptr;


float* cuda_ptr_conv_in = nullptr;
float* cuda_ptr_relu_in = nullptr;
float* cuda_ptr_pool_in = nullptr;
float* cuda_ptr_fc_in = nullptr;
float* cuda_ptr_predict = nullptr;

float* cuda_ptr_conv_weights = nullptr;
float* cuda_ptr_fc_weights = nullptr;

float* cuda_ptr_conv_grad_old = nullptr;
float* cuda_ptr_conv_grad_new = nullptr;
float* cuda_ptr_fc_grad_old = nullptr;
float* cuda_ptr_fc_grad_new = nullptr;

float* cuda_ptr_grad_src = nullptr;
float* cuda_ptr_grad_dst = nullptr;


void weights_init(){
    float weights[11520];

    for (int z = 0; z < 8; z++)
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                weights[i+j*5+z*25] = 1.0f / 25 * rand() / float(RAND_MAX);
    cudaMemcpy(cuda_ptr_conv_weights, weights, 200*sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < 11520; i++)
        weights[i] = 2.19722f / 1152 * rand() / float(RAND_MAX);
    cudaMemcpy(cuda_ptr_fc_weights, weights, 11520*sizeof(float), cudaMemcpyHostToDevice);

}

void malloc_cuda() {
    cudaMalloc(&cuda_ptr_img_array, 67108864*sizeof(float));
    cudaMalloc(&cuda_ptr_lbl_array, 1048576*sizeof(float));
    cudaMallocHost(&cuda_ptr_err, 65536*sizeof(float));
    cudaMallocHost(&cuda_ptr_tot_err, 65536*sizeof(float));

    cudaMalloc(&cuda_ptr_conv_in, 1024*sizeof(float));
    cudaMalloc(&cuda_ptr_relu_in, 8192*sizeof(float));
    cudaMalloc(&cuda_ptr_pool_in, 8192*sizeof(float));
    cudaMalloc(&cuda_ptr_fc_in, 8192*sizeof(float));
    cudaMalloc(&cuda_ptr_predict, 16*sizeof(float));

    cudaMalloc(&cuda_ptr_conv_weights, 256*sizeof(float));
    cudaMalloc(&cuda_ptr_fc_weights, 16384*sizeof(float));

    cudaMalloc(&cuda_ptr_grad_src, 16*sizeof(float));
    cudaMalloc(&cuda_ptr_grad_dst, 8192 * sizeof(float));

    cudaMalloc(&cuda_ptr_conv_grad_old, 256*sizeof(float));
    cudaMalloc(&cuda_ptr_conv_grad_new, 256*sizeof(float));
    cudaMalloc(&cuda_ptr_fc_grad_old, 16*sizeof(float));
    cudaMalloc(&cuda_ptr_fc_grad_new, 16*sizeof(float));


    cudaMemset(&cuda_ptr_grad_dst, 0.0f, 8192*sizeof(float));
    cudaMemset(&cuda_ptr_grad_src, 0.0f, 8192*sizeof(float));

    cudaMemset(&cuda_ptr_conv_grad_old, 0.0f, 256*sizeof(float));
    cudaMemset(&cuda_ptr_conv_grad_new, 0.0f, 256*sizeof(float));
    cudaMemset(&cuda_ptr_fc_grad_old, 0.0f, 16*sizeof(float));
    cudaMemset(&cuda_ptr_fc_grad_new, 0.0f, 16*sizeof(float));
}


