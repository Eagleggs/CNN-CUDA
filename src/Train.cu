#include "Train.cuh"
#include "utils/io.h"
#include "cnn/global_ptr_cuda.cuh"
#include "utils/Timer.h"
#include <algorithm>
#include "cnn/optimization_method.h"

__device__ unsigned dcuda_iterator = 0;
__device__ float *dcuda_ptr_img_array;
__device__ float *dcuda_ptr_lbl_array;
__device__ float *dcuda_ptr_err;
__device__ float *dcuda_ptr_tot_err;

__device__ float *dcuda_ptr_relu_in;
__device__ float *dcuda_ptr_pool_in;
__device__ float *dcuda_ptr_fc_in;
__device__ float *dcuda_ptr_predict;

__device__ float *dcuda_ptr_conv_weights;
__device__ float *dcuda_ptr_fc_weights;

__device__ float *dcuda_ptr_conv_grad_old;
__device__ float *dcuda_ptr_conv_grad_new;
__device__ float *dcuda_ptr_fc_grad_old;
__device__ float *dcuda_ptr_fc_grad_new;

__device__ float *dcuda_ptr_grad_src;
__device__ float *dcuda_ptr_grad_dst;

void init_dptrs(){
    cudaGetSymbolAddress((void **)&cuda_ptr_iterator, dcuda_iterator);

    cudaMemcpyToSymbol(dcuda_ptr_img_array, &cuda_ptr_img_array, sizeof(float *));
    cudaMemcpyToSymbol(dcuda_ptr_lbl_array, &cuda_ptr_lbl_array, sizeof(float *));
    cudaMemcpyToSymbol(dcuda_ptr_err, &cuda_ptr_err, sizeof(float *));
    cudaMemcpyToSymbol(dcuda_ptr_tot_err, &cuda_ptr_tot_err, sizeof(float *));

    cudaMemcpyToSymbol(dcuda_ptr_relu_in, &cuda_ptr_relu_in, sizeof(float *));
    cudaMemcpyToSymbol(dcuda_ptr_pool_in, &cuda_ptr_pool_in, sizeof(float *));
    cudaMemcpyToSymbol(dcuda_ptr_fc_in, &cuda_ptr_fc_in, sizeof(float *));
    cudaMemcpyToSymbol(dcuda_ptr_predict, &cuda_ptr_predict, sizeof(float *));

    cudaMemcpyToSymbol(dcuda_ptr_conv_weights, &cuda_ptr_conv_weights, sizeof(float *));
    cudaMemcpyToSymbol(dcuda_ptr_fc_weights, &cuda_ptr_fc_weights, sizeof(float *));

    cudaMemcpyToSymbol(dcuda_ptr_grad_src, &cuda_ptr_grad_src, sizeof(float *));
    cudaMemcpyToSymbol(dcuda_ptr_grad_dst, &cuda_ptr_grad_dst, sizeof(float *));

    cudaMemcpyToSymbol(dcuda_ptr_conv_grad_new, &cuda_ptr_conv_grad_new, sizeof(float *));
    cudaMemcpyToSymbol(dcuda_ptr_conv_grad_old, &cuda_ptr_conv_grad_old, sizeof(float *));
    cudaMemcpyToSymbol(dcuda_ptr_fc_grad_new, &cuda_ptr_fc_grad_new, sizeof(float *));
    cudaMemcpyToSymbol(dcuda_ptr_fc_grad_old, &cuda_ptr_fc_grad_old, sizeof(float *));
}

void readTrainingData() {
    Timer timer;
    timer.start();
    Timer t;
    std::vector<case_t> cases{};

    t.start();
    auto train_image = readFile("../train-images.idx3-ubyte");
    t.stop();
    t.start();
    auto train_labels = readFile("../train-labels.idx1-ubyte");
    t.stop();
    t.start();

    // Swap endianness of case count:
    uint32_t case_count = bswap_32(*(uint32_t *) (train_image.data() + sizeof(uint32_t)));

    // Convert each image to a test case
    for (int i = 0; i < case_count; i++) {
        case_t c{tensor_t<float>(img_dim, img_dim, 1), tensor_t<float>(num_digits, 1, 1)};

        // Actual images start at offset 16
        uint8_t *img = train_image.data() + 16 + i * (img_dim * img_dim);

        // Actual labels start at offset 8
        uint8_t *label = train_labels.data() + 8 + i;

        // Normalize the pixel intensity values to a floating point number between 0 and 1.
        for (int x = 0; x < img_dim; x++) {
            for (int y = 0; y < img_dim; y++) {
                c.data(x, y, 0) = img[x + y * img_dim] / 255.f;
            }
        }

        // Convert the labels to a floating point number at the correct digit index
        for (int b = 0; b < num_digits; b++) {
            c.out(b, 0, 0) = *label == b ? 1.0f : 0.0f;
        }
        cudaMemcpy(cuda_ptr_lbl_array + i * 10, &c.out.data[0], 10 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_ptr_img_array + i * 784, &c.data.data[0], 784 * sizeof(float), cudaMemcpyHostToDevice);
        cases.push_back(c);
    }
    timer.stop();
}



__global__ void activate_conv_reduction_kernel() {
    __shared__ float sdata[500];
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= 115200)
        return;

    const float
            *in = dcuda_ptr_img_array + dcuda_iterator * 784,
            *weights = dcuda_ptr_conv_weights;
    float
            *out = dcuda_ptr_relu_in,
            *relu_out = dcuda_ptr_pool_in,
            *pool_out = dcuda_ptr_fc_in;

    unsigned n = index / 25 / 24 / 24;

    unsigned px = index / 25 % 2;
    unsigned py = index / 25 / 2 % 2;
    unsigned cx = index / 25 / 2 / 2 % 12;
    unsigned cy = index / 25 / 2 / 2 / 12 % 12;
    unsigned x = cx * 2 + px;
    unsigned y = cy * 2 + py;

    unsigned i = threadIdx.x;
    unsigned wy = i / 5 % 5;
    unsigned wx = i % 5;
    unsigned m = i % 25;

    sdata[i] = in[(x + wx) + (y + wy) * 28] * weights[wx + wy * 5 + n * 25];
    __syncthreads();
    if (m < 9) sdata[i] += sdata[i + 16];
    if (m >= 8) return;
    __syncthreads();
    sdata[i] += sdata[i + 8];
    if (m >= 4) return;
    __syncthreads();
    sdata[i] += sdata[i + 4];
    if (m >= 2) return;
    __syncthreads();
    sdata[i] += sdata[i + 2];
    if (m >= 1) return;
    if (m == 0) {
        float f = sdata[i] + sdata[i + 1];
        out[x + y * 24 + n * 576] = f;
        f = f < 0 ? 0 : f;
        relu_out[x + y * 24 + n * 576] = f;
        __syncthreads();
        sdata[cx%10*4 + cy%2*40 + px + py*2] = f;
        __syncthreads();
        if (px == 0 && py == 0){
            pool_out[cx + cy * 12 + n * 144] = fmax(fmax(f, sdata[cx%10*4 + cy%2*40 + 1]),
                                                    fmax(sdata[cx%10*4 + cy%2*40 + 2],
                                                         sdata[cx%10*4 + cy%2*40 + 3]));
        }
    }
}
__global__ void calc_grads_conv_reduction_kernel() {

    __shared__ float sdata[576];
    int index = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    if (index >= 115200)
        return;
    const float
            *in = dcuda_ptr_img_array + dcuda_iterator * 784,
            *grad_dst = dcuda_ptr_grad_dst;
    float
            *weights = dcuda_ptr_conv_weights,
            *grad_new = dcuda_ptr_conv_grad_new,
            *grad_old = dcuda_ptr_conv_grad_old;

    int x = threadIdx.x % 24;
    int y = threadIdx.x / 24;

    int wx = blockIdx.x % 5;
    int wy = blockIdx.x / 5 % 5;
    int n = blockIdx.x / 5 / 5 % 8;
    int wi = blockIdx.x;

    int i = threadIdx.x;
    sdata[i] = in[(x + wx) + (y + wy) * 28] * grad_dst[x + y * 24 + n * 576];
    __syncthreads();
    if (i < 64) sdata[i] += sdata[i + 512];
    if (i >= 256) return;
    __syncthreads();
    sdata[i] += sdata[i + 256];
    if (i >= 128) return;
    __syncthreads();
    sdata[i] += sdata[i + 128];
    if (i >= 64) return;
    __syncthreads();
    sdata[i] += sdata[i + 64];
    if (i >= 32) return;
    __syncthreads();
    sdata[i] += sdata[i + 32];
    if (i >= 16) return;
    __syncthreads();
    sdata[i] += sdata[i + 16];
    if (i >= 8) return;
    sdata[i] += sdata[i + 8];
    if (i >= 4) return;
    sdata[i] += sdata[i + 4];
    if (i >= 2) return;
    sdata[i] += sdata[i + 2];
    if (i >= 1) return;
    if (i == 0) {
        float grad = sdata[i] + sdata[i + 1];
        grad_new[wi] = grad;
        float w = weights[wi];
        float m = (grad + grad_old[wi] * MOMENTUM);
        w -= LEARNING_RATE * m + LEARNING_RATE * WEIGHT_DECAY * w;
        weights[wi] = w;
        grad_old[wi] = m;
    }
    if(index == 0)
        dcuda_iterator++;
}

__global__ void activate_fc_reduction_kernel() {

    __shared__ float sdata[576];
    int n = blockIdx.x;
    int i = threadIdx.x;
    int index = i + n * 1152;

    const float
            *expected = dcuda_ptr_lbl_array + dcuda_iterator * 10,
            *in = dcuda_ptr_fc_in,
            *weights = dcuda_ptr_fc_weights;
    float
            *out = dcuda_ptr_predict,
            *grad_src = dcuda_ptr_grad_src,
            *tot_err = dcuda_ptr_tot_err + dcuda_iterator;

    sdata[i] = in[i] * weights[index] + in[i + 576] * weights[index + 576];

    __syncthreads();
    if (i < 64) {
        sdata[i] += sdata[i + 512];
    }
    if (i >= 256) return;
    __syncthreads();
    sdata[i] += sdata[i + 256];
    if (i >= 128) return;
    __syncthreads();
    sdata[i] += sdata[i + 128];
    if (i >= 64) return;
    __syncthreads();
    sdata[i] += sdata[i + 64];
    if (i >= 32) return;
    __syncthreads();
    sdata[i] += sdata[i + 32];
    if (i >= 16) return;
    __syncthreads();
    sdata[i] += sdata[i + 16];
    if (i >= 8) return;
    sdata[i] += sdata[i + 8];
    if (i >= 4) return;
    sdata[i] += sdata[i + 4];
    if (i >= 2) return;
    sdata[i] += sdata[i + 2];
    if (i >= 1) return;
    if (i == 0) {
        float predict = 1.0f / (1.0f + expf(-(sdata[i] + sdata[i + 1])));
        out[n] = predict;
        float exp = expected[n];
        float grad = predict - exp;
        grad_src[n] = grad;
        if (exp > 0.5){
            if(dcuda_iterator == 0)
                *tot_err = fabs(grad);
            else
                *tot_err = fabs(grad)*100 + *(tot_err-1);
        }
    }
}
__global__ void calc_grads_fc_reduction_kernel() {
    // 12 * 500
    __shared__ float sdata[500];
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 5760)
        return;
    unsigned i = threadIdx.x;
    unsigned n = i % 5;
    unsigned x = index / 5;
    const float
            *grad_src = dcuda_ptr_grad_src,
            *out = dcuda_ptr_predict,
            *pool_in = dcuda_ptr_pool_in,
            *relu_in = dcuda_ptr_relu_in,
            *fc_in = dcuda_ptr_fc_in;
    float
        *grad_new = dcuda_ptr_fc_grad_new,
        *grad_dst = dcuda_ptr_grad_dst,
        *grad_old = dcuda_ptr_fc_grad_old,
        *weights = dcuda_ptr_fc_weights
        ;
    float sig = out[n];
    float grad = grad_src[n] * sig * (1 - sig);
    float sig2 = out[n + 5];
    float grad2 = grad_src[n + 5] * sig2 * (1 - sig2);


    sdata[i] = grad * weights[x + n * 1152] + grad2 * weights[x + (n + 5) * 1152];
    __syncthreads();
    if (n < 2) sdata[i] += sdata[i + 2];
    __syncthreads();

    if (n == 0) {
        auto ox = x % 12;
        auto oy = x / 12 % 12;
        auto on = x / 144;
        auto iid = ox * 2 + oy * 48 + on * 576;
        float sum_error = sdata[i] + sdata[i + 1] + sdata[i + 4];
        float out_max = fc_in[x];
        grad_dst[iid + 00] = relu_in[iid + 00] < 0 ? 0 : pool_in[iid + 00] == out_max ? sum_error : 0;
        grad_dst[iid + 01] = relu_in[iid + 01] < 0 ? 0 : pool_in[iid + 01] == out_max ? sum_error : 0;
        grad_dst[iid + 24] = relu_in[iid + 24] < 0 ? 0 : pool_in[iid + 24] == out_max ? sum_error : 0;
        grad_dst[iid + 25] = relu_in[iid + 25] < 0 ? 0 : pool_in[iid + 25] == out_max ? sum_error : 0;
    }

    {
        float m = (grad + grad_old[n] * MOMENTUM);
        if (x == 0){
            grad_new[n] = grad;
            grad_old[n] = m;
        }

        float w = weights[x + 1152 * n];
        float multp = fc_in[x];
        w -= LEARNING_RATE * m * multp + LEARNING_RATE * WEIGHT_DECAY * w;
        weights[x + 1152 * n] = w;
    }
    {
        n += 5;
        float m = (grad2 + grad_old[n] * MOMENTUM);
        if (x == 0){
            grad_old[n] = m;
            grad_new[n + 5] = grad2;
        }
        float w = weights[x + 1152 * n];
        float multp = fc_in[x];
        w -= LEARNING_RATE * m * multp + LEARNING_RATE * WEIGHT_DECAY * w;
        weights[x + 1152 * n] = w;
    }
}

float trainCNN() {

    // Iterate over all the test cases
    for (int i = 0; i < 60000; i++) {

        // CONV activate & RELU activate & POOL activate
        activate_conv_reduction_kernel<<<461, 250>>>();
        // FC activate
        activate_fc_reduction_kernel<<<10, 576>>>();

        // FC grads & FC fix & POOL grads & RELU grads
        calc_grads_fc_reduction_kernel<<<24, 250>>>();
        // CONV grads & CONV fix
        calc_grads_conv_reduction_kernel<<<200, 576>>>();
    }
    return cuda_ptr_tot_err[59999];
}
