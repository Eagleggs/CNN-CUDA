

#include "cuda.h"
#include "../utils/Timer.h"

extern unsigned *cuda_ptr_iterator;

extern float *cuda_ptr_img_array;
extern float *cuda_ptr_lbl_array;
extern float *cuda_ptr_err;
extern float *cuda_ptr_tot_err;

extern float *cuda_ptr_conv_in;
extern float *cuda_ptr_relu_in;
extern float *cuda_ptr_pool_in;
extern float *cuda_ptr_fc_in;
extern float *cuda_ptr_predict;

extern float *cuda_ptr_conv_weights;
extern float *cuda_ptr_fc_weights;

extern float *cuda_ptr_conv_grad_old;
extern float *cuda_ptr_conv_grad_new;
extern float *cuda_ptr_fc_grad_old;
extern float *cuda_ptr_fc_grad_new;

extern float *cuda_ptr_grad_src;
extern float *cuda_ptr_grad_dst;


void malloc_cuda();

void weights_init();

