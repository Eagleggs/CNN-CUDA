
#include <iostream>
#include <algorithm>

#include "utils/Timer.h"

#include "Train.cuh"
#include "Infer.cuh"
#include "cnn/global_ptr_cuda.cuh"

int main() {
    Timer t;
    malloc_cuda();
    init_dptrs();

    /* Training phase */
    std::cout << "Reading test cases." << std::endl;
    readTrainingData();

    // The following layers make up the CNN in sequence.
    std::cout << "Creating layers." << std::endl;
    weights_init();


    // Train the CNN
    t.start();
    auto total_error = trainCNN();
    t.stop();

    cudaDeviceSynchronize();
    for (int i = 6000; i <= 60000; i += 6000)
        std::cout << "Case " << std::setw(5) << i << ". Err=" << cuda_ptr_tot_err[i - 1] / (float) i << std::endl;
    total_error = cuda_ptr_tot_err[59999 - 1];
    std::cout << "Training completed.\n";
    std::cout << "  Time (baseline): " << t.seconds() << " s.\n";
    std::cout << "  Total error: " << total_error << std::endl;

    auto image_tensor = loadImageAsTensor("../test.ppm");

    forward(image_tensor);

    printResult();


    return 0;

}
