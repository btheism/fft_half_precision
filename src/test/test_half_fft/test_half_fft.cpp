#include <cuda_runtime.h>
#include <other_function_library.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>

int main(int argc, char *argv[]) {
    short *input_half;
    cudaMallocManaged((void**)&input_half,sizeof(unsigned char)*100);
    float input_float[10]={1.,2.,3.,4.,5.,6.,7.,8.,9.,10.};
    float_2_half((void*)input_float,(void*)input_half,0,10);
    print_data_half(input_half,0,15);
    print_data_half_for_copy(input_half,0,15);
    fft_1d_half((void*)input_half, (void*)input_half,1024,4);
    print_data_half(input_half,0,15);
    print_data_half_for_copy(input_half,0,15);
}
