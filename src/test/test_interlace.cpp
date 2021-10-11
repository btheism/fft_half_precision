#include <string>
#include <cuda_runtime.h>

#include <other_function_library.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>

int main(int argc, char *argv[]) {
    
    signed char* input_char;
    short *input_half;
    double *average;
    cudaMallocManaged((void**)&input_char,sizeof(unsigned char)*32);
    cudaMallocManaged((void**)&input_half,sizeof(short)*48);
    cudaMallocManaged((void**)&average,sizeof(double)*4);
    float *input_float=(float *)input_half;
    //signed input_char[32]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    //设这组数据有4个batch,每个batch有2个通道,每个通道有4个采样点,则共计8个fft变换,变换后的fft长度为3(3个复数,6个实数),因此需要48位数组储存
    for(int i=0;i<32;i++)
        input_char[i]=i+1;
    
    print_data_signed_char(input_char,0,32);
    
    char2float_interlace(input_char,input_half,4,4,4);
    print_data_half(input_half,0,48);
    
    fft_1d_half((void*)input_half, (void*)input_half,4,8);
    print_data_half(input_half,0,48);
    
    complex_modulus_squared_interlace(input_half,input_half,2,1,1,4,2);
    print_data_float(input_float,0,24);
    
    channels_average((void*)(input_float+1),(void*)average,2,3*2,2,2);
    print_data_double(average,0,4);
    
    cudaFree(input_char);
    cudaFree(input_half);
    
    
    return 0;
    
}

 
