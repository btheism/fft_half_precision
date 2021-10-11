#include<iostream>

#include <string>
#include <cuda_runtime.h>

#include <other_function_library.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>

#include <stdlib.h>
#include <time.h> 

int main(int argc, char *argv[]) {
    int fft_length=16;
    int batch=6;
    int window_size=3;
    signed char* input_char;
    unsigned char* compressed_data;
    short *input_half;
    double *average_data;
    cudaMallocManaged((void**)&input_char,sizeof(signed char)*fft_length*batch*2);
    cudaMallocManaged((void**)&input_half,sizeof(short)*(fft_length+2)*batch*2);
    cudaMallocManaged((void**)&average_data,sizeof(double)*fft_length/2);
    cudaMallocManaged((void**)&compressed_data,sizeof(unsigned char)*fft_length/2/8);
    float *input_float=(float *)input_half;
    
    srand((unsigned)time(NULL));
    for(int i=0;i<fft_length*batch*2;i++)
        input_char[i]=rand()%256;
    
    print_data_signed_char(input_char,0,fft_length*batch*2,fft_length);
    
    //thread_num<=fft_length
    char2float_interlace(input_char,input_half,fft_length,batch,fft_length/4);
    print_data_half(input_half,0,(fft_length+2)*batch*2,fft_length+2);
    
    //双通道数据的fft_num需要乘2
    fft_1d_half(input_half,input_half,fft_length,batch*2);
    print_data_half(input_half,0,(fft_length+2)*batch*2,fft_length+2);
    
    //调整input_half的地址为第2个通道,由于一个复数占用两个half,故加2,thread_num<=fft_length/2
    complex_modulus_squared_interlace(input_half+2,input_half+2,fft_length/2,1,1,batch,fft_length/4);
    print_data_float(input_float,0,(fft_length/2+1)*batch*2,fft_length/2+1);
    
    //调整input_float的地址为第2个通道,此时已把复数转为浮点数,故加1,thread_num<=fft_length/2
    channels_average((input_float+1),average_data,window_size,(fft_length/2+1)*2,fft_length/2,fft_length/4);
    print_data_double(average_data,0,fft_length/2,fft_length/2);
    
    //调整input_float的地址为第2个通道,thread_num<=fft_length/2/8
    compress(average_data,(input_float+1),compressed_data,(fft_length/2+1)*2,batch-window_size,1,fft_length/2,window_size, fft_length/2/8);
    print_data_binary(compressed_data,0,fft_length/2/8*batch,fft_length/2/8);
    
    //打印各个window区间内的平均数以验证数据压缩是否正确
    for(int i=0;i<(batch-window_size);i++)
    {
        std::cout<<"compute average data from batch "<<i<<std::endl;
        channels_average((input_float+1+i),average_data,window_size,(fft_length/2+1),fft_length/2,fft_length/4);
        print_data_double(average_data,0,fft_length/2,fft_length/2);
    }
    
    cudaFree(input_char);
    cudaFree(input_half);
    cudaFree(average_data);
    cudaFree(compressed_data);
    return 0;
}

 
