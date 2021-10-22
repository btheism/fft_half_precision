#define GENERATE_RANDOM_DATA

#include<iostream>
#include<fstream>

#include <string>
#include <cuda_runtime.h>

#include <other_function_library.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>

#include <stdlib.h>
#include <time.h> 

int main(int argc, char *argv[]) {
    long long int fft_length=16;
    long long int batch=50;
    long long int window_size=8;
    long long int step=1;
    long long int thread_num=1;
    
    float factor_A=1.0;
    float factor_B=1.0;
    
    long long int begin_channel=1;
    long long int channel_num=fft_length/2;
    
    signed char* input_char;
    unsigned char* compressed_data;
    short *input_half;
    double *average_data;
    cudaMallocManaged((void**)&input_char,sizeof(signed char)*fft_length*batch*2);
    cudaMallocManaged((void**)&input_half,sizeof(short)*(fft_length+2)*(batch+window_size-step)*2);
    cudaMallocManaged((void**)&average_data,sizeof(double)*fft_length/2);
    cudaMallocManaged((void**)&compressed_data,sizeof(unsigned char)*fft_length/2/8*((batch+window_size-step)-(window_size-step))/step);
    std::cout<<"fft length = "<<fft_length<<std::endl;
    std::cout<<"batch= "<<batch<<std::endl;
    std::cout<<"window size = "<<window_size<<std::endl;
    std::cout<<"step = "<<step<<std::endl;
    std::cout<<"input char size = "<<fft_length*batch*2<<std::endl;
    std::cout<<"input half size = "<<(fft_length+2)*(batch+window_size-step)*2<<std::endl;
    std::cout<<"average data size = "<<fft_length/2<<std::endl;
    std::cout<<"compressed data size = "<<fft_length/2/8*batch/step<<std::endl;
    float *input_float=(float *)input_half;
    
    //srand((unsigned)time(NULL));
    srand(932563);
    for(long long int i=0;i<fft_length*batch*2;i++)
        input_char[i]=rand()%256;
    
    #ifdef GENERATE_RANDOM_DATA
    //全新写入模式
    std::ofstream random_data("random_data_for_2_channel.dat",std::ios::trunc|std::ios::binary);
    if(random_data.is_open()) std::cout << "Succeed opening output random data file \n";
    random_data.write((const char *)input_char,fft_length*batch*2*sizeof(char));
    random_data.close();
    #endif
    
    print_data_signed_char(input_char,0,fft_length*batch*2,fft_length*2);
    
    //thread_num<=fft_length
    char2float_interlace(input_char,input_half,fft_length,batch,thread_num);
    
    //最后step个batch不用反射,从倒数第二组batch反射
    char2float_interlace_reflect(input_char+fft_length*(batch-step)*2-1,input_half+(fft_length+2)*batch*2,fft_length,window_size-step,thread_num);
    print_data_half(input_half,0,(fft_length+2)*(batch+window_size-step)*2,fft_length+2);
    print_data_half_for_copy(input_half,0,(fft_length+2)*(batch+window_size-step)*2,fft_length+2);
    
    //双通道数据的fft_num需要乘2
    fft_1d_half(input_half,input_half,fft_length,(batch+window_size-step)*2);
    print_data_half(input_half,0,(fft_length+2)*(batch+window_size-step)*2,fft_length+2);
    
    //thread_num<=fft_length/2
    complex_modulus_squared_interlace(input_half,input_half,fft_length/2,factor_A,factor_B,(batch+window_size-step),thread_num);
    print_data_float(input_float,0,(fft_length/2+1)*(batch+window_size-step)*2,fft_length/2+1);
    
    //thread_num<=fft_length/2,这里减去step是为了配合compress函数
    channels_sum(input_float,average_data,window_size-step,(double)window_size,(fft_length/2+1)*2,fft_length/2,thread_num);
    print_data_double(average_data,0,fft_length/2,fft_length/2);
    
    //thread_num<=fft_length/2/8
    compress(average_data, input_float, input_float+(fft_length/2+1)*2*window_size,compressed_data,(fft_length/2+1)*2, ((batch+window_size-step)-(window_size-step))/step, step, begin_channel ,channel_num, window_size, thread_num);
    print_data_binary(compressed_data,0,(channel_num/8)*(((batch+window_size-step)-(window_size-step))/step),channel_num/8);
    
    //打印各个window区间内的平均数以验证数据压缩是否正确
    for(int i=0;i<((batch+window_size-step)-(window_size-step))/step;i++)
    {
        std::cout<<"print step "<<i<<" step_average vs window_average"<<std::endl;
        channels_sum((input_float+i*(fft_length/2+1)*2*step),average_data,step,step,(fft_length/2+1)*2,fft_length/2,thread_num);
        print_data_double(average_data,0,fft_length/2,fft_length/2);
        channels_sum((input_float+i*(fft_length/2+1)*2*step),average_data,window_size,window_size,(fft_length/2+1)*2,fft_length/2,thread_num);
        print_data_double(average_data,0,fft_length/2,fft_length/2);
    }
    
    cudaFree(input_char);
    cudaFree(input_half);
    cudaFree(average_data);
    cudaFree(compressed_data);
    return 0;
}

  
