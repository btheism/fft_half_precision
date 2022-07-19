#include<iostream>
#include<fstream>
#include<exception>

#include <string>
#include <cuda_runtime.h>

#include <io_wrapper.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>
#include <yaml2cpp.hpp>

#include <stdlib.h>
#include <time.h> 

int main(int argc, char *argv[]) {
    std::string config_file_name = GetExeDirPath()+"test_1_channel_with_reflect.yaml";
    yaml_node config(config_file_name);
    int print_middle_result_flag=std::stoi(config("print_middle_result_flag")());
    
    long long int fft_length=std::stoll(config("fft_length")());
    long long int batch=std::stoll(config("batch")());
    long long int window_size=std::stoll(config("window_size")());
    long long int step=std::stoll(config("step")());
    int thread_num=std::stoll(config("thread_num")());
    if(thread_num==0)
    {
        thread_num=fft_length/2/8;
    }
    
    long long int begin_channel=1;
    long long int channel_num=fft_length/2;
    
    char* input_char;
    char* compressed_data;
    short *input_half;
    double *average_data;
    cudaMallocManaged((void**)&input_char,sizeof(char)*fft_length*batch);
    cudaMallocManaged((void**)&input_half,sizeof(short)*(fft_length+2)*(batch+window_size));
    cudaMallocManaged((void**)&average_data,sizeof(double)*fft_length/2);
    cudaMallocManaged((void**)&compressed_data,sizeof(unsigned char)*fft_length/2/8*((batch+window_size-step)-(window_size-step))/step);
    std::cout<<"fft length = "<<fft_length<<std::endl;
    std::cout<<"batch= "<<batch<<std::endl;
    std::cout<<"window size = "<<window_size<<std::endl;
    std::cout<<"step = "<<step<<std::endl;
    std::cout<<"input char size = "<<fft_length*batch<<std::endl;
    std::cout<<"input half size = "<<(fft_length+2)*(batch+window_size-step)<<std::endl;
    std::cout<<"average data size = "<<fft_length/2<<std::endl;
    std::cout<<"compressed data size = "<<fft_length/2/8*batch/step<<std::endl;
    std::cout<<"begin channel is "<<begin_channel<<std::endl;
    std::cout<<"compressed channel number is "<<channel_num<<std::endl;
    
    float *input_float=(float *)input_half;
    
    read_stream data(config("stream")());
    if (data.stream_size<fft_length*batch)
    {
        throw std::runtime_error("input stream is to small");
    }
    data.read(input_char, fft_length*batch);
    
    if(print_middle_result_flag)
    {
        print_data_signed_char((signed char*)input_char,0,fft_length*batch,fft_length);
    }
    
    //thread_num<=fft_length
    int2float(input_char,input_half,fft_length,batch,thread_num,0);
    
    //从倒数第一组batch反射
    char2float_reflect(input_char+fft_length*batch-1,input_half+(fft_length+2)*batch,fft_length,window_size,thread_num);
    
    if(print_middle_result_flag)
    {
        print_data_half(input_half,0,(fft_length+2)*(batch+window_size),fft_length+2);
        print_data_half_for_copy(input_half,0,(fft_length+2)*(batch+window_size),fft_length+2);
    }
    
    fft_1d_half(input_half,input_half,fft_length,(batch+window_size));
    
    if(print_middle_result_flag)
    {
    print_data_half(input_half,0,(fft_length+2)*(batch+window_size),fft_length+2);
    print_data_half_for_copy(input_half,0,(fft_length+2)*(batch+window_size),fft_length+2);
    }
    
    //thread_num<=fft_length/2
    complex_modulus_squared(input_half,input_half,fft_length/2,(batch+window_size),thread_num);
    
    if(print_middle_result_flag)
    {
        print_data_float(input_float,0,(fft_length/2+1)*(batch+window_size),fft_length/2+1);
    }
    
    //thread_num<=fft_length/2,这里减去step是为了配合compress函数
    channels_sum(input_float,average_data,window_size,(double)window_size,(fft_length/2+1),fft_length/2,thread_num);
    
    if(print_middle_result_flag)
    {
        print_data_double(average_data,0,fft_length/2,fft_length/2);
    }
    
    //thread_num<=fft_length/2/8
    compress(average_data, input_float, input_float+(fft_length/2+1)*window_size,compressed_data,(fft_length/2+1), batch/step, step, begin_channel ,channel_num, window_size, thread_num);
    
    print_data_binary(compressed_data,0,(channel_num/8)*(batch/step),channel_num/8);
    
    //打印各个window区间内的平均数以验证数据压缩是否正确
    if(print_middle_result_flag)
    {
        for(int i=0;i<batch/step;i++)
        {
            std::cout<<"print step "<<i<<" step_average vs window_average"<<std::endl;
            channels_sum(input_float+i*(fft_length/2+1)*step,average_data,step,step,(fft_length/2+1),fft_length/2,thread_num);
            print_data_double(average_data,0,fft_length/2,fft_length/2);
            channels_sum(input_float+i*(fft_length/2+1)*step,average_data,window_size,window_size,(fft_length/2+1),fft_length/2,thread_num);
            print_data_double(average_data,0,fft_length/2,fft_length/2);
        }
    }
    
    cudaFree(input_char);
    cudaFree(input_half);
    cudaFree(average_data);
    cudaFree(compressed_data);
    return 0;
}
 
