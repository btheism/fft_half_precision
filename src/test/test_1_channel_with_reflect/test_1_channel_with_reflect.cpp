constexpr bool generate_random_data_flag = 1 ;
//constexpr bool print_middle_result_flag = 0 ;
constexpr bool print_middle_result_flag = 1 ;

#include<iostream>
#include<fstream>

#include <string>
#include <cuda_runtime.h>

#include <io_wrapper.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>

#include <stdlib.h>
#include <time.h> 

int main(int argc, char *argv[]) {
    long long int fft_length=64;
    long long int batch=6;
    long long int window_size=2;
    long long int step=1;
    int thread_num=4;    
    
    long long int begin_channel=1;
    long long int channel_num=32;
    
    char* input_char;
    char* compressed_data;
    short *input_half;
    double *average_data;
    cudaMallocManaged((void**)&input_char,sizeof(signed char)*fft_length*batch);
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
    
    //srand((unsigned)time(NULL));
    srand(932563);
    for(long long int i=0;i<fft_length*batch;i++)
        input_char[i]=rand()%256;
    
    if(generate_random_data_flag)
    {
        //全新写入模式
        std::ofstream random_data("/tmp/random_data_for_1_channel.dat",std::ios::trunc|std::ios::binary);
        if(random_data.is_open()) std::cout << "Succeed opening output random data file \n";
        random_data.write((const char *)input_char,fft_length*batch*sizeof(char));
        random_data.close();
    }
    
    if(print_middle_result_flag)
    {
        print_data_signed_char(input_char,0,fft_length*batch,fft_length);
    }
    
    //thread_num<=fft_length
    char2float(input_char,input_half,fft_length,batch,thread_num,0);
    
    //最后step个batch不用反射,从倒数第二组batch反射
    char2float_reflect(input_char+fft_length*(batch-step)-1,input_half+(fft_length+2)*batch,fft_length,window_size,thread_num);
    
    if(print_middle_result_flag)
    {
        print_data_half(input_half,0,(fft_length+2)*(batch+window_size-step),fft_length+2);
        print_data_half_for_copy(input_half,0,(fft_length+2)*(batch+window_size-step),fft_length+2);
    }
    
    fft_1d_half(input_half,input_half,fft_length,(batch+window_size-step));
    
    if(print_middle_result_flag)
    {
    print_data_half(input_half,0,(fft_length+2)*(batch+window_size-step),fft_length+2);
    print_data_half_for_copy(input_half,0,(fft_length+2)*(batch+window_size-step),fft_length+2);
    }
    
    //thread_num<=fft_length/2
    complex_modulus_squared(input_half,input_half,fft_length/2,(batch+window_size-step),thread_num);
    
    if(print_middle_result_flag)
    {
        print_data_float(input_float,0,(fft_length/2+1)*(batch+window_size-step),fft_length/2+1);
    }
    
    //thread_num<=fft_length/2,这里减去step是为了配合compress函数
    channels_sum(input_float,average_data,window_size,(double)window_size,(fft_length/2+1),fft_length/2,thread_num);
    
    if(print_middle_result_flag)
    {
        print_data_double(average_data,0,fft_length/2,fft_length/2);
    }
    
    //thread_num<=fft_length/2/8
    compress(average_data, input_float, input_float+(fft_length/2+1)*window_size,compressed_data,(fft_length/2+1), batch/step, step, begin_channel ,channel_num, window_size, thread_num);
    print_data_binary(compressed_data,0,(channel_num/8)*(((batch+window_size-step)-(window_size-step))/step),channel_num/8);
    
    //打印各个window区间内的平均数以验证数据压缩是否正确
    if(print_middle_result_flag)
    {
        for(int i=0;i<((batch+window_size-step)-(window_size-step))/step;i++)
        {
            std::cout<<"print step "<<i<<" step_average vs window_average"<<std::endl;
            channels_sum((input_float+i*(fft_length/2+1)*step),average_data,step,step,(fft_length/2+1),fft_length/2,thread_num);
            print_data_double(average_data,0,fft_length/2,fft_length/2);
            channels_sum((input_float+i*(fft_length/2+1)*step),average_data,window_size,window_size,(fft_length/2+1),fft_length/2,thread_num);
            print_data_double(average_data,0,fft_length/2,fft_length/2);
        }
    }
    
    cudaFree(input_char);
    cudaFree(input_half);
    cudaFree(average_data);
    cudaFree(compressed_data);
    return 0;
}
 
