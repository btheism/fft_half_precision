#include <string>
#include <cuda_runtime.h>

#include <other_function_library.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>

int main(int argc, char *argv[]) {

    //生成文件列表
    std::string file_list[argc-1];
    long long int file_size_list[argc-1];
    singal_length=generate_file_list(argc, argv, file_list, file_size_list);
    
    int fft_length=131072;
    int window_size=4096;
    //由于会对末尾数据进行反射,故batch数需要加上windowsize
    int total_batch=singal_length/2/fft_length+window_size;
    int batch=4096
    //注意,由于compress函数一次读取step个数据,为避免内存访问越界,batch_buffer_size必须被step整除
    int batch_buffer_size=8192
    int reamin_batch=total_batch-batch_loop*batch
    int step=4;
    int thread_num=1024;
    int compress_batch=(batch-(window_size-step))/step;
    int factor_A=2;
    int factor_B=1;
    
    signed char *input_char;
    short *input_half;
    double *average_data;
    unsigned char* compressed_data;
    //signed char *input_char_cpu=new signed char[4096];
    //short *input_float_cpu=new short[4096];
    
    cudaMallocManaged((void**)&input_char,sizeof(signed char)*fft_length*batch*2);
    cudaMalloc((void**)&input_half_A,sizeof(short)*(fft_length+2)*batch*2);
    cudaMalloc((void**)&input_half_B,sizeof(short)*(fft_length+2)*batch*2);
    cudaMalloc((void**)&average_data,sizeof(double)*fft_length/2);
    cudaMallocManaged((void**)&compressed_data,sizeof(unsigned char)*fft_length/2/8*compress_batch);
    std::cout<<"fft length = "<<fft_length<<std::endl;
    std::cout<<"batch= "<<batch<<std::endl;
    std::cout<<"window size = "<<window_size<<std::endl;
    std::cout<<"step = "<<step<<std::endl;
    std::cout<<"input char size = "<<fft_length*batch*2<<std::endl;
    std::cout<<"input half size = "<<(fft_length+2)*batch*2<<std::endl;
    std::cout<<"average data size = "<<fft_length/2<<std::endl;
    std::cout<<"compressed data size = "<<fft_length/2/8*(batch-(window_size-step))/step<<std::endl;

    readfile(input_char,file_list,fft_length*batch*2);
    char2float_interlace(input_char,input_half_A,fft_length,batch,thread_num);
    fft_1d_half(input_half,input_half,fft_length,batch*2);
    complex_modulus_squared_interlace(input_half_A+2,input_half_A+2,fft_length/2,factor_A,factor_B,batch,thread_num);
    
    for(int batch_loop_num=0;batch_loop_num<batch_loop;batch_loop_num++)
    {
        readfile(input_char,file_list,fft_length*batch*2);
        char2float_interlace(input_char,input_half_B,fft_length,batch,thread_num);
        fft_1d_half(input_half_B,input_half_B,fft_length,batch*2);
        complex_modulus_squared_interlace(input_half_B+2,input_half_B+2,fft_length/2,factor_A,factor_B,batch,thread_num);
        compress(average_data,(input_float+1),compressed_data,(fft_length/2+1)*2,(batch-(window_size-step))/step,step,fft_length/2,window_size, thread_num);

    }
    
    
    cudaFree(input_char);
    cudaFree(input_half);
    
    
    return 0;
    
}

