
#include <string>
#include <fstream>
#include <cuda_runtime.h>

#include <other_function_library.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>

int main(int argc, char *argv[]) {

    //计算输入数据的总长度
    char** file_list=argv+1;
    long long int signal_length=GetFilelistSize(argc-1, file_list);
    std::cout<<"signal length = "<<signal_length<<std::endl;
    
    //设置缓冲区大小
    long long int cache_size=1024*1024*1024;
    
    char *input_char, *output_char_A, *output_char_B;
    cudaMallocHost((void**)&input_char,sizeof(signed char)*cache_size*sizeof(char));
    cudaMallocHost((void**)&output_char_A,sizeof(signed char)*cache_size/2*sizeof(char));
    cudaMallocHost((void**)&output_char_B,sizeof(signed char)*cache_size/2*sizeof(char));
    
    std::string first_input_name=std::string(file_list[0]);
    std::string output_A = first_input_name.substr(0, first_input_name.length() - 4) + "_A.dat";
    std::string output_B = first_input_name.substr(0, first_input_name.length() - 4) + "_B.dat";
    std::ofstream output_A_stream(output_A,std::ios::trunc|std::ios::binary);
    std::ofstream output_B_stream(output_B,std::ios::trunc|std::ios::binary);
    
    
    long long int read_length=0;

    long long int input_offset=0;
    long long int output_offset=0;
    
    //足够读满一个cache
    while(cache_size<=signal_length-read_length)
    {   
        input_offset=0;
        output_offset=0;
        readfile(input_char,file_list,cache_size);
        while(input_offset<cache_size)
        {
             output_char_A[output_offset]=input_char[input_offset];
             input_offset++;
             output_char_B[output_offset]=input_char[input_offset];
             input_offset++;
             output_offset++;
        }
        read_length+=cache_size;
        output_A_stream.write((const char *)output_char_A,cache_size/2*sizeof(char));
        output_B_stream.write((const char *)output_char_B,cache_size/2*sizeof(char));
        std::cout<<"read length = "<<read_length<<std::endl;
    }
    long long int remain_length=signal_length%cache_size;
    if(remain_length>0)
    {
        input_offset=0;
        output_offset=0;
        readfile(input_char,file_list,remain_length);
        while(input_offset<remain_length)
        {
             
             output_char_A[output_offset]=input_char[input_offset];
             input_offset++;
             output_char_B[output_offset]=input_char[input_offset];
             input_offset++;
             output_offset++;
        }
        output_A_stream.write((const char *)output_char_A,remain_length/2*sizeof(char));
        output_B_stream.write((const char *)output_char_B,remain_length/2*sizeof(char));
        
        read_length+=remain_length;
        std::cout<<"read length = "<<read_length<<std::endl;
    }
    readfile(input_char,file_list,-1);
    output_A_stream.close();
    output_B_stream.close();
}
        
