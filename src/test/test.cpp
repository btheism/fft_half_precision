//一个测试程序，测试char2float_interlace()函数和 fft_1d_half()函数的正确性
#include <string>
#include <cuda_runtime.h>

#include <other_function_library.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>

int main(int argc, char *argv[]) {

    //生成文件列表
    std::string file_list[argc-1];
    generate_file_list(argc ,argv,file_list);
    
    
    signed char *input_char;
    short *input_half;
    //signed char *input_char_cpu=new signed char[4096];
    //short *input_float_cpu=new short[4096];
    
    
    cudaMallocManaged((void**)&input_char,sizeof(unsigned char)*4096);
    cudaMallocManaged((void**)&input_half,sizeof(short)*5100);
    
    readfile(input_char,file_list,4096);
    
    print_data_signed_char(input_char,0,4096);
    
    char2float_interlace(input_char,input_half,2,1024);
    
    
    
    cudaFree(input_char);
    cudaFree(input_half);
    
    
    return 0;
    
} 
