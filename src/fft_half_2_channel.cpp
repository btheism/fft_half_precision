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
    double *average;
    //signed char *input_char_cpu=new signed char[4096];
    //short *input_float_cpu=new short[4096];
    
    
    cudaMallocManaged((void**)&input_char,sizeof(unsigned char)*4096);
    cudaMallocManaged((void**)&input_half,sizeof(short)*5100);
    cudaMallocManaged((void**)&average,sizeof(double)*1);
    float *input_float=(float *)input_half;
    readfile(input_char,file_list,4096);
    
    print_data_signed_char(input_char,0,4096);
    
    char2float_interlace(input_char,input_half,1024,2,1024);
    
    print_data_half(input_half,0,1026);
    //print_data_half(input_half,1026,2052);
    //print_data_half(input_half,2052,3078);
    //print_data_half(input_half,3078,4104);
    
    fft_1d_half((void*)input_half, (void*)input_half,1024,4);
 
    //print_data_half(input_half,0,1026);
    print_data_half(input_half,1026,2052);
    print_data_half(input_half,2052,3078);
    //print_data_half(input_half,3078,4104);
    
    complex_modulus_squared_interlace(input_half,input_half,1,0,1024,2,512);
    
    print_data_float_for_copy(input_float,1,513);
    channels_average((void*)(input_float+1),(void*)average,512,1,1,1);
    std::cout<<average[0]<<std::endl;
    //print_data_float(input_float,1539,2052);
    
    cudaFree(input_char);
    cudaFree(input_half);
    
    
    return 0;
    
}

