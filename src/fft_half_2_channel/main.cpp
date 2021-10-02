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
    
    print_data_signed_char(input_char,0,4110);
    
    char2float_interlace(input_char,input_half,2,1024);
    
    print_data_half(input_half,0,4110);
    print_data_half_for_copy(input_half,0,2048);
    print_data_half_for_copy(input_half,2050,4098);
    
    fft_1d_half((void*)input_half, (void*)input_half,1024,4);
    
    print_data_half(input_half,0,4110);
    print_data_half_for_copy(input_half,0,2050);
    print_data_half_for_copy(input_half,2050,4100);

    
    cudaFree(input_char);
    cudaFree(input_half);
    
    
    
    //cudaMemcpy ( (void*)input_char,( void*) input_char_cpu, 1024*1,cudaMemcpyHostToDevice);
    
    /*readfile(input_char,file_list,32);
    print_data_signed_char(input_char,0,32);
    kernal_add_test(input_char,input_char,input_char,32);
    print_data_signed_char(input_char,0,32);
    
    kernal_parameter_pass_test(1,2,3,4,5.5,6.6);
    kernal_call_test();*/
    
    return 0;
    
}

