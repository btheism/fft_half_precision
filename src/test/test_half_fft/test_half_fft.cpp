#include <cuda_runtime.h>
#include <other_function_library.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>

int main(int argc, char *argv[]) {
    short *input_half;
    cudaMallocManaged((void**)&input_half,sizeof(unsigned char)*100);
    
    float input_float[20]={12.,-119.,-90.,123.,-72.,102.,98.,11.,0.,0.,101.,65.,53.,-14.,76.,-72.,-19.,112.,0.,0.,};
    float_2_half((void*)input_float,(void*)input_half,0,20);
    print_data_half_for_copy(input_half,0,10,8);
    fft_1d_half((void*)input_half, (void*)input_half,8,2);
    print_data_half_for_copy(input_half,0,10,8);
    ifft_1d_half((void*)input_half, (void*)input_half,8,2);
    print_data_half_for_copy(input_half,0,10,8);
    
    float input_float_B[34]={1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.,12.,43.};
    float_2_half((void*)input_float_B,(void*)input_half,0,34);
    print_data_half_for_copy(input_half,0,34,8);
    fft_1d_half((void*)input_half, (void*)input_half,32,1);
    print_data_half_for_copy(input_half,0,34,8);
    ifft_1d_half((void*)input_half,(void*)input_half,32,1);
    print_data_half_for_copy(input_half,0,34,8);
    //complex_multiply((void *)input_half , 1.0 , 1.0 , 1.0/32.0 , 1.0 , 8 , 2 , 1);
    //print_data_half_for_copy(input_half,0,34,8);
    complex_add((void *)input_half , (void *)input_half , 8 , 2 , 1);
    print_data_half_for_copy(input_half,0,34,8);
    
    /*
    float input_float_C[10]={0.,0.,1.,2.,3.,4.,5.,6.,7.,8.};
    float_2_half((void*)input_float_C,(void*)input_half,0,10);
    print_data_half_for_copy(input_half,0,10,8);
    ifft_1d_half((void*)input_half, (void*)input_half,8,2);
    print_data_half_for_copy(input_half,0,10,8);
    
    float input_float_D[10]={1.,0.,1.,2.,3.,4.,5.,6.,7.,8.};
    float_2_half((void*)input_float_D,(void*)input_half,0,10);
    print_data_half_for_copy(input_half,0,10,8);
    ifft_1d_half((void*)input_half, (void*)input_half,8,2);
    print_data_half_for_copy(input_half,0,10,8);
    
    float input_float_E[10]={1.,1.,1.,2.,3.,4.,5.,6.,7.,8.};
    float_2_half((void*)input_float_E,(void*)input_half,0,10);
    print_data_half_for_copy(input_half,0,10,8);
    ifft_1d_half((void*)input_half, (void*)input_half,8,2);
    print_data_half_for_copy(input_half,0,10,8);
    */
}
