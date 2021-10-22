#include<cufft_wrapper.hpp>
#include <cuda_runtime.h>
#include <iostream>
int main(int argc, char *argv[]) {
    long long int fft_length=131072;
        for(int j=12288-10;j<=12288+10;j++)
        {
        std::cout<<"fft_length = "<<2*fft_length<<", batch = "<<j<<std::endl;
        std::cout<<test_fft_plan_memory_size(fft_length,j)/1024/1024<<"MB"<<std::endl;
        }
}
