#include <cuda_fp16.h>
#include <cufftXt.h>
#include<stdio.h>

//注意，FFT变换要求fft_length必为2的N次幂
void fft_1d_half(void* input_half_void, void* output_half_complex_void, long long int fft_length, long long int fft_num)
{
         printf("Function fft_1d_half is being invoked\n,fft_length=%lld,fft_num=%lld\n",
           fft_length,
           fft_num
           );
        cufftHandle plan;
        cufftCreate(&plan);
        long long int fftdimarray[]={fft_length};
        long long int embed_sample[] = {0};
        long long int embed_freq[] = {0};
        size_t worksize;
        cufftXtMakePlanMany(plan, 1, fftdimarray, embed_sample,1,fft_length+2, CUDA_R_16F,embed_freq, 1,fft_length/2+1,CUDA_C_16F, fft_num, &worksize,CUDA_C_16F);
        cufftXtExec(plan, input_half_void, output_half_complex_void, CUFFT_FORWARD);
        cudaDeviceSynchronize();
}
