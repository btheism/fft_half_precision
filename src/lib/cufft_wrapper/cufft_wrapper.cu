#include <cuda_fp16.h>
#include <cufftXt.h>
#include<stdio.h>

void fft_1d_half(void* input_half_void, void* output_half_complex_void, long long int fft_length, long long int batch)
{
         printf("Function fft_1d_half is being invoked\n,fft_length=%lld,batch=%lld\n",
           fft_length,
           batch
           );
        cufftHandle plan;
        cufftCreate(&plan);
        long long int fftdimarray[]={fft_length};
        long long int embed_sample[] = {0};
        long long int embed_freq[] = {0};
        size_t worksize;
        cufftXtMakePlanMany(plan, 1, fftdimarray, embed_sample,1,fft_length+2, CUDA_R_16F,embed_freq, 1,fft_length/2+1,CUDA_C_16F, batch, &worksize,CUDA_C_16F);
        cufftXtExec(plan, input_half_void, output_half_complex_void, CUFFT_FORWARD);
        cudaDeviceSynchronize();
}
