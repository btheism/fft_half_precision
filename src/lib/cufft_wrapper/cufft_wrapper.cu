#ifdef DEBUG
constexpr bool debug_flag=1;
#else
constexpr bool debug_flag=0;
#endif

#include <cuda_fp16.h>
#include <cufftXt.h>
#include<stdio.h>

#include <cuda_runtime.h>

#include<iostream>
#include<exception>

#include <cuda_macros.hpp>
//#define PRINT_INFO //打印函数调用信息

const char* cufftGetErrorString(cufftResult code)
{
    static const char* message[]=
    {
        "The cuFFT operation was successful",
        "cuFFT was passed an invalid plan handle",
        "cuFFT failed to allocate GPU or CPU memory",
        "No longer used",
        "User specified an invalid pointer or parameter",
        "Driver or internal cuFFT library error",
        "Failed to execute an FFT on the GPU",
        "The cuFFT library failed to initialize",
        "User specified an invalid transform size",
        "No longer used",
        "Missing parameters in call",
        "Execution of a plan was on different GPU than plan creation",
        "Internal plan database error",
        "No workspace has been provided prior to plan execution",
        "Function does not implement functionality for parameters given",
        "Used in previous versions",
        "Operation is not supported for parameters given",
    };
    return message[(int)code];
};


#define cufftErrchk(ans) { cufftAssert((ans), __FILE__, __LINE__);}

inline void cufftAssert(cufftResult code, const char *file, int line)
{
    if (code != CUFFT_SUCCESS) 
    {
        throw std::runtime_error("GPUassert"+std::string(cufftGetErrorString(code))+std::string(file));
        //printf("GPUassert: %s %s %d\n", cufftGetErrorString(code), file, line);
    }
    /*
    else
    {
        printf("GPUassert: %s %s %d\n", cufftGetErrorString(code), file, line);
    }
    */
};


//注意，FFT变换要求fft_length必为2的N次幂
//可以通过传入fft_length或num等于-1来销毁计划
void fft_1d_half(void* input_half_void, void* output_half_complex_void, long long int fft_length, long long int fft_num)
{
        static long long int cached_fft_length=-1;
        static long long int cached_fft_num=-1;
        static cufftHandle plan;
        static int has_plan=0;
        
    if(debug_flag){
        printf("Function fft_1d_half is called.\nfft_length=%lld,fft_num=%lld\n",
            fft_length,
            fft_num
        );}
        //传入含有-1的参数,则不执行FFT，而是销毁已有的计划释放内存
        if((fft_length==-1)or(fft_num==-1))
        {
            if(has_plan)
            {
                //删除计划
                cufftErrchk(cufftDestroy(plan));
                has_plan=0;
            }
        }
        else
        //执行FFT
        {   //旧计划与新计划的参数不同
            if(!(has_plan&&(cached_fft_length==fft_length)&&(cached_fft_num==fft_num)))
            {
                //记录创建的计划的参数
                cached_fft_length=fft_length;
                cached_fft_num=fft_num;
                //删除旧计划
                if(has_plan)
                {
                    cufftErrchk(cufftDestroy(plan));
                    has_plan=0;
                }
                cufftErrchk(cufftCreate(&plan));
                long long int fftdimarray[]={fft_length};
                long long int inembed[] = {0};
                long long int onembed[] = {0};
                size_t worksize;
                cufftErrchk(cufftXtMakePlanMany(plan, 1, fftdimarray, inembed,1,fft_length+2, CUDA_R_16F,onembed, 1,fft_length/2+1,CUDA_C_16F, fft_num, &worksize,CUDA_C_16F));
                has_plan=1;
            }   
            cufftErrchk(cufftXtExec(plan, input_half_void, output_half_complex_void, CUFFT_FORWARD));
            gpuErrchk(cudaDeviceSynchronize());
        }
}




void ifft_1d_half(void* input_half_complex_void, void* output_half_void, long long int ifft_length, long long int ifft_num)
{
        static long long int cached_ifft_length=-1;
        static long long int cached_ifft_num=-1;
        static cufftHandle plan;
        static int has_plan=0;
        
    if(debug_flag){
        printf("Function ifft_1d_half is called.\nifft_length=%lld,ifft_num=%lld\n",
            ifft_length,
            ifft_num
        );}
        
        //传入含有-1的参数,则不执行FFT，而是销毁已有的计划释放内存
        if((ifft_length==-1)or(ifft_num==-1))
        {
            if(has_plan)
            {
                //删除计划
                cufftErrchk(cufftDestroy(plan));
                has_plan=0;
            }
        }
        else
        //执行FFT
        {   //旧计划与新计划的参数不同
            if(!(has_plan&&(cached_ifft_length==ifft_length)&&(cached_ifft_num==ifft_num)))
            {
                //记录创建的计划的参数
                cached_ifft_length=ifft_length;
                cached_ifft_num=ifft_num;
                //删除旧计划
                if(has_plan)
                {
                    cufftErrchk(cufftDestroy(plan));
                    has_plan=0;
                }
                cufftErrchk(cufftCreate(&plan));
                long long int ifftdimarray[]={ifft_length};
                long long int iembed[] = {0};
                long long int oembed[] = {0};
                size_t worksize;
                cufftErrchk(cufftXtMakePlanMany(plan, 1, ifftdimarray, iembed,1,ifft_length/2+1, CUDA_C_16F,oembed, 1,ifft_length+2,CUDA_R_16F, ifft_num, &worksize,CUDA_C_16F));
                
                has_plan=1;
            }   
            cufftErrchk(cufftXtExec(plan, input_half_complex_void, output_half_void, CUFFT_INVERSE));
            gpuErrchk(cudaDeviceSynchronize());
        }
}

/*

void test_fft_plan_memory_size_step( long long int fft_length, long long int fft_num)
{
        printf("Function test_fft_plan_memory_size is being invoked\n,fft_length=%lld,fft_num=%lld\n",
           fft_length,
           fft_num
           );
        cufftHandle plan;
        cufftCreate(&plan);
        long long int fftdimarray[]={fft_length};
        long long int embed_sample[] = {0};
        long long int embed_freq[] = {0};
        size_t worksize;
        
        std::cout << "Press y to create this plan , other keys to skip."<<std::endl;
        
        char choice='\n';
        while(choice=='\n')
            std::cin.get(choice);
        
        if(choice=='y')
        {
        cufftXtMakePlanMany(plan, 1, fftdimarray, embed_sample,1,fft_length+2, CUDA_R_16F,embed_freq, 1,fft_length/2+1,CUDA_C_16F, fft_num, &worksize,CUDA_C_16F);
        }

        //cufftXtExec(plan, input_half_void, output_half_complex_void, CUFFT_FORWARD);
        //cudaDeviceSynchronize();
        std::cout << "Press y to destroy this plan , other keys to save this plan."<<std::endl;
        
        choice='\n';
        while(choice=='\n')
            std::cin.get(choice);
        
        if(choice=='y')
        {
            cufftDestroy(plan);
            cudaDeviceSynchronize();
        }
        
        int error=cudaGetLastError();
        printf("Error code is %d\n",error);
}

*/


long long int test_fft_plan_memory_size( long long int fft_length, long long int fft_num)
{
    size_t free_after,free_before,total;
    cufftHandle plan;
    cufftCreate(&plan);
    long long int fftdimarray[]={fft_length};
    long long int embed_sample[] = {0};
    long long int embed_freq[] = {0};
    size_t worksize;
    cufftXtMakePlanMany(plan, 1, fftdimarray, embed_sample,1,fft_length+2, CUDA_R_16F,embed_freq, 1,fft_length/2+1,CUDA_C_16F, fft_num, &worksize,CUDA_C_16F);
    cudaMemGetInfo(&free_before,&total);
    cufftDestroy(plan);
    cudaMemGetInfo(&free_after,&total);
    return (long long int)(free_after-free_before);
}
