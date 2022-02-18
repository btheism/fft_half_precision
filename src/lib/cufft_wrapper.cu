#include <cuda_fp16.h>
#include <cufftXt.h>
#include<stdio.h>

#include <cuda_runtime.h>

#include<iostream>
//#define PRINT_INFO //打印函数调用信息

//注意，FFT变换要求fft_length必为2的N次幂
void fft_1d_half(void* input_half_void, void* output_half_complex_void, long long int fft_length, long long int fft_num)
{
        static long long int cached_fft_length=-1;
        static long long int cached_fft_num=-1;
        static cufftHandle plan;
        static int has_plan=0;
        
    #ifdef PRINT_INFO
        printf("Function fft_1d_half is called.\nfft_length=%lld,fft_num=%lld\n",
            fft_length,
            fft_num
        );
    #endif
        //传入含有-1的参数,则不执行FFT，而是销毁已有的计划释放内存
        if((fft_length==-1)or(fft_num==-1))
        {
            if(has_plan)
            {
                //删除计划
                cufftDestroy(plan);
                has_plan=0;
                #ifdef PRINT_INFO
                printf("Succeed destorying plan.\n");
                #endif
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
                    cufftDestroy(plan);
                    has_plan=0;
                    #ifdef PRINT_INFO
                    printf("Succeed destorying plan.\n");
                    #endif
                }
                cufftCreate(&plan);
                long long int fftdimarray[]={fft_length};
                long long int embed_sample[] = {0};
                long long int embed_freq[] = {0};
                size_t worksize;
                cufftXtMakePlanMany(plan, 1, fftdimarray, embed_sample,1,fft_length+2, CUDA_R_16F,embed_freq, 1,fft_length/2+1,CUDA_C_16F, fft_num, &worksize,CUDA_C_16F);
                has_plan=1;
                #ifdef PRINT_INFO
                printf("Succeed creating plan.\n");
                #endif
            }   
            cufftXtExec(plan, input_half_void, output_half_complex_void, CUFFT_FORWARD);
            cudaDeviceSynchronize();
            #ifdef PRINT_INFO
            printf("Succeed executing plan.\n");
            int error=cudaGetLastError();
            printf("Error code is %d\n",error);
            #endif
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
