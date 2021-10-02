#include <cuda_fp16.h>
#include<stdio.h>

//用于转换输入数据为float的函数,同时调整数据的布局(分离两个通道，在每个batch后补0便于进行原位fft变换)
//一个block的线程负责一个batch内数据的转置，grid数=batch数
__global__ void char2float_interlace_gpu(signed char*input_char,half *input_half,long long int fft_length,long long int thread_loop) 
{
    
    //计算不同block的内存偏移量
    long long int input_grid_offset = fft_length * 2 * blockIdx.x;
    long long int output_grid_offset = (fft_length + 2) * 2 * blockIdx.x;
    
    //input_block_offset = i * 1024 * 2 + threadIdx.x * 2;
    //output_block_offset = i * 1024 + threadIdx.x;
    //计算block内部的内存偏移量(该偏移量已与不同block的内存偏移量相加)
    long long int input_block_offset=threadIdx.x*2+input_grid_offset;
    long long int output_block_offset_A=threadIdx.x+output_grid_offset;
    long long int output_block_offset_B=output_block_offset_A+(fft_length+2);
    // __shared__ signed char sharedMemory[2048];
    //用1024个线程分128次(131072/1024)完成一个batch内的数据转置
    for (long long int i = 0; i < thread_loop; i++) {
        //printf("thread %d , block %d\n",threadIdx.x,blockIdx.x);

        //sharedMemory[threadIdx.x]=(signed char)input_char[input_block_offset];
        //sharedMemory[threadIdx.x+1]=(signed char)input_char[input_block_offset+1];
        input_half[output_block_offset_A] =
                (half)input_char[input_block_offset];
        /*printf("%d,%d,%lld,%lld,A\n",
               blockIdx.x,
               threadIdx.x,
               output_block_offset_A,
               input_block_offset
               );        
        */
        input_half[output_block_offset_B] =
                (half)input_char[input_block_offset+1] ;
        /*printf("%d,%d,%lld,%lld,B\n",
               blockIdx.x,
               threadIdx.x,
               output_block_offset_B,
               input_block_offset
               );     
        */
        input_block_offset+=2048;
        output_block_offset_A+=1024;
        output_block_offset_B+=1024;
    }
}

//对gpu函数的封装
void char2float_interlace(void* input_char_void,void* input_half_void,int batch,long long int fft_length)
{
    printf("Function char2float_interlace is being invoked\n,batch=%d , fft_length=%lld\n",
           batch,
           fft_length
           );
    //把数据从input_char_gpu复制到fft_real,并分离两个通道的数据
    dim3 char2float_blockSize(1024,1,1);
    dim3 char2float_gridSize(batch,1,1);
    signed char*input_char=(signed char*)input_char_void;
    half *input_half=(half*)input_half_void;
    long long int thread_loop=fft_length/1024;
    char2float_interlace_gpu<<<char2float_gridSize,char2float_blockSize>>>(input_char,input_half,fft_length,thread_loop);
    cudaDeviceSynchronize();
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
}








//以下为一些辅助的测试函数

//一个测试函数,利用一个加法测试kernal是否工作正常
__global__ void kernal_add_test_gpu(void *input_A_void,void *input_B_void,void *output_void) {
    signed char *input_A=(signed char *)input_A_void;
    signed char *input_B=(signed char *)input_B_void;
    signed char *output=(signed char *)output_void;
    //output[blockIdx.x]=input_A[blockIdx.x]+input_B[blockIdx.x];
    output[threadIdx.x]=input_A[threadIdx.x]+input_B[threadIdx.x];
    printf("This is thread %d , block %d\n",threadIdx.x,blockIdx.x);
    //printf("Block[%d] is %d\n",blockIdx.x,output[blockIdx.x]);
}

//对测试函数的封装
void kernal_add_test(void *input_A_void,void *input_B_void,void *output_void ,long long int input_length) {
    printf("Call function kernal_add_test\n");
    //dim3 kernal_add_test_blockSize(input_length,1,1);
    dim3 kernal_add_test_blockSize(input_length,1,1);
    dim3 kernal_add_test_gridSize(1,1,1);
    kernal_add_test_gpu<<<kernal_add_test_gridSize,kernal_add_test_blockSize>>>(input_A_void , input_B_void , output_void);
    cudaDeviceSynchronize();
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
}

//一个测试函数,测试向kernal函数传递参数的情况

__global__ void kernal_parameter_pass_test_gpu(signed char a,short b,int c,long long int d,float e,double f ) {
    
    printf("thread %d , block %d\n",threadIdx.x,blockIdx.x);
    printf("From device , signed char a=%d , short b=%d , int c=%d , long long int d=%lld , float e=%f , double f=%f\n",
            a,b,c,d,e,f
            );
}

//对测试函数的封装
void kernal_parameter_pass_test(signed char a,short b,int c,long long int d,float e,double f ) {
     printf("Call function kernal_parameter_test\n");
     printf("From host , signed char a=%d , short b=%d , int c=%d , long long int d=%lld , float e=%f , double f=%f\n",
            a,b,c,d,e,f
            );
     dim3 kernal_parameter_pass_test_blockSize(10,1,1);
     dim3 kernal_parameter_pass_test_gridSize(1,1,1);
     kernal_parameter_pass_test_gpu<<<kernal_parameter_pass_test_gridSize,kernal_parameter_pass_test_blockSize>>>(a,b,c,d,e,f);
     cudaDeviceSynchronize();
     //kernal_parameter_test_gpu<<<1,1>>>(a,b,c,d,e,f);
     int error=cudaGetLastError();
     printf("Error code is %d\n",error);
}

//一个测试函数,测试调用kernal函数的情况

__global__ void kernal_call_test_gpu(void) {
    printf("This is thread %d , block %d\n",threadIdx.x,blockIdx.x);
    printf("Hello\n");
}


void kernal_call_test(void) {
     printf("Call function kernal_call_test\n");
     dim3 kernal_call_test_blockSize(10,1,1);
     dim3 kernal_call_test_gridSize(1,1,1);
     kernal_call_test_gpu<<<kernal_call_test_gridSize,kernal_call_test_blockSize>>>();
     cudaDeviceSynchronize();
     int error=cudaGetLastError();
     printf("Error code is %d\n",error);
}





