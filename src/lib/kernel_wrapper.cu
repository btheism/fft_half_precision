#include <cuda_fp16.h>
#include<stdio.h>
//内核编写的大致原则：
//涉及到多个fft区间的，每个区间用一个block处理，该block内可包含很多线程，故block数对应区间数。
//涉及到单个fft区间的，则进一步分割这个区间。
//每个内核函数都由一个同名的(去掉内核函数后缀的gpu)host函数包装。

//用于转换双通道输入数据为float的函数,同时调整数据的布局(分离两个通道，在每个读入的预定进行fft变换的数组补充空隙便于进行原位fft变换)，fft区间数=batch*2
//一个block的线程负责一个batch内数据的转置,生成两组需要fft变换的数,grid数=batch数
__global__ void char2float_interlace_gpu(signed char*input_char,half *input_half,int fft_length,int thread_loop,int thread_num) 
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
    for (int i = 0; i < thread_loop; i++) {

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
        input_block_offset+=2*thread_num;
        output_block_offset_A+=thread_num;
        output_block_offset_B+=thread_num;
    }
}

//对gpu函数的封装
//thread_num最大等于fft_length
void char2float_interlace(void* input_char_void,void* input_half_void,int fft_length,int batch,int thread_num)
{
    dim3 char2float_blockSize(thread_num,1,1);
    dim3 char2float_gridSize(batch,1,1);
    signed char*input_char=(signed char*)input_char_void;
    half *input_half=(half*)input_half_void;
    int thread_loop=fft_length/thread_num;
    printf("Function char2float_interlace is being invoked\n,batch=%d , fft_length=%d , thread_num=%d , thread_loop=%d\n",
           batch,
           fft_length,
           thread_num,
           thread_loop
           );
    char2float_interlace_gpu<<<char2float_gridSize,char2float_blockSize>>>(input_char,input_half,fft_length,thread_loop,thread_num);
    cudaDeviceSynchronize();
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
}


//用于转换单通道输入数据为float的函数，在每个读入的预定进行fft变换的数组补充空隙便于进行原位fft变换
//一个block的线程负责一个batch内数据的转置，grid数=batch数
__global__ void char2float_gpu(signed char*input_char,half *input_half,int fft_length,int thread_loop,int thread_num) 
{
    
    //计算不同block的内存偏移量
    long long int input_grid_offset = fft_length * blockIdx.x;
    long long int output_grid_offset = (fft_length + 2) * blockIdx.x;
    
    //input_block_offset = i * 1024 * 2 + threadIdx.x * 2;
    //output_block_offset = i * 1024 + threadIdx.x;
    //计算block内部的内存偏移量(该偏移量已与不同block的内存偏移量相加)
    long long int input_block_offset=threadIdx.x+input_grid_offset;
    long long int output_block_offset=threadIdx.x+output_grid_offset;
    // __shared__ signed char sharedMemory[2048];
    //用1024个线程分128次(131072/1024)完成一个batch内的数据转置
    for (int i = 0; i < thread_loop; i++) {

        //sharedMemory[threadIdx.x]=(signed char)input_char[input_block_offset];
        //sharedMemory[threadIdx.x+1]=(signed char)input_char[input_block_offset+1];
        input_half[output_block_offset] =
                (half)input_char[input_block_offset];
        /*printf("%d,%d,%lld,%lld,A\n",
               blockIdx.x,
               threadIdx.x,
               output_block_offset_A,
               input_block_offset
               );        
        */
        input_block_offset+=thread_num;
        output_block_offset+=thread_num;
    }
}

//对gpu函数的封装
//thread_num最大等于fft_length
void char2float(void* input_char_void,void* input_half_void,int fft_length,int batch,int thread_num)
{
    dim3 char2float_blockSize(thread_num,1,1);
    dim3 char2float_gridSize(batch,1,1);
    signed char*input_char=(signed char*)input_char_void;
    half *input_half=(half*)input_half_void;
    int thread_loop=fft_length/thread_num;
    printf("Function char2float is being invoked\n,batch=%d , fft_length=%d , thread_num=%d , thread_loop=%d\n",
           batch,
           fft_length,
           thread_num,
           thread_loop
           );
    char2float_interlace_gpu<<<char2float_gridSize,char2float_blockSize>>>(input_char,input_half,fft_length,thread_loop,thread_num);
    cudaDeviceSynchronize();
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
}

//计算FFT变换后的结果的模方的函数,计算前为32bit复数(2*16bit)，计算后为32bits实数。
//该函数用于双通道数据，需要指定两个通道的权重
//一个block的线程负责一个fft区间内数据的计算，grid数=batch数

__global__ void complex_modulus_squared_gpu(half2 *complex_number,float *float_number,int channel_num ,int thread_loop,int thread_num) {
    //计算偏移量,由于不同的fft变换结果之间有间隔(通道0)，因此此处需要乘channel_num+1
    long long int index= (threadIdx.x + blockIdx.x*(channel_num+1));;
    for(int i=0;i<thread_loop;i++)
    {
        /*
        printf("%d,%d,%lld,%f,%f\n",
               blockIdx.x,
               threadIdx.x,
               (float)complex_number[block_offset].x,
               (float)complex_number[block_offset].y
               );  
        */

        //计算复数的模方
        if(__hisinf(complex_number[index].x))
            printf("%lld,%f,x",index,(float)complex_number[index].x);
        if(__hisinf(complex_number[index].y))
            printf("%lld,%f,x",index,(float)complex_number[index].y);
        /*
        if(__hisinf(complex_number[block_offset_A].y) ||__hisnan(complex_number[block_offset_A].y))
            printf("E,%d,A.y",block_offset_A);
        */
        float_number[index]=((float)complex_number[index].x*(float)complex_number[index].x+(float)complex_number[index].y*(float)complex_number[index].y);
        /*printf("%d,%d,%lld,%f\n",
               blockIdx.x,
               threadIdx.x,
               index,
               real_number[index]
               );        
        */
        index+=thread_num;
    }
}

//对gpu函数的封装
//thread_num最大等于channel_num,fft_num代表需要计算的fft区间的数量,单通道的情况下等于batch,双通道的情况下等于batch*2
void complex_modulus_squared(void *complex_number_void,void *float_number_void, int channel_num,int batch,int thread_num)
{
    dim3 complex_modulus_squared_blocksize(thread_num,1,1);
    dim3 complex_modulus_squared_gridsize(batch,1,1);
    half2 *complex_number=(half2*)complex_number_void;
    float *float_number=(float*)float_number_void;
    int thread_loop=channel_num/thread_num;
    printf("Function complex_modulus_squared is being invoked.\vbatch=%d , channel_num=%d , thread_num=%d , thread_loop=%d.\n",
           batch,
           channel_num,
           thread_num,
           thread_loop);
    complex_modulus_squared_gpu<<<complex_modulus_squared_gridsize,complex_modulus_squared_blocksize>>>
    (complex_number,float_number,channel_num,thread_loop,thread_num);
    cudaDeviceSynchronize();
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
}


//计算FFT变换后的结果的模方的函数,计算前为32bit复数(2*16bit)，计算后为32bits实数。
//一个block的线程负责一个fft区间内数据的计算，grid数=batch数

__global__ void complex_modulus_squared_interlace_gpu(half2 *complex_number,float *float_number,int channel_num, int factor_A, int factor_B, int thread_loop,int thread_num) {
    //计算单个通道的的偏移量,由于不同的fft变换结果之间有间隔(通道0)，因此此处需要乘channel_num+1
    long long int index_A= (threadIdx.x + blockIdx.x * (channel_num+1)*2);;
    long long int index_B= index_A+channel_num+1;
    for(int i=0;i<thread_loop;i++)
    {
        /*
        printf("%d,%d,%lld,%f,%f\n",
               blockIdx.x,
               threadIdx.x,
               index_A,
               (float)complex_number[index_A].x,
               (float)complex_number[index_A].y
               );  
                
        printf("%d,%d,%lld,%f,%f\n",
               blockIdx.x,
               threadIdx.x,
               index_B,
               (float)complex_number[index_B].x,
               (float)complex_number[index_B].y
               );  
        */

        //计算复数的模方
        if(__hisinf(complex_number[index_A].x))
            printf("%lld,%f,x,A",index_A,(float)complex_number[index_A].x);
        if(__hisinf(complex_number[index_A].y))
            printf("%lld,%f,x,A",index_A,(float)complex_number[index_A].y);

        float_number[index_A]=((float)complex_number[index_A].x*(float)complex_number[index_A].x+(float)complex_number[index_A].y*(float)complex_number[index_A].y);
        float_number[index_A]*=factor_A;
        if(__hisinf(complex_number[index_B].x))
            printf("%lld,%f,x,B",index_B,(float)complex_number[index_B].x);
        if(__hisinf(complex_number[index_B].y))
            printf("%lld,%f,x,B",index_B,(float)complex_number[index_B].y);
         float_number[index_B]=((float)complex_number[index_B].x*(float)complex_number[index_B].x+(float)complex_number[index_B].y*(float)complex_number[index_B].y);
         float_number[index_B]*=factor_B;
         float_number[index_A]+=float_number[index_B];
        
        /*printf("%d,%d,%lld,%f\n",
               blockIdx.x,
               threadIdx.x,
               index,
               real_number[index_A]
               );        
        */
        index_A+=thread_num;
        index_B+=thread_num;
    }
}


//对gpu函数的封装
//thread_num最大等于channel_num,fft_num代表需要计算的fft区间的数量,单通道的情况下等于batch,双通道的情况下等于batch*2
void complex_modulus_squared_interlace(void *complex_number_void,void *float_number_void, int channel_num, int factor_A, int factor_B, int batch,int thread_num)
{
    dim3 complex_modulus_squared_interlace_blocksize(thread_num,1,1);
    dim3 complex_modulus_squared_interlace_gridsize(batch,1,1);
    half2 *complex_number=(half2*)complex_number_void;
    float *float_number=(float*)float_number_void;
    int thread_loop=channel_num/thread_num;
    printf("Function complex_modulus_squared_interlace is being invoked.\nbatch=%d , channel_num=%d , thread_num=%d , thread_loop=%d.\n",
           batch,
           channel_num,
           thread_num,
           thread_loop);
    complex_modulus_squared_interlace_gpu<<<complex_modulus_squared_interlace_gridsize,complex_modulus_squared_interlace_blocksize>>>
    (complex_number,float_number,channel_num,factor_A,factor_B,thread_loop,thread_num);
    cudaDeviceSynchronize();
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
}

//计算初始各通道的平均值的函数
__global__ void channels_average_gpu(float *input_data,double *average_data,int window_size,int batch_interval) {
    long long int index = (threadIdx.x + blockIdx.x * blockDim.x);
    average_data[index]=0;
    for(int step=0;step<window_size;step++)
    {
       /* printf("%d,%d,%lld,%lld,%f\n",
               blockIdx.x,
               threadIdx.x,
               index,
               index+step*interval,
               input_data[index+step*interval]);*/
        average_data[index]+=(double)input_data[index+step*batch_interval];
    }
    average_data[index]/=(double)window_size;
}


//对gpu函数的封装,thread_num不超过channel_num
void channels_average(void *input_data_void,void *average_data_void,int window_size,int batch_interval,int channel_num,int thread_num)
{
    dim3 channels_average_blocksize(thread_num,1,1);
    dim3 channels_average_gridsize(channel_num/thread_num,1,1);
    float *input_data=(float *)input_data_void;
    double *average_data=(double*)average_data_void;
    printf("Function channels_average is being invoked.\nwindow_size=%d , batch_interval=%d , channel_num=%d , thread_num=%d.\n",
           window_size,
           batch_interval,
           channel_num,
           thread_num
           );
    channels_average_gpu<<<channels_average_gridsize,channels_average_blocksize>>>(input_data,average_data,window_size,batch_interval);
    cudaDeviceSynchronize();
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
}


__global__ void compress_gpu(double *average_data,float *uncompressed_data_head,float *uncompressed_data_tail,unsigned char*compressed_data, int batch_interval ,int batch, int step, int output_channel_num ,int window_size) {
    long long int output_index= threadIdx.x + blockIdx.x * blockDim.x;
    long long int input_index=output_index*8;
    long long int average_index=input_index;
    double head_sum;
    double tail_sum;
    for(long long int batch_num=0;batch_num<batch;batch_num+=step)
    {
        compressed_data[output_index]=0;
        for(long long int bit_step=0;bit_step<8;bit_step++)
        {
            for(long long int sum_step=0;sum_step<step;sum_step++)
            {
                head_sum+=(double)uncompressed_data_head[input_index+bit_step];
                tail_sum+=(double)uncompressed_data_head[input_index+bit_step];
                input_index+=batch_interval;
            }
            compressed_data[output_index]+=(((head_sum/step)>average_data[average_index])?1:0)<<bit_step;
            average_data[average_index+bit_step]+=(tail_sum-head_sum)/(double)window_size;
            head_sum=0;
            tail_sum=0;
        }
        output_index+=output_channel_num;
    }
}

//对gpu函数的封装, thread_num不超过channel_num/8, step指定时间分辨率
void compress(void *average_data_void,float *uncompressed_data_head_void,void *compressed_data_void, int batch_interval ,int batch, int step, int channel_num ,int window_size,int thread_num)
{
    double *average_data=(double*)average_data_void;
    float *uncompressed_data_head=(float*)uncompressed_data_head_void;
    float *uncompressed_data_tail=uncompressed_data_head+=window_size*batch_interval;
    unsigned char *compressed_data=(unsigned char*)compressed_data_void;
    dim3 compress_blocksize(thread_num,1,1);
    dim3 compress_gridsize(channel_num/thread_num/8,1,1);
    int output_channel_num=channel_num/8;
    printf("Function compress is being invoked.\steps=%d , channel_num=%d , thread_num=%d.\n",
           step,
           channel_num,
           thread_num
           );
    compress_gpu<<<compress_gridsize,compress_blocksize>>>(average_data,uncompressed_data_head,uncompressed_data_tail,compressed_data,batch_interval,batch,step,output_channel_num,window_size);
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





