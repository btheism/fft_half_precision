#include <cuda_fp16.h>
#include<stdio.h>

#define SCALE 256 //定义对输入数据的压缩系数
//#define PRINT_INFO //是否打印函数调用信息
#define TEST_INF //是否检测fft的变换结果出现溢出

//内核编写的大致原则：
//涉及到多个fft区间的，每个区间用一个block处理，该block内可包含很多线程，故block数对应区间数。
//涉及到单个fft区间的，则进一步分割这个区间。
//每个内核函数都由一个同名的(去掉内核函数后缀的gpu)host函数包装。

//用于转换双通道输入数据为float的函数,同时调整数据的布局(分离两个通道，在每个读入的预定进行fft变换的数组补充空隙便于进行原位fft变换)，fft区间数=batch*2
//一个block的线程负责一个batch内数据的转置,生成两组需要fft变换的数,grid数=batch数
__global__ void char2float_interlace_gpu(signed char*input_char,half *input_half,long long int fft_length,long long int thread_loop,long long int thread_num) 
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
    //半精度浮点数的相对精度约为1/1000,把正负128以内的数据归一化到正负0.5，既有足够的精度表示数据，又保证了fft变换不会出现溢出(131072*0.5=65536,正好是半精度浮点数表示范围的上界)
        //sharedMemory[threadIdx.x]=(signed char)input_char[input_block_offset];
        //sharedMemory[threadIdx.x+1]=(signed char)input_char[input_block_offset+1];
        input_half[output_block_offset_A] =(half)((float)input_char[input_block_offset]/(float)SCALE);
        //input_half[output_block_offset_A] =(half)((float)input_char[input_block_offset]);
        /*printf("%d,%d,%lld,%lld,A\n",
               blockIdx.x,
               threadIdx.x,
               output_block_offset_A,
               input_block_offset
               );        
        */
        input_half[output_block_offset_B] =(half)((float)input_char[input_block_offset+1]/(float)SCALE);
        //input_half[output_block_offset_B] =(half)((float)input_char[input_block_offset+1]);
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
void char2float_interlace(void* input_char_void,void* input_half_void,long long int fft_length,long long int batch,long long int thread_num)
{
    dim3 char2float_interlace_blocksize(thread_num,1,1);
    dim3 char2float_interlace_gridsize(batch,1,1);
    signed char*input_char=(signed char*)input_char_void;
    half *input_half=(half*)input_half_void;
    long long int thread_loop=fft_length/thread_num;
#ifdef PRINT_INFO
    printf("Function char2float_interlace is called\n,batch=%lld , fft_length=%lld , thread_num=%lld , thread_loop=%lld\n",
           batch,
           fft_length,
           thread_num,
           thread_loop
           );
#endif
    char2float_interlace_gpu<<<char2float_interlace_gridsize,char2float_interlace_blocksize>>>(input_char,input_half,fft_length,thread_loop,thread_num);
    cudaDeviceSynchronize();
#ifdef PRINT_INFO
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
#endif

}


//用于转换单通道输入数据为float的函数，在每个读入的预定进行fft变换的数组补充空隙便于进行原位fft变换
//一个block的线程负责一个batch内数据的转置，grid数=batch数
__global__ void char2float_gpu(signed char*input_char,half *input_half,long long int fft_length,long long int thread_loop,long long int thread_num) 
{
    
    //计算不同block的内存偏移量
    long long int input_grid_offset = fft_length * blockIdx.x;
    long long int output_grid_offset = (fft_length + 2) * blockIdx.x;
    
    //计算block内部的内存偏移量(该偏移量已与不同block的内存偏移量相加)
    long long int input_block_offset=threadIdx.x+input_grid_offset;
    long long int output_block_offset=threadIdx.x+output_grid_offset;

    for (int i = 0; i < thread_loop; i++) {

        //sharedMemory[threadIdx.x]=(signed char)input_char[input_block_offset];
        //sharedMemory[threadIdx.x+1]=(signed char)input_char[input_block_offset+1];
        input_half[output_block_offset] =(half)((float)input_char[input_block_offset]/(float)SCALE);
        /*printf("%d,%d,%lld,%lld,A\n",
               blockIdx.x,
               threadIdx.x,
               output_block_offset,
               input_block_offset
               );        
        */
        input_block_offset+=thread_num;
        output_block_offset+=thread_num;
    }
}

//对gpu函数的封装
//thread_num最大等于fft_length
void char2float(void* input_char_void,void* input_half_void,long long int fft_length,long long int batch,long long int thread_num)
{
    dim3 char2float_blocksize(thread_num,1,1);
    dim3 char2float_gridsize(batch,1,1);
    signed char*input_char=(signed char*)input_char_void;
    half *input_half=(half*)input_half_void;
    long long int thread_loop=fft_length/thread_num;
#ifdef PRINT_INFO
    printf("Function char2float is called\n,batch=%lld , fft_length=%lld , thread_num=%lld , thread_loop=%lld\n",
           batch,
           fft_length,
           thread_num,
           thread_loop
           );
#endif
    char2float_gpu<<<char2float_gridsize,char2float_blocksize>>>(input_char,input_half,fft_length,thread_loop,thread_num);
    cudaDeviceSynchronize();
#ifdef PRINT_INFO
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
#endif
}

//以下两个函数与之前的两个函数类似,只是按照倒序读取输入数据.用于对末尾的输入数据进行反射,差别仅在于input_block_offset多一个负号,此外;
//在双通道数据的情况下,两个通道的先后次序也需要交换,另外,input_block_offset需要不断减小
//输入的地址为需要反射的数据末尾-1;

//用于转换双通道输入数据为float的函数,同时调整数据的布局(分离两个通道，在每个读入的预定进行fft变换的数组补充空隙便于进行原位fft变换)，fft区间数=batch*2
//一个block的线程负责一个batch内数据的转置,生成两组需要fft变换的数,grid数=batch数
__global__ void char2float_interlace_reflect_gpu(signed char*input_char,half *input_half,long long int fft_length,long long int thread_loop,long long int thread_num) 
{
    
    //计算不同block的内存偏移量
    long long int input_grid_offset = fft_length * 2 * blockIdx.x;
    long long int output_grid_offset = (fft_length + 2) * 2 * blockIdx.x;
    
    //input_block_offset = i * 1024 * 2 + threadIdx.x * 2;
    //output_block_offset = i * 1024 + threadIdx.x;
    //计算block内部的内存偏移量(该偏移量已与不同block的内存偏移量相加)
    long long int input_block_offset=-(threadIdx.x*2+input_grid_offset);
    long long int output_block_offset_A=threadIdx.x+output_grid_offset;
    long long int output_block_offset_B=output_block_offset_A+(fft_length+2);
    // __shared__ signed char sharedMemory[2048];
    for (int i = 0; i < thread_loop; i++) {
    //半精度浮点数的相对精度约为1/1000,把正负128以内的数据归一化到正负0.5，既有足够的精度表示数据，又保证了fft变换不会出现溢出(131072*0.5=65536,正好是半精度浮点数表示范围的上界)
        //sharedMemory[threadIdx.x]=(signed char)input_char[input_block_offset];
        //sharedMemory[threadIdx.x+1]=(signed char)input_char[input_block_offset+1];
        input_half[output_block_offset_B] =(half)((float)input_char[input_block_offset]/(float)SCALE);
        //input_half[output_block_offset_B] =(half)(float)input_char[input_block_offset];
        /*printf("%d,%d,%lld,%lld,A\n",
               blockIdx.x,
               threadIdx.x,
               output_block_offset_A,
               input_block_offset
               );        
        */
        input_half[output_block_offset_A] =(half)((float)input_char[input_block_offset-1]/(float)SCALE);
        //input_half[output_block_offset_A] =(half)(float)input_char[input_block_offset-1];
        /*printf("%d,%d,%lld,%lld,B\n",
               blockIdx.x,
               threadIdx.x,
               output_block_offset_B,
               input_block_offset
               );     
        */
        input_block_offset-=2*thread_num;
        output_block_offset_A+=thread_num;
        output_block_offset_B+=thread_num;
    }
}

//对gpu函数的封装
//thread_num最大等于fft_length
void char2float_interlace_reflect(void* input_char_void,void* input_half_void,long long int fft_length,long long int batch,long long int thread_num)
{
    dim3 char2float_interlace_reflect_blocksize(thread_num,1,1);
    dim3 char2float_interlace_reflect_gridsize(batch,1,1);
    signed char*input_char=(signed char*)input_char_void;
    half *input_half=(half*)input_half_void;
    long long int thread_loop=fft_length/thread_num;
#ifdef PRINT_INFO
    printf("Function char2float_interlace_reflect is called\n,batch=%lld , fft_length=%lld , thread_num=%lld , thread_loop=%lld\n",
           batch,
           fft_length,
           thread_num,
           thread_loop
           );
#endif
    char2float_interlace_reflect_gpu<<<char2float_interlace_reflect_gridsize,char2float_interlace_reflect_blocksize>>>(input_char,input_half,fft_length,thread_loop,thread_num);
    cudaDeviceSynchronize();
#ifdef PRINT_INFO
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
#endif
}


//用于转换单通道输入数据为float的函数，在每个读入的预定进行fft变换的数组补充空隙便于进行原位fft变换
//一个block的线程负责一个batch内数据的转置，grid数=batch数
__global__ void char2float_reflect_gpu(signed char*input_char,half *input_half,long long int fft_length,long long int thread_loop,long long int thread_num) 
{
    
    //计算不同block的内存偏移量
    long long int input_grid_offset = fft_length * blockIdx.x;
    long long int output_grid_offset = (fft_length + 2) * blockIdx.x;
    
    //input_block_offset = i * 1024 * 2 + threadIdx.x * 2;
    //output_block_offset = i * 1024 + threadIdx.x;
    //计算block内部的内存偏移量(该偏移量已与不同block的内存偏移量相加)
    long long int input_block_offset=-(threadIdx.x+input_grid_offset);
    long long int output_block_offset=threadIdx.x+output_grid_offset;
    // __shared__ signed char sharedMemory[2048];
    //用1024个线程分128次(131072/1024)完成一个batch内的数据转置
    for (int i = 0; i < thread_loop; i++) {

        //sharedMemory[threadIdx.x]=(signed char)input_char[input_block_offset];
        //sharedMemory[threadIdx.x+1]=(signed char)input_char[input_block_offset+1];
        input_half[output_block_offset] =(half)((float)input_char[input_block_offset]/(float)SCALE);
        //input_half[output_block_offset] =(half)((float)input_char[input_block_offset]);
        /*printf("%d,%d,%lld,%lld,A\n",
               blockIdx.x,
               threadIdx.x,
               output_block_offset,
               input_block_offset
               );        
        */
        input_block_offset-=thread_num;
        output_block_offset+=thread_num;
    }
}


//对gpu函数的封装
//thread_num最大等于fft_length
void char2float_reflect(void* input_char_void,void* input_half_void,long long int fft_length,long long int batch,long long int thread_num)
{
    dim3 char2float_reflect_blocksize(thread_num,1,1);
    dim3 char2float_reflect_gridsize(batch,1,1);
    signed char*input_char=(signed char*)input_char_void;
    half *input_half=(half*)input_half_void;
    long long int thread_loop=fft_length/thread_num;
#ifdef PRINT_INFO
    printf("Function char2float_reflect is called\n,batch=%lld , fft_length=%lld , thread_num=%lld , thread_loop=%lld\n",
           batch,
           fft_length,
           thread_num,
           thread_loop
           );
#endif
    char2float_reflect_gpu<<<char2float_reflect_gridsize,char2float_reflect_blocksize>>>(input_char,input_half,fft_length,thread_loop,thread_num);
    cudaDeviceSynchronize();
#ifdef PRINT_INFO
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
#endif
}


//计算FFT变换后的结果的模方的函数,计算前为32bit复数(2*16bit)，计算后为32bits实数。
//该函数用于双通道数据，需要指定两个通道的权重
//一个block的线程负责一个fft区间内数据的计算，grid数=batch数

__global__ void complex_modulus_squared_gpu(half2 *complex_number,float *float_number,long long int channel_num ,long long int thread_loop,long long int thread_num) {
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
#ifdef TEST_INF
        if(__hisinf(complex_number[index].x))
            printf("%lld,%f,x",index,(float)complex_number[index].x);
        if(__hisinf(complex_number[index].y))
            printf("%lld,%f,x",index,(float)complex_number[index].y);
#endif
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
void complex_modulus_squared(void *complex_number_void,void *float_number_void, long long int channel_num,long long int batch,long long int thread_num)
{
    dim3 complex_modulus_squared_blocksize(thread_num,1,1);
    dim3 complex_modulus_squared_gridsize(batch,1,1);
    half2 *complex_number=(half2*)complex_number_void;
    float *float_number=(float*)float_number_void;
    complex_number++;
    float_number++;
    int thread_loop=channel_num/thread_num;
#ifdef PRINT_INFO
    printf("Function complex_modulus_squared is called.\nbatch=%lld , channel_num=%lld , thread_num=%lld , thread_loop=%lld.\n",
           batch,
           channel_num,
           thread_num,
           thread_loop);
#endif
    complex_modulus_squared_gpu<<<complex_modulus_squared_gridsize,complex_modulus_squared_blocksize>>>
    (complex_number,float_number,channel_num,thread_loop,thread_num);
    cudaDeviceSynchronize();
#ifdef PRINT_INFO
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
#endif
}


//计算FFT变换后的结果的模方的函数,计算前为32bit复数(2*16bit)，计算后为32bits实数。
//一个block的线程负责一个fft区间内数据的计算，grid数=batch数

__global__ void complex_modulus_squared_interlace_gpu(half2 *complex_number,float *float_number,long long int channel_num, float factor_A, float factor_B, long long  int thread_loop,long long int thread_num) {
    //计算单个通道的的偏移量,由于不同的fft变换结果之间有间隔(通道0)，因此此处需要乘channel_num+1
    long long int index_A= (threadIdx.x + blockIdx.x * (channel_num+1)*2);
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
#ifdef TEST_INF
        if(__hisinf(complex_number[index_A].x))
            printf("%lld,%f,x,A\n",index_A,(float)complex_number[index_A].x);
        if(__hisinf(complex_number[index_A].y))
            printf("%lld,%f,x,A\n",index_A,(float)complex_number[index_A].y);
#endif
        float_number[index_A]=((float)complex_number[index_A].x*(float)complex_number[index_A].x+(float)complex_number[index_A].y*(float)complex_number[index_A].y);
        float_number[index_A]*=factor_A;
#ifdef TEST_INF
        if(__hisinf(complex_number[index_B].x))
            printf("%lld,%f,x,B\n",index_B,(float)complex_number[index_B].x);
        if(__hisinf(complex_number[index_B].y))
            printf("%lld,%f,x,B\n",index_B,(float)complex_number[index_B].y);
#endif
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
void complex_modulus_squared_interlace(void *complex_number_void,void *float_number_void, long long int channel_num, float factor_A, float factor_B, long long int batch,long long int thread_num)
{
    dim3 complex_modulus_squared_interlace_blocksize(thread_num,1,1);
    dim3 complex_modulus_squared_interlace_gridsize(batch,1,1);
    half2 *complex_number=(half2*)complex_number_void;
    float *float_number=(float*)float_number_void;
    //printf("complex_number=%p,float_number=%p\n",complex_number,float_number);
    //printf("size of half2=%lld\n",sizeof(half2));
    complex_number++;
    float_number++;
    //printf("complex_number=%p,float_number=%p\n",complex_number,float_number);
    long long int thread_loop=channel_num/thread_num;
#ifdef PRINT_INFO
    printf("Function complex_modulus_squared_interlace is called.\nbatch=%lld , channel_num=%lld , thread_num=%lld , thread_loop=%lld.\n",
           batch,
           channel_num,
           thread_num,
           thread_loop);
#endif
    complex_modulus_squared_interlace_gpu<<<complex_modulus_squared_interlace_gridsize,complex_modulus_squared_interlace_blocksize>>>
    (complex_number,float_number,channel_num,factor_A,factor_B,thread_loop,thread_num);
    cudaDeviceSynchronize();
#ifdef PRINT_INFO
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
#endif
}

//计算初始各通道的平均值的函数
__global__ void channels_sum_gpu(float *input_data,double *sum_data,long long int window_size, double coefficient,long long int batch_interval) {
    long long int index = (threadIdx.x + blockIdx.x * blockDim.x);
    sum_data[index]=0;
    for(int step=0;step<window_size;step++)
    {
       /* printf("%d,%d,%lld,%lld,%f\n",
               blockIdx.x,
               threadIdx.x,
               index,
               index+step*interval,
               input_data[index+step*interval]);*/
        sum_data[index]+=(double)input_data[index+step*batch_interval];
    }
    sum_data[index]/=coefficient;
}


//对gpu函数的封装,thread_num不超过channel_num
void channels_sum(void *input_data_void,void *sum_data_void,long long int window_size,double coefficient ,long long int batch_interval,long long int channel_num,long long int thread_num)
{
    dim3 channels_sum_blocksize(thread_num,1,1);
    dim3 channels_sum_gridsize(channel_num/thread_num,1,1);
    float *input_data=(float *)input_data_void;
    input_data++;
    double *sum_data=(double*)sum_data_void;
#ifdef PRINT_INFO
    printf("Function channels_average is called.\nwindow_size=%lld , coefficient=%f, batch_interval=%lld , channel_num=%lld , thread_num=%lld.\n",
           window_size,
           coefficient,
           batch_interval,
           channel_num,
           thread_num
           );
#endif
    channels_sum_gpu<<<channels_sum_gridsize,channels_sum_blocksize>>>(input_data,sum_data,window_size,coefficient,batch_interval);
    cudaDeviceSynchronize();
#ifdef PRINT_INFO
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
#endif
}


//这里的batch指按照步长滑动的次数

__global__ void compress_gpu(double *average_data,float *uncompressed_data_head,float *uncompressed_data_tail,unsigned char*compressed_data, long long int batch_interval ,long long int batch, long long int step, long long int output_channel_num ,long long int window_size) {
    long long int output_index= threadIdx.x + blockIdx.x * blockDim.x;
    long long int input_index=output_index*8;
    long long int average_index=input_index;
    //printf("%d,%d,%lld,%lld,o,i,O\n",threadIdx.x,blockIdx.x,output_index,input_index);
    double head_sum;
    double tail_sum;
    unsigned char bit_sum;
    long long int batch_num=0;
    for(batch_num=0;batch_num<batch;batch_num++)
    {
        //printf("b_n=%d\n",batch_num);
        bit_sum=0;
        //完成相邻8个通道一个step内数据的压缩
        for(int bit_step=0;bit_step<8;bit_step++)
        {
            head_sum=0;
            tail_sum=0;
            //完成单个通道的step个数据的读取
            for(int sum_step=0;sum_step<step;sum_step++)
            {
                head_sum+=uncompressed_data_head[input_index+bit_step];
                tail_sum+=uncompressed_data_tail[input_index+bit_step];
                //printf("%d,%d,%d,%lld,Iht\n",threadIdx.x,blockIdx.x,batch_num,input_index+bit_step);
                input_index+=batch_interval;
            }
            input_index-=batch_interval*step;
            average_data[average_index+bit_step]+=tail_sum/(float)window_size;
            bit_sum+=(((head_sum/step)>average_data[average_index+bit_step])?1:0)<<bit_step;
            //printf("%d,%d,%d,%lld,%f,%f,CP\n",threadIdx.x,blockIdx.x,batch_num,average_index+bit_step,(head_sum/step),average_data[average_index+bit_step]);
            average_data[average_index+bit_step]-=head_sum/(float)window_size;
            //printf("%d,%d,%d,%lld,%f,%f,%f,NA\n",threadIdx.x,blockIdx.x,batch_num,average_index+bit_step,(tail_sum/step),(head_sum/step),average_data[average_index+bit_step]);
            //printf("%d,%d,%d,O+\n",threadIdx.x,blockIdx.x,bit_sum);
        }
        compressed_data[output_index]=bit_sum;
        //printf("%d,%d,%lld,O\n",threadIdx.x,blockIdx.x,output_index);
        input_index+=batch_interval*step;
        output_index+=output_channel_num;
    }
}

//对gpu函数的封装, thread_num不超过channel_num/8, step指定时间分辨率
void compress(void *average_data_void,float *uncompressed_data_head_void, float *uncompressed_data_tail_void, void *compressed_data_void, long long int batch_interval ,long long int batch, long long int step, long long int begin_channel,long long int channel_num ,long long int window_size,long long int thread_num)
{
    double *average_data=(double*)average_data_void;
    float *uncompressed_data_head=(float*)uncompressed_data_head_void;
    uncompressed_data_head++;
    float *uncompressed_data_tail=(float*)uncompressed_data_tail_void;
    uncompressed_data_tail++;
    uncompressed_data_tail-=step*batch_interval;
    
    average_data+=(begin_channel-1);
    uncompressed_data_head+=(begin_channel-1);
    uncompressed_data_tail+=(begin_channel-1);
    
    unsigned char *compressed_data=(unsigned char*)compressed_data_void;
    long long int output_channel_num=channel_num/8;
    dim3 compress_blocksize(thread_num,1,1);
    dim3 compress_gridsize(output_channel_num/thread_num,1,1);
#ifdef PRINT_INFO
    printf("Function compress is called.\nstep=%lld, tail offset=%lld, batch=%lld, batch_interval=%lld, begin_channel=%lld, channel_num=%lld, thread_num=%lld.\n",
           step,
           window_size*batch_interval,
           batch,
           batch_interval,
           begin_channel,
           channel_num,
           thread_num
           );
#endif
    compress_gpu<<<compress_gridsize,compress_blocksize>>>(average_data,uncompressed_data_head,uncompressed_data_tail,compressed_data,batch_interval,batch,step,output_channel_num,window_size);
    cudaDeviceSynchronize();
#ifdef PRINT_INFO
    int error=cudaGetLastError();
    printf("Error code is %d\n",error);
#endif
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
    printf("thread %d , block %d\n",threadIdx.x,blockIdx.x);
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





