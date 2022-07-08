#ifdef DEBUG
constexpr bool debug_flag=1;
#else
constexpr bool debug_flag=0;
#endif

#include <cuda_fp16.h>
#include<stdio.h>
#include<iostream>
#include<exception>
#include<cuda_macros.hpp>

#define INT8_SCALE 256 //定义对输入数据的压缩系数
#define INT16_SCALE 65536
//#define PRINT_INFO //是否打印函数调用信息
#define TEST_INF //是否检测fft的变换结果出现溢出

//内核编写的大致原则：
//涉及到多个fft区间的，每个区间用一个block处理，该block内可包含很多线程，故block数对应区间数。
//涉及到单个fft区间的，则进一步分割这个区间。
//每个内核函数都由一个同名的(去掉内核函数后缀的gpu)host函数包装。

//用于转换双通道输入数据为float的函数,同时调整数据的布局(分离两个通道，在每个读入的预定进行fft变换的数组补充空隙便于进行原位fft变换)，fft区间数=batch*2
//一个block的线程负责一个batch内数据的转置,生成两组需要fft变换的数,grid数=batch数
__global__ void char2float_interlace_gpu(signed char*input_char,half *input_half,long long int fft_length,long long int thread_loop,int thread_num) 
{
    
    //计算不同block的内存偏移量
    long long int input_grid_offset = fft_length * 2 * blockIdx.x;
    long long int output_grid_offset = (fft_length + 2) * 2 * blockIdx.x;
    //计算block内部的内存偏移量(该偏移量已与不同block的内存偏移量相加)
    long long int input_block_offset=threadIdx.x*2+input_grid_offset;
    long long int output_block_offset_A=threadIdx.x+output_grid_offset;
    long long int output_block_offset_B=output_block_offset_A+(fft_length+2);
    // __shared__ signed char sharedMemory[2048];
    for (int i = 0; i < thread_loop; i++) {
    //半精度浮点数的相对精度约为1/1000,把正负128以内的数据归一化到正负0.5，既有足够的精度表示数据，又保证了fft变换不会出现溢出(131072*0.5=65536,正好是半精度浮点数表示范围的上界)
        input_half[output_block_offset_A] =(half)((float)input_char[input_block_offset]/(float)INT8_SCALE);
        input_half[output_block_offset_B] =(half)((float)input_char[input_block_offset+1]/(float)INT8_SCALE);
        input_block_offset+=2*thread_num;
        output_block_offset_A+=thread_num;
        output_block_offset_B+=thread_num;
    }
}

//对gpu函数的封装
//thread_num最大等于fft_length
void char2float_interlace(void* input_char_void,void* input_half_void,long long int fft_length,long long int batch,int thread_num)
{
    dim3 char2float_interlace_blocksize(thread_num,1,1);
    dim3 char2float_interlace_gridsize(batch,1,1);
    signed char*input_char=(signed char*)input_char_void;
    half *input_half=(half*)input_half_void;
    long long int thread_loop=fft_length/thread_num;
#ifdef PRINT_INFO
    printf("Function char2float_interlace is called\n,batch=%lld , fft_length=%lld , thread_num=%d , thread_loop=%lld\n",
           batch,
           fft_length,
           thread_num,
           thread_loop
           );
#endif
    char2float_interlace_gpu<<<char2float_interlace_gridsize,char2float_interlace_blocksize>>>(input_char,input_half,fft_length,thread_loop,thread_num);
    gpuErrchk(cudaDeviceSynchronize());
}


//用于转换单通道输入数据为float的函数，在每个读入的预定进行fft变换的数组补充空隙便于进行原位fft变换
//一个block的线程负责一个batch内数据的转置，grid数=batch数
__global__ void char2float_gpu(signed char*input_char,half *input_half,long long int fft_length,long long int thread_loop,int thread_num) 
{
    
    //计算不同block的内存偏移量
    long long int input_grid_offset = fft_length * blockIdx.x;
    long long int output_grid_offset = (fft_length + 2) * blockIdx.x;
    
    //计算block内部的内存偏移量(该偏移量已与不同block的内存偏移量相加)
    long long int input_block_offset=threadIdx.x+input_grid_offset;
    long long int output_block_offset=threadIdx.x+output_grid_offset;

    for (int i = 0; i < thread_loop; i++) {
        input_half[output_block_offset] =(half)((float)input_char[input_block_offset]/(float)INT8_SCALE);
        input_block_offset+=thread_num;
        output_block_offset+=thread_num;
    }
}

__global__ void short2float_gpu(short*input_short,half *input_half,long long int fft_length,long long int thread_loop,int thread_num) 
{
    
    //计算不同block的内存偏移量
    long long int input_grid_offset = fft_length * blockIdx.x;
    long long int output_grid_offset = (fft_length + 2) * blockIdx.x;
    
    //计算block内部的内存偏移量(该偏移量已与不同block的内存偏移量相加)
    long long int input_block_offset=threadIdx.x+input_grid_offset;
    long long int output_block_offset=threadIdx.x+output_grid_offset;

    for (int i = 0; i < thread_loop; i++) {
        input_half[output_block_offset] =(half)((float)input_short[input_block_offset]/(float)INT16_SCALE);
        input_block_offset+=thread_num;
        output_block_offset+=thread_num;
    }
}

//对gpu函数的封装
//thread_num最大等于fft_length
//flag表示使用的是8位整数还是16位整数(8位0,16位1)
void int2float(void* input_int_void,void* input_half_void,long long int fft_length,long long int batch,int thread_num,int input_type_flag)
{
    dim3 int2float_blocksize(thread_num,1,1);
    dim3 int2float_gridsize(batch,1,1);
    half *input_half=(half*)input_half_void;
    long long int thread_loop=fft_length/thread_num;
#ifdef PRINT_INFO
    printf("Function int2float is called\n,batch=%lld , fft_length=%lld , thread_num=%d , thread_loop=%lld,input_type_flag=%d\n",
           batch,
           fft_length,
           thread_num,
           thread_loop,
           input_type_flag
           );
#endif
    switch(input_type_flag)
    {
        case 0:
            char2float_gpu<<<int2float_gridsize,int2float_blocksize>>>((signed char*)input_int_void,input_half,fft_length,thread_loop,thread_num);
            break;
        case 1:
            short2float_gpu<<<int2float_gridsize,int2float_blocksize>>>((short*)input_int_void,input_half,fft_length,thread_loop,thread_num);
            break;
        default:
            throw std::runtime_error("input_type_flag is not legal");
    }
    gpuErrchk(cudaDeviceSynchronize());
}

//上个函数辅助函数
size_t input_type_size(int input_type_flag)
{
    switch(input_type_flag)
    {
        case 0:
            //8位整型
            return 1;
        case 1:
            //16位整型
            return 2;
        default:
            throw std::runtime_error("invalid input type flag");
    }
}

//保留与之前的代码的兼容性
void char2float(void* input_int_void,void* input_half_void,long long int fft_length,long long int batch,int thread_num)
{
    int2float(input_int_void,input_half_void,fft_length,batch,thread_num,0);
};

//以下两个函数与之前的两个函数类似,只是按照倒序读取输入数据.用于对末尾的输入数据进行反射,差别仅在于input_block_offset多一个负号,此外;
//在双通道数据的情况下,两个通道的先后次序也需要交换,另外,input_block_offset需要不断减小
//输入的地址为需要反射的数据末尾-1;

//用于转换双通道输入数据为float的函数,同时调整数据的布局(分离两个通道，在每个读入的预定进行fft变换的数组补充空隙便于进行原位fft变换)，fft区间数=batch*2
//一个block的线程负责一个batch内数据的转置,生成两组需要fft变换的数,grid数=batch数
__global__ void char2float_interlace_reflect_gpu(signed char*input_char,half *input_half,long long int fft_length,long long int thread_loop,int thread_num) 
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
        input_half[output_block_offset_B] =(half)((float)input_char[input_block_offset]/(float)INT8_SCALE);

        input_half[output_block_offset_A] =(half)((float)input_char[input_block_offset-1]/(float)INT8_SCALE);

        input_block_offset-=2*thread_num;
        output_block_offset_A+=thread_num;
        output_block_offset_B+=thread_num;
    }
}

//对gpu函数的封装
//thread_num最大等于fft_length
void char2float_interlace_reflect(void* input_char_void,void* input_half_void,long long int fft_length,long long int batch,int thread_num)
{
    dim3 char2float_interlace_reflect_blocksize(thread_num,1,1);
    dim3 char2float_interlace_reflect_gridsize(batch,1,1);
    signed char*input_char=(signed char*)input_char_void;
    half *input_half=(half*)input_half_void;
    long long int thread_loop=fft_length/thread_num;
#ifdef PRINT_INFO
    printf("Function char2float_interlace_reflect is called\n,batch=%lld , fft_length=%lld , thread_num=%d , thread_loop=%lld\n",
           batch,
           fft_length,
           thread_num,
           thread_loop
           );
#endif
    char2float_interlace_reflect_gpu<<<char2float_interlace_reflect_gridsize,char2float_interlace_reflect_blocksize>>>(input_char,input_half,fft_length,thread_loop,thread_num);
    gpuErrchk(cudaDeviceSynchronize());
}


//用于转换单通道输入数据为float的函数，在每个读入的预定进行fft变换的数组补充空隙便于进行原位fft变换
//一个block的线程负责一个batch内数据的转置，grid数=batch数
__global__ void char2float_reflect_gpu(signed char*input_char,half *input_half,long long int fft_length,long long int thread_loop,int thread_num) 
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
        input_half[output_block_offset] =(half)((float)input_char[input_block_offset]/(float)INT8_SCALE);
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
void char2float_reflect(void* input_char_void,void* input_half_void,long long int fft_length,long long int batch,int thread_num)
{
    dim3 char2float_reflect_blocksize(thread_num,1,1);
    dim3 char2float_reflect_gridsize(batch,1,1);
    signed char*input_char=(signed char*)input_char_void;
    half *input_half=(half*)input_half_void;
    long long int thread_loop=fft_length/thread_num;
#ifdef PRINT_INFO
    printf("Function char2float_reflect is called\n,batch=%lld , fft_length=%lld , thread_num=%d , thread_loop=%lld\n",
           batch,
           fft_length,
           thread_num,
           thread_loop
           );
#endif
    char2float_reflect_gpu<<<char2float_reflect_gridsize,char2float_reflect_blocksize>>>(input_char,input_half,fft_length,thread_loop,thread_num);
    gpuErrchk(cudaDeviceSynchronize());
}


//计算FFT变换后的结果的模方的函数,计算前为32bit复数(2*16bit)，计算后为32bits实数。
//该函数用于单通道数据
//一个block的线程负责一个fft区间内数据的计算，grid数=batch数

__global__ void complex_modulus_squared_gpu(half2 *complex_number,float *float_number,long long int channel_num ,long long int thread_loop,int thread_num) {
    //计算偏移量,由于不同的fft变换结果之间有间隔(通道0)，因此此处需要乘channel_num+1
    long long int index= (threadIdx.x + blockIdx.x*(channel_num+1));
    for(int i=0;i<thread_loop;i++)
    {
#ifdef TEST_INF
        if(__hisinf(complex_number[index].x))
            printf("%lld,%f,x",index,(float)complex_number[index].x);
        if(__hisinf(complex_number[index].y))
            printf("%lld,%f,x",index,(float)complex_number[index].y);
#endif
    //计算复数的模方
        float_number[index]=(float)((complex_number[index].x*complex_number[index].x)+(complex_number[index].y*complex_number[index].y));
        index+=thread_num;
    }
}

//对gpu函数的封装
//thread_num最大等于channel_num,fft_num代表需要计算的fft区间的数量,单通道的情况下等于batch,双通道的情况下等于batch*2
void complex_modulus_squared(void *complex_number_void,void *float_number_void, long long int channel_num,long long int batch,int thread_num)
{
    dim3 complex_modulus_squared_blocksize(thread_num,1,1);
    dim3 complex_modulus_squared_gridsize(batch,1,1);
    half2 *complex_number=(half2*)complex_number_void;
    float *float_number=(float*)float_number_void;
    //跳过通道0,需要++;
    complex_number++;
    float_number++;
    long long int thread_loop=channel_num/thread_num;
#ifdef PRINT_INFO
    printf("Function complex_modulus_squared is called.\nbatch=%lld , channel_num=%lld , thread_num=%d , thread_loop=%lld.\n",
           batch,
           channel_num,
           thread_num,
           thread_loop);
#endif
    complex_modulus_squared_gpu<<<complex_modulus_squared_gridsize,complex_modulus_squared_blocksize>>>
    (complex_number,float_number,channel_num,thread_loop,thread_num);
    gpuErrchk(cudaDeviceSynchronize());
}


//计算频谱乘法的函数,用于给频谱乘一个系数
//一个block的线程负责一个fft区间内数据的计算，grid数=batch数

__global__ void complex_multiply_gpu(half2 *complex_number , half2 multiplier , long long int channel_num , long long int thread_loop , int thread_num) {
    //计算偏移量,由于不同的fft变换结果之间有间隔(通道0)，因此此处需要乘channel_num+1
    long long int index= (threadIdx.x + blockIdx.x*(channel_num+1));
    half2 tmp_complex_number;
    for(int i=0;i<thread_loop;i++)
    {
#ifdef TEST_INF
        if(__hisinf(complex_number[index].x))
            printf("%lld,%f,x",index,(float)complex_number[index].x);
        if(__hisinf(complex_number[index].y))
            printf("%lld,%f,x",index,(float)complex_number[index].y);
#endif
        tmp_complex_number = complex_number[index];
        complex_number[index].x=
        (tmp_complex_number.x * multiplier.x) - (tmp_complex_number.y * multiplier.y);
        complex_number[index].y=
        (tmp_complex_number.x * multiplier.y) + (tmp_complex_number.y * multiplier.x);
        index+=thread_num;
    }
};

//计算频谱相移的函数,用于给频谱施加相移,phase指中心频率的相移动
//grid包含的总线程数等于fft的通道数,一个fft线程负责相移一个通道的数据

__global__ void phase_shift_gpu(half2 *complex_number , float phase , long long int channel_num , long long int batch) {
    //计算该线程对应的通道
    long long int index = (threadIdx.x + blockIdx.x*blockDim.x);
    //计算该通道的相移
    float local_phase = 2.0*(float)index/(float)channel_num;
    half2 tmp_complex_number;
    half2 multiplier;
    multiplier.x=cos(local_phase);
    multiplier.y=sin(local_phase);
    for(int i=0;i<batch;i++)
    {
        tmp_complex_number = complex_number[index];
        complex_number[index].x=
        (tmp_complex_number.x * multiplier.x) - (tmp_complex_number.y * multiplier.y);
        complex_number[index].y=
        (tmp_complex_number.x * multiplier.y) + (tmp_complex_number.y * multiplier.x);
        index+=(channel_num+1);
    }
};

//对gpu函数的封装
//thread_num最大等于channel_num,fft_num代表需要计算的fft区间的数量,单通道的情况下等于batch,双通道的情况下等于batch*2
void phase_shift(void *complex_number_void , float phase , float normalization_factor , float weight_factor , long long int channel_num , long long int batch , int thread_num)
{
    dim3 complex_multiply_blocksize(thread_num,1,1);
    dim3 complex_multiply_gridsize(batch,1,1);
    half2 *complex_number=(half2*)complex_number_void;
    //跳过通道0,需要++;
    complex_number++;
    long long int thread_loop=channel_num/thread_num;
    half2 multiplier;
    multiplier.x=normalization_factor*weight_factor;
    multiplier.y=normalization_factor*weight_factor;
#ifdef PRINT_INFO
    printf("Function phase_shift is called.\ncenter phase shift=%f , normalization_factor=%f , weight_factor=%f , batch=%lld , channel_num=%lld , thread_num=%d , thread_loop=%lld.\n",
           phase,
           normalization_factor,
           weight_factor,
           batch,
           channel_num,
           thread_num,
           thread_loop);
#endif
    complex_multiply_gpu<<<complex_multiply_gridsize,complex_multiply_blocksize>>>
    (complex_number , multiplier , channel_num , thread_loop ,thread_num);
    
    dim3 phase_shift_blocksize(thread_num,1,1);
    dim3 phase_shift_gridsize(channel_num/thread_num,1,1);
    
    phase_shift_gpu<<<phase_shift_gridsize,phase_shift_blocksize>>>(complex_number ,phase ,channel_num ,batch);
    
    
    gpuErrchk(cudaDeviceSynchronize());
}

//计算复数加法的函数,用于把一路信号加到总的信号上,即把complex_number_in加到complex_number_out上
//一个block的线程负责一个fft区间内数据的计算，grid数=batch数

__global__ void complex_add_gpu(half2 *complex_number_out , half2 *complex_number_in , long long int channel_num , long long int thread_loop , int thread_num) {
    //计算偏移量,由于不同的fft变换结果之间有间隔(通道0)，因此此处需要乘channel_num+1
    long long int index= (threadIdx.x + blockIdx.x*(channel_num+1));
    for(int i=0;i<thread_loop;i++)
    {
        //计算复数的模方
        /*
        if(__hisinf(complex_number[block_offset_A].y) ||__hisnan(complex_number[block_offset_A].y))
            printf("E,%d,A.y",block_offset_A);
        */
        complex_number_out[index].x+=complex_number_in[index].x;
        complex_number_out[index].y+=complex_number_in[index].y;
#ifdef TEST_INF
        if(__hisinf(complex_number_out[index].x))
            printf("%lld,%f,x",index,(float)complex_number_out[index].x);
        if(__hisinf(complex_number_out[index].y))
            printf("%lld,%f,x",index,(float)complex_number_out[index].y);
#endif
        index+=thread_num;
    }
};

//对gpu函数的封装
//thread_num最大等于channel_num,fft_num代表需要计算的fft区间的数量,单通道的情况下等于batch,双通道的情况下等于batch*2
void complex_add(void *complex_number_out_void , void *complex_number_in_void , long long int channel_num , long long int batch , int thread_num)
{
    dim3 complex_add_blocksize(thread_num,1,1);
    dim3 complex_add_gridsize(batch,1,1);
    half2 *complex_number_out=(half2*)complex_number_out_void;
    half2 *complex_number_in=(half2*)complex_number_in_void;
    
    //此处不应该跳过通道0,而应该跳过最后一个通道;
    //complex_number_out++;
    //complex_number_in++;
    long long int thread_loop=channel_num/thread_num;
#ifdef PRINT_INFO
    printf("Function complex_add is called.\nbatch=%lld , channel_num=%lld , thread_num=%d , thread_loop=%lld.\n",
           batch,
           channel_num,
           thread_num,
           thread_loop);
#endif
    complex_add_gpu<<<complex_add_gridsize,complex_add_blocksize>>>
    (complex_number_out , complex_number_in , channel_num , thread_loop ,thread_num);
    gpuErrchk(cudaDeviceSynchronize());
}

//把一些位置的数据设置为半精度的0，用于清除fft变换结果的频域的第0通道，防止进行ifft时使得输出数据过大超过半精度浮点数的表示范围，interval表示需要置0的数据的位置间隔，length表示每个位置需要置0的连续数据的个数, num表示这些位置的数量

__global__ void set_to_half_zero_gpu(half *location , long long int interval , long long int length ,int thread_loop)
{
    half * begin_addr = location+(blockIdx.x*blockDim.x+threadIdx.x)*interval;
    long long int step = blockDim.x*gridDim.x*interval;
    //地址的序号
    for(long long int serial_out=0 ; serial_out<thread_loop ; serial_out+=1)
    {
        //该地址的连续数据的序号
        for(long long int serial_in=0 ; serial_in<length ; serial_in+=1)
        {
            begin_addr[serial_in]=(half)0.0;
        }
        begin_addr+=step;
    }
}

void set_to_half_zero(void *location_void , long long int interval , long long int length , long long int num , int thread_num , long long int grid_size)
{
    if(debug_flag)
    {
        std::cout<<"function set_to_half_zero is called grid_size = "<<grid_size<<" thread number =  "<<thread_num<<std::endl;
    }
    half* location = (half*)location_void;
    long long int main_num = num/(thread_num*grid_size);
    long long int remain_num1 = num%(thread_num*grid_size);
    long long int remain_grid_size = remain_num1/thread_num;
    long long int remain_num2;
    /*if(remain_num!=0)
    {
        throw std::invalid_argument;
    }*/

    if(main_num>0)
    {
        if(debug_flag){std::cout<<"main_num = "<<main_num<<std::endl;}
        set_to_half_zero_gpu<<<dim3{(unsigned int)grid_size,1,1},dim3{(unsigned int)thread_num,1,1}>>>(location , interval , length , main_num);
        location+=main_num*thread_num*grid_size*interval;
    }
    if(remain_num1>0)
    {
        if(debug_flag){std::cout<<"remain_num1 = "<<remain_num1<<std::endl;}
        if(remain_grid_size>0)
        {
            set_to_half_zero_gpu<<<dim3{(unsigned int)remain_grid_size,1,1},dim3{(unsigned int)thread_num,1,1}>>>(location, interval , length , 1);
            location+=thread_num*remain_grid_size*interval;
            remain_num2 = remain_num1%(thread_num*remain_grid_size);
        }
        else
        {
            remain_num2 = remain_num1;
        }
        if(remain_num2>0)
        {
            if(debug_flag){std::cout<<"remain_num2 = "<<remain_num2<<std::endl;}
            set_to_half_zero_gpu<<<dim3{(unsigned int)1,1,1},dim3{(unsigned int)remain_num2,1,1}>>>(location, interval , length , 1);
        }
    }
    gpuErrchk(cudaDeviceSynchronize());
}

//计算FFT变换后的结果的模方的函数,计算前为32bit复数(2*16bit)，计算后为32bits实数
//该函数用于双通道数据，需要指定两个通道的权重
//一个block的线程负责一个fft区间内数据的计算，grid数=batch数

__global__ void complex_modulus_squared_interlace_gpu(half2 *complex_number,float *float_number,long long int channel_num, float factor_A, float factor_B, long long int thread_loop,int thread_num) {
    //计算单个通道的的偏移量,由于不同的fft变换结果之间有间隔(通道0)，因此此处需要乘channel_num+1
    long long int index_A= (threadIdx.x + blockIdx.x * (channel_num+1)*2);
    long long int index_B= index_A+channel_num+1;
    for(int i=0;i<thread_loop;i++)
    {
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
};


//对gpu函数的封装
//thread_num最大等于channel_num,fft_num代表需要计算的fft区间的数量,单通道的情况下等于batch,双通道的情况下等于batch*2
void complex_modulus_squared_interlace(void *complex_number_void,void *float_number_void, long long int channel_num, float factor_A, float factor_B, long long int batch,int thread_num)
{
    dim3 complex_modulus_squared_interlace_blocksize(thread_num,1,1);
    dim3 complex_modulus_squared_interlace_gridsize(batch,1,1);
    half2 *complex_number=(half2*)complex_number_void;
    float *float_number=(float*)float_number_void;
    //printf("complex_number=%p,float_number=%p\n",complex_number,float_number);
    //printf("size of half2=%lld\n",sizeof(half2));
    //跳过通道0,需要++;
    complex_number++;
    float_number++;
    //printf("complex_number=%p,float_number=%p\n",complex_number,float_number);
    long long int thread_loop=channel_num/thread_num;
#ifdef PRINT_INFO
    printf("Function complex_modulus_squared_interlace is called.\nbatch=%lld , channel_num=%lld , thread_num=%d , thread_loop=%lld , factor_A=%f , factor_B=%f .\n",
           batch,
           channel_num,
           thread_num,
           thread_loop,
           factor_A,
           factor_B);
#endif
    complex_modulus_squared_interlace_gpu<<<complex_modulus_squared_interlace_gridsize,complex_modulus_squared_interlace_blocksize>>>
    (complex_number,float_number,channel_num,factor_A,factor_B,thread_loop,thread_num);
    gpuErrchk(cudaDeviceSynchronize());
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
void channels_sum(void *input_data_void,void *sum_data_void,long long int window_size,double coefficient ,long long int batch_interval,long long int channel_num,int thread_num)
{
    dim3 channels_sum_blocksize(thread_num,1,1);
    dim3 channels_sum_gridsize(channel_num/thread_num,1,1);
    float *input_data=(float *)input_data_void;
    input_data++;
    double *sum_data=(double*)sum_data_void;
#ifdef PRINT_INFO
    printf("Function channels_average is called.\nwindow_size=%lld , coefficient=%f, batch_interval=%lld , channel_num=%lld , thread_num=%d.\n",
           window_size,
           coefficient,
           batch_interval,
           channel_num,
           thread_num
           );
#endif
    channels_sum_gpu<<<channels_sum_gridsize,channels_sum_blocksize>>>(input_data,sum_data,window_size,coefficient,batch_interval);
    gpuErrchk(cudaDeviceSynchronize());
}


//这里的batch指按照步长滑动的次数

__global__ void compress_gpu(double *average_data,float *uncompressed_data_head,float *uncompressed_data_tail,unsigned char*compressed_data, long long int batch_interval, long long int batch, long long int step, long long int output_channel_num ,float window_size) {
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
                input_index+=batch_interval;
            }
            //回退input_index
            input_index-=batch_interval*step;
            bit_sum+=(((head_sum/step)>average_data[average_index+bit_step])?1:0)<<bit_step;
            average_data[average_index+bit_step]-=head_sum/window_size;
            average_data[average_index+bit_step]+=tail_sum/window_size;
        }
        compressed_data[output_index]=bit_sum;
        input_index+=batch_interval*step;
        output_index+=output_channel_num;
    }
}

//对gpu函数的封装, thread_num不超过channel_num/8, step指定时间分辨率
void compress(void *average_data_void,float *uncompressed_data_head_void, float *uncompressed_data_tail_void, void *compressed_data_void, long long int batch_interval ,long long int batch, long long int step, long long int begin_channel,long long int channel_num ,long long int window_size,int thread_num)
{
    double *average_data=(double*)average_data_void;
    float *uncompressed_data_head=(float*)uncompressed_data_head_void;
    //++跳过通道0
    uncompressed_data_head++;
    float *uncompressed_data_tail=(float*)uncompressed_data_tail_void;
    uncompressed_data_tail++;
    
    //调整指针到begin_channel
    average_data+=(begin_channel-1);
    uncompressed_data_head+=(begin_channel-1);
    uncompressed_data_tail+=(begin_channel-1);
    
    unsigned char *compressed_data=(unsigned char*)compressed_data_void;
    long long int output_channel_num=channel_num/8;
    dim3 compress_blocksize(thread_num,1,1);
    dim3 compress_gridsize(output_channel_num/thread_num,1,1);
#ifdef PRINT_INFO
    printf("Function compress is called.\nstep=%lld, tail offset=%lld, batch=%lld, batch_interval=%lld, begin_channel=%lld, channel_num=%lld, thread_num=%d.\n",
           step,
           window_size*batch_interval,
           batch,
           batch_interval,
           begin_channel,
           channel_num,
           thread_num
           );
#endif
    compress_gpu<<<compress_gridsize,compress_blocksize>>>(average_data,uncompressed_data_head,uncompressed_data_tail,compressed_data,batch_interval,batch,step,output_channel_num,(float)window_size);
    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void compress_reflect_gpu(double *average_data,float *uncompressed_data_head,float *uncompressed_data_tail,unsigned char*compressed_data, long long int batch_interval, long long int batch, long long int step, long long int output_channel_num ,long long int window_size) {
    long long int output_index= threadIdx.x + blockIdx.x * blockDim.x;
    long long int input_index_head=output_index*8;
    long long int input_index_tail=input_index_head;
    long long int average_index=input_index_head;
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
                head_sum+=uncompressed_data_head[input_index_head+bit_step];
                tail_sum+=uncompressed_data_tail[input_index_tail+bit_step];
                //printf("%d,%d,%d,%lld,Iht\n",threadIdx.x,blockIdx.x,batch_num,input_index+bit_step);
                input_index_head+=batch_interval;
                input_index_tail-=batch_interval;
            }
            //回退input_index,以计算相邻8个通道的下一个通道
            input_index_head-=batch_interval*step;
            input_index_tail+=batch_interval*step;
            bit_sum+=(((head_sum/step)>average_data[average_index+bit_step])?1:0)<<bit_step;
            average_data[average_index+bit_step]-=head_sum/(float)window_size;
            average_data[average_index+bit_step]+=tail_sum/(float)window_size;
            //printf("%d,%d,%d,%lld,%f,%f,CP\n",threadIdx.x,blockIdx.x,batch_num,average_index+bit_step,(head_sum/step),average_data[average_index+bit_step]);
            //printf("%d,%d,%d,%lld,%f,%f,%f,NA\n",threadIdx.x,blockIdx.x,batch_num,average_index+bit_step,(tail_sum/step),(head_sum/step),average_data[average_index+bit_step]);
            //printf("%d,%d,%d,O+\n",threadIdx.x,blockIdx.x,bit_sum);
        }
        compressed_data[output_index]=bit_sum;
        //printf("%d,%d,%lld,O\n",threadIdx.x,blockIdx.x,output_index);
        input_index_head+=batch_interval*step;
        input_index_tail-=batch_interval*step;
        output_index+=output_channel_num;
    }
}

//对gpu函数的封装, thread_num不超过channel_num/8, step指定时间分辨率
void compress_reflect(void *average_data_void,float *uncompressed_data_head_void, float *uncompressed_data_tail_void, void *compressed_data_void, long long int batch_interval ,long long int batch, long long int step, long long int begin_channel,long long int channel_num ,long long int window_size,int thread_num)
{
    double *average_data=(double*)average_data_void;
    float *uncompressed_data_head=(float*)uncompressed_data_head_void;
    //++跳过通道0
    uncompressed_data_head++;
    float *uncompressed_data_tail=(float*)uncompressed_data_tail_void;
    uncompressed_data_tail++;
    
    //调整指针到begin_channel
    average_data+=(begin_channel-1);
    uncompressed_data_head+=(begin_channel-1);
    uncompressed_data_tail+=(begin_channel-1);
    
    unsigned char *compressed_data=(unsigned char*)compressed_data_void;
    long long int output_channel_num=channel_num/8;
    dim3 compress_reflect_blocksize(thread_num,1,1);
    dim3 compress_reflect_gridsize(output_channel_num/thread_num,1,1);
#ifdef PRINT_INFO
    printf("Function compress_reflect is called.\nstep=%lld, tail offset=%lld, batch=%lld, batch_interval=%lld, begin_channel=%lld, channel_num=%lld, thread_num=%d.\n",
           step,
           window_size*batch_interval,
           batch,
           batch_interval,
           begin_channel,
           channel_num,
           thread_num
           );
#endif
    compress_reflect_gpu<<<compress_reflect_gridsize,compress_reflect_blocksize>>>(average_data,uncompressed_data_head,uncompressed_data_tail,compressed_data,batch_interval,batch,step,output_channel_num,window_size);
    gpuErrchk(cudaDeviceSynchronize());
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
    gpuErrchk(cudaDeviceSynchronize());
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
     gpuErrchk(cudaDeviceSynchronize());
     //kernal_parameter_test_gpu<<<1,1>>>(a,b,c,d,e,f);
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
     gpuErrchk(cudaDeviceSynchronize());
}
