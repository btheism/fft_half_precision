#include <cuda_fp16.h>

//用于转换输入数据为float的函数,同时调整数据的布局(分离两个通道，在每个batch后补0便于进行原位fft变换)
//一个block的线程负责一个batch内数据的转置，grid数=block数
__global__ void char2float_interlace_gpu(signed char *input_char,half *input_float,long long int fft_length) {
    long long int input_grid_offset = fft_length * 2 * blockIdx.x;
    long long int output_grid_offset = (fft_length + 2) * 2 * blockIdx.x;
    long long int thread_loop=fft_length/1024;
    //input_block_offset = i * 1024 * 2 + threadIdx.x * 2;
    //output_block_offset = i * 1024 + threadIdx.x;
    long long int input_block_offset=threadIdx.x*1024*2;
    long long int output_block_offset_A=threadIdx.x*1024;
    long long int output_block_offset_B=threadIdx.x*1024+(fft_length+2) ;
    // __shared__ signed char sharedMemory[2048];
    //用1024个线程分128次(131072/1024)完成一个batch内的数据转置
    for (long long int i = 0; i < thread_loop; i++) {

        //sharedMemory[threadIdx.x]=(signed char)input_char[input_inside_batch_offset+input_batch_offset];
        //sharedMemory[threadIdx.x+1]=(signed char)input_char[input_inside_batch_offset+input_batch_offset+1];
        input_float[output_block_offset_A + output_grid_offset] =
                input_char[input_block_offset + input_grid_offset];
        input_float[output_block_offset_B + output_grid_offset] =
                input_char[input_block_offset + input_grid_offset + 1] ;
        input_block_offset+=2;
        output_block_offset_A++;
        output_block_offset_B++;
    }
}

//对gpu函数的封装
void char2float_interlace(signed char* input_char_gpu,half* fft_real,int batch,long long int fft_length)
{
    //把数据从input_char_gpu复制到fft_real,并分离两个通道的数据
    dim3 char2float_blockSize(1024,1,1);
    dim3 char2float_gridSize(batch,1,1);
    char2float_interlace_gpu<<<char2float_gridSize,char2float_blockSize>>>(input_char_gpu,fft_real,fft_length);
    cudaDeviceSynchronize();
}
