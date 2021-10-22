
#include<iostream>
#include<fstream>

#include <string>
#include <cuda_runtime.h>

#include <other_function_library.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>

#include <stdlib.h>
#include <time.h> 

int main(int argc, char *argv[]) {
//signal_batch用于生成虚假的signal_length
    long long int simulate_batch=50;
    long long int simulate_fft_length=16;
    //在实际的程序中,使用generate_file_list()函数得到该数字
    long long int signal_length=simulate_batch*simulate_fft_length*2;
    
//生成虚拟数据
    signed char *simulate_input_char;
    cudaMallocManaged((void**)&simulate_input_char,sizeof(signed char)*signal_length);
    signed char * simulate_array;
    //srand((unsigned)time(NULL));
    srand(932563);
    for(int i=0;i<signal_length;i++)
        simulate_input_char[i]=rand()%256;
    
//下列数据均为自定义参量    
    long long int fft_length=simulate_fft_length;
    long long int window_size=8;
    long long int step=1;
    long long int thread_num=1;
    
    float factor_A=1.0;
    float factor_B=1.0;
    
    long long int begin_channel=1;
    long long int compress_channel_num=fft_length/2;
    //规定缓冲区的大小,该大小由gpu的显存决定,太大会导致程序运行失败
    //注意,由于compress函数一次读取step个数据,为避免内存访问越界,batch_buffer_size必须被step整除,此外,为了确保对可以对数据末尾的反射,以及在拷贝数据时出错batch_buffer_size>=2*window_size
    long long int batch_buffer_size=20;
 

//下列参数由程序自行计算    
    
    //计算数据被划分的batch数,双通道数据需要额外除以2
    long long int signal_batch=(signal_length/2/fft_length/step)*step;
    //缓冲区减去窗口大小是每次压缩的数据量
    long long int per_batch=batch_buffer_size-window_size;
    
    //计算需要在循环中压缩数据的次数
    long long int compress_num=(signal_batch-window_size)/per_batch;
    
    //若signal_batch无法被compress_batch整除,则需要单独处理剩余的batch
    long long int remain_batch=signal_batch-compress_num*per_batch-window_size;

    //分配内存空间
    signed char *input_char;
    short *input_half;
    double *average_data;
    unsigned char* compressed_data;
    
    long long int input_char_size=fft_length*(((per_batch+remain_batch)>window_size)?per_batch+remain_batch:window_size)*2;
    long long int input_half_size=(fft_length+2)*batch_buffer_size*2;
    long long int average_data_size=fft_length/2;
    long long int compressed_data_size=compress_channel_num/8*per_batch;
    cudaMallocManaged((void**)&input_char,sizeof(signed char)*input_char_size);
    cudaMallocManaged((void**)&input_half,sizeof(short)*input_half_size);
    cudaMallocManaged((void**)&average_data,sizeof(double)*average_data_size);
    cudaMallocManaged((void**)&compressed_data,sizeof(unsigned char)*compressed_data_size);
    
    float *input_float=(float*)input_half;
    long long int input_half_offset=(fft_length+2)*2*window_size;
    long long int input_float_offset=(fft_length/2+1)*2*window_size;
    
    
    std::cout<<"fft length = "<<fft_length<<std::endl;
    std::cout<<"signal batch= "<<signal_batch<<std::endl;
    std::cout<<"window size = "<<window_size<<std::endl;
    std::cout<<"step = "<<step<<std::endl;
    std::cout<<"batch buffer size="<<batch_buffer_size<<std::endl;
    std::cout<<"per batch ="<<per_batch<<std::endl;
    std::cout<<"compress num="<<compress_num<<std::endl;
    std::cout<<"remain batch="<<remain_batch<<std::endl;
    std::cout<<"input char size = "<<input_char_size<<std::endl;
    std::cout<<"input half size = "<<input_half_size<<std::endl;
    std::cout<<"average data size = "<<average_data_size<<std::endl;
    std::cout<<"compressed data size = "<<compressed_data_size<<std::endl;
    
    
    read_char_array(input_char, simulate_input_char ,fft_length*window_size*2);
    char2float_interlace(input_char,input_half,fft_length,window_size,thread_num);
    print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
    
    fft_1d_half(input_half,input_half,fft_length,window_size*2);
    print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
    
    complex_modulus_squared_interlace(input_half,input_half,fft_length/2,factor_A,factor_B,window_size,thread_num);
    print_data_float(input_float,0,(fft_length/2+1)*batch_buffer_size*2,fft_length/2+1);

    channels_sum(input_float,average_data,window_size-step,window_size,(fft_length/2+1)*2,fft_length/2,thread_num);
    print_data_double(average_data,0,fft_length/2,fft_length/2);

    for(int loop_num=0;loop_num<compress_num;loop_num++)
    {
        printf("compress_loop=%d\n",loop_num);
        
        read_char_array(input_char, simulate_input_char ,fft_length*per_batch*2);        
        char2float_interlace(input_char,input_half+input_half_offset,fft_length,per_batch,thread_num);
        print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
        
        fft_1d_half(input_half+input_half_offset,input_half+input_half_offset,fft_length,per_batch*2);
        print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
        
        complex_modulus_squared_interlace(input_half+input_half_offset,input_float+input_float_offset,fft_length/2,factor_A,factor_B,per_batch,thread_num);
        print_data_float(input_float,0,(fft_length/2+1)*batch_buffer_size*2,fft_length/2+1);
        
        compress(average_data, input_float, input_float+input_float_offset,compressed_data,(fft_length/2+1)*2,per_batch/step, step,begin_channel, compress_channel_num, window_size, thread_num);
        print_data_binary(compressed_data,0,(compress_channel_num/8)*(per_batch/step),compress_channel_num/8);
        
        //注意,调用cudaMemcpy时要传入复制的内存大小,因此要乘sizeof
        cudaMemcpy(input_float,input_float+(fft_length/2+1)*2*per_batch,(fft_length/2+1)*2*window_size*sizeof(float),cudaMemcpyDeviceToDevice);
        print_data_float(input_float,0,(fft_length/2+1)*batch_buffer_size*2,fft_length/2+1);
    }
    
    
     if(remain_batch>0)
     {
         //把余下的数据补到最后一次读取的数据的后面
         printf("begin compress remained batch\n");
         read_char_array(input_char+fft_length*per_batch*2,simulate_input_char,fft_length*remain_batch*2);
         //先读取余下的数据
         char2float_interlace(input_char+fft_length*per_batch*2,input_half+input_half_offset,fft_length,remain_batch,thread_num);
         print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);

         
         //由于缓冲区空间不足,无法一次性完成反射
         if(per_batch-remain_batch<window_size-step)
         {
             
             //反射读取数据
             printf("complete reflect in two steps\n");
            
             
             //第一次反射,读取了per_batch-remain_batch,还剩(window_size-step)-(per_batch-remain_batch),反射到window_size+remain_batch的位置
             printf("first step\n");
             char2float_interlace_reflect(input_char+(per_batch+remain_batch-step)*fft_length*2-1,input_half+(fft_length+2)*2*(window_size+remain_batch),fft_length,per_batch-remain_batch,thread_num);
             print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
             
             
             //由于缓冲区现在读满了数据,与循环中的处理方式相同
             fft_1d_half(input_half+input_half_offset,input_half+input_half_offset,fft_length,per_batch*2);
             print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
             
             complex_modulus_squared_interlace(input_half+input_half_offset,input_float+input_float_offset,fft_length/2,factor_A,factor_B,per_batch,thread_num);
             print_data_float(input_float,0,(fft_length/2+1)*batch_buffer_size*2,fft_length/2+1);
             
             compress(average_data, input_float, input_float+input_float_offset,compressed_data,(fft_length/2+1)*2,per_batch/step, step, begin_channel, compress_channel_num, window_size, thread_num);
             print_data_binary(compressed_data,0,(compress_channel_num/8)*(per_batch/step),compress_channel_num/8);
             
             cudaMemcpy(input_float,input_float+(fft_length/2+1)*2*per_batch,(fft_length/2+1)*2*window_size*sizeof(float),cudaMemcpyDeviceToDevice);
             print_data_float(input_float,0,(fft_length/2+1)*batch_buffer_size*2,fft_length/2+1);  
             
             //第二次反射,还剩window_size-step-per_batch+remain_batch,char的偏移量为(per_batch+remain_batch-step)-(per_batch-remain_batch)=2*remain_batch-step
             printf("second step\n");
             char2float_interlace_reflect(input_char+(2*remain_batch-step)*fft_length*2-1,input_half+input_half_offset,fft_length,(window_size-step)-(per_batch-remain_batch),thread_num);
             print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
             
             fft_1d_half(input_half+input_half_offset,input_half+input_half_offset,fft_length,((window_size-step)-(per_batch-remain_batch))*2);
             print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
             
             complex_modulus_squared_interlace(input_half+input_half_offset,input_float+input_float_offset,fft_length/2,factor_A,factor_B,((window_size-step)-(per_batch-remain_batch)),thread_num);
             print_data_float(input_float,0,(fft_length/2+1)*batch_buffer_size*2,fft_length/2+1);
             
             compress(average_data, input_float, input_float+input_float_offset,compressed_data,(fft_length/2+1)*2,((window_size-step)-(per_batch-remain_batch))/step+1, step, begin_channel, compress_channel_num, window_size, thread_num);
             print_data_binary(compressed_data,0,(compress_channel_num/8)*((window_size-step)-(per_batch-remain_batch))/step+1,compress_channel_num/8);
         }
        //缓冲区空间够用,可以一次性完成反射
         else
         {
             printf("complete reflect in one step\n");
             
             //从char的数据末尾开始反射读取,per_batch+remain_batch-step为数据末尾回退一个step的位置
             //读取到float末尾位置,偏移量为window_size+remain_batch
             //读取量为window_size-step
             char2float_interlace_reflect(input_char+(per_batch+remain_batch-step)*fft_length*2-1,input_half+(fft_length+2)*2*(window_size+remain_batch),fft_length,window_size-step,thread_num);
             print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
             
             //计算量为remain_batch+window_size-step
             fft_1d_half(input_half+input_half_offset,input_half+input_half_offset,fft_length,(remain_batch+window_size-step)*2);
             print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
             //计算量为remain_batch+window_size-step
             complex_modulus_squared_interlace(input_half+input_half_offset,input_float+input_float_offset,fft_length/2,factor_A,factor_B,(remain_batch+window_size-step),thread_num);
             print_data_float(input_float,0,(fft_length/2+1)*batch_buffer_size*2,fft_length/2+1);
             
             //压缩量为(remain_batch+window_size-step)/step+1
             compress(average_data, input_float, input_float+(fft_length/2+1)*2*window_size,compressed_data,(fft_length/2+1)*2,(window_size-step+remain_batch)/step+1, step, begin_channel, compress_channel_num, window_size, thread_num);
             print_data_binary(compressed_data,0,(compress_channel_num/8)*((window_size-step+remain_batch)/step+1),compress_channel_num/8);
         }
     }
     

    cudaFree(input_char);
    cudaFree(input_half);
    cudaFree(average_data);
    cudaFree(compressed_data);
    
    cudaFree(simulate_input_char);
    return 0;
}
 
