//该程序由test_2_channel_with_reflect_and_loop修改而来
//#define PRINT_MEMORY //是否打印变换结果
//#define WRITE_HEADER //是否写入文件头
//#define WRITE_DATA //是否写入文件

#include <string>
#include <fstream>
#include <cuda_runtime.h>

#include <other_function_library.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>
#include<filheader.h>

int main(int argc, char *argv[]) {

    //生成文件列表
    std::string file_list[argc-1];
    long long int file_size_list[argc-1];
    //从generate_file_list函数得到输入数据的总长度
    long long int signal_length=generate_file_list(argc, argv, file_list, file_size_list);
    
    //下列数据均为自定义参量    
    long long int fft_length=131072;
    long long int window_size=4096;
    long long int step=4;
    
    float factor_A=1.0;
    float factor_B=2.0;
    
    //long long int begin_channel=fft_length/4;
    //compress_channel_num必须可以被8整除
    //long long int compress_channel_num=fft_length/4;
    
    long long int begin_channel=fft_length/4;
    //compress_channel_num必须可以被8整除
    long long int compress_channel_num=fft_length/4;
    
    //动态决定thread_num
    long long int thread_num=(compress_channel_num/8>1024)?1024:(compress_channel_num/8);
    
    //规定缓冲区的大小,该大小由gpu的显存决定,太大会导致程序运行失败
    //注意,由于compress函数一次读取step个数据,为避免内存访问越界,batch_buffer_size必须被step整除,此外,为了确保对可以对数据末尾的反射,以及在拷贝数据时出错batch_buffer_size>=2*window_size
    
    //不知为何，当cufft传入的batch为2的整数次幂时，所用的缓冲空间约为非整数次幂的一半，故最好保证batch_buffer_size-window_size是2的整数次幂
    long long int batch_buffer_size=8192;
    
    double tsamp=1.0/(2500000000/2);
    //下列参数由程序自行计算    
    
    //计算数据被划分的batch数,双通道数据需要额外除以2 ，且要去除无法被step整除的部分
    long long int signal_batch=(signal_length/2/fft_length/step)*step;
    //缓冲区减去窗口大小是每次压缩的数据量
    long long int per_batch=batch_buffer_size-window_size;
    
    //计算需要在循环中压缩数据的次数
    long long int compress_num=(signal_batch-window_size)/per_batch;
    
    //若signal_batch无法被compress_batch整除,则需要单独处理剩余的batch
    long long int remain_batch=signal_batch-compress_num*per_batch-window_size;

    //分配内存空间
    signed char *input_char_cpu;
    signed char *input_char;
    short *input_half;
    double *average_data;
    unsigned char* compressed_data;
    
    long long int input_char_size=fft_length*(((per_batch+remain_batch)>window_size)?(per_batch+remain_batch):window_size)*(long long int)2;
    long long int input_half_size=(fft_length+2)*batch_buffer_size*(long long int)2;
    long long int average_data_size=fft_length/(long long int)2;
    long long int compressed_data_size=compress_channel_num/8*(long long int)per_batch;
    cudaMallocHost((void**)&input_char_cpu,sizeof(signed char)*input_char_size);
    
    #ifdef PRINT_MEMORY
    //所有内存均为unified memory
    cudaMallocManaged((void**)&input_char,sizeof(signed char)*input_char_size);
    cudaMallocManaged((void**)&input_half,sizeof(short)*input_half_size);
    cudaMallocManaged((void**)&compressed_data,sizeof(unsigned char)*compressed_data_size);
    cudaMallocManaged((void**)&average_data,sizeof(double)*average_data_size);
    #else
    cudaMalloc((void**)&input_char,sizeof(signed char)*input_char_size);
    cudaMalloc((void**)&input_half,sizeof(short)*input_half_size);
    cudaMalloc((void**)&average_data,sizeof(double)*average_data_size);
    cudaMallocManaged((void**)&compressed_data,sizeof(unsigned char)*compressed_data_size);
    #endif
    
    float *input_float=(float*)input_half;
    long long int input_half_offset=(fft_length+2)*2*window_size;
    long long int input_float_offset=(fft_length/2+1)*2*window_size;
    
    std::cout<<"fft length = "<<fft_length<<std::endl;
    std::cout<<"signal length = "<<signal_length<<std::endl;
    std::cout<<"signal batch= "<<signal_batch<<std::endl;
    std::cout<<"window size = "<<window_size<<std::endl;
    std::cout<<"step = "<<step<<std::endl;
    std::cout<<"batch buffer size="<<batch_buffer_size<<std::endl;
    std::cout<<"per batch="<<per_batch<<std::endl;
    std::cout<<"compress num="<<compress_num<<std::endl;
    std::cout<<"remain batch="<<remain_batch<<std::endl;
    std::cout<<"input char size = "<<input_char_size<<std::endl;
    std::cout<<"input half size = "<<input_half_size<<std::endl;
    std::cout<<"average data size = "<<average_data_size<<std::endl;
    std::cout<<"compressed data size = "<<compressed_data_size<<std::endl;
    
    //打开写入的文件
    #ifdef WRITE_DATA
    //写入文件头
    #ifdef WRITE_HEADER
    std::string compressed_data_file = file_list[0].substr(0, file_list[0].length() - 4) + ".fil";
    //该函数会根据传入的文件名创建一个新文件,写入并关闭
    write_header((char *)compressed_data_file.c_str(), "rawdata", 8, 21, compress_channel_num, 1, 1, 1, 58849., 0.0,  tsamp*fft_length*step, 1.0/tsamp/1.0e6/fft_length*begin_channel, 1.0/tsamp/1.0e6/fft_length, 0.0, 0.0, 0.0, 0.0);
    std::cout<<"Succeed writing file header ."<<std::endl;
    //此时写入模式为追加模式(写入文件已由write_header函数创建)
    std::ofstream compressed_data_file_stream(compressed_data_file.c_str(),std::ios::app|std::ios::binary);
    #else
    std::string compressed_data_file = file_list[0].substr(0, file_list[0].length() - 4) + "_no_head.fil";
    //此时写入模式为全新模式
    std::ofstream compressed_data_file_stream(compressed_data_file.c_str(),std::ios::trunc|std::ios::binary);
    #endif
    if(compressed_data_file_stream.is_open()) 
    {
        std::cout << "Succeed opening output fft file "<<compressed_data_file<<std::endl;
    }
    else
    {
        std::cout << "Fail to open output fft file "<<compressed_data_file<<std::endl;
    }
    #endif
    
    readfile(input_char_cpu,file_list,file_size_list,fft_length*window_size*2);
    printf("copy input char to gpu memory\n");
    cudaMemcpy(input_char,input_char_cpu,fft_length*window_size*2*sizeof(char),cudaMemcpyHostToDevice);
    //read_char_array(input_char, simulate_input_char ,fft_length*window_size*2);
    char2float_interlace(input_char,input_half,fft_length,window_size,thread_num);
    #ifdef PRINT_MEMORY
    print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
    #endif
    
    fft_1d_half(input_half,input_half,fft_length,window_size*2);
    #ifdef PRINT_MEMORY
    print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
    #endif
    
    complex_modulus_squared_interlace(input_half,input_half,fft_length/2,factor_A,factor_B,window_size,thread_num);
    #ifdef PRINT_MEMORY
    print_data_float(input_float,0,(fft_length/2+1)*batch_buffer_size*2,fft_length/2+1);
    #endif

    channels_sum(input_float,average_data,window_size-step,window_size,(fft_length/2+1)*2,fft_length/2,thread_num);
    #ifdef PRINT_MEMORY
    print_data_double(average_data,0,fft_length/2,fft_length/2);
    #endif

    for(int loop_num=0;loop_num<compress_num;loop_num++)
    {
        printf("compress_loop=%d\n",loop_num);
        
        readfile(input_char_cpu,file_list,file_size_list,fft_length*per_batch*2);
        
        printf("copy input char to gpu memory\n");
        cudaMemcpy(input_char,input_char_cpu,fft_length*per_batch*2*sizeof(char),cudaMemcpyHostToDevice);
        //read_char_array(input_char, simulate_input_char ,fft_length*per_batch*2);        
        char2float_interlace(input_char,input_half+input_half_offset,fft_length,per_batch,thread_num);
        #ifdef PRINT_MEMORY
        print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
        #endif
        
        fft_1d_half(input_half+input_half_offset,input_half+input_half_offset,fft_length,per_batch*2);
        #ifdef PRINT_MEMORY
        print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
        #endif
        
        complex_modulus_squared_interlace(input_half+input_half_offset,input_float+input_float_offset,fft_length/2,factor_A,factor_B,per_batch,thread_num);
        #ifdef PRINT_MEMORY
        print_data_float(input_float,0,(fft_length/2+1)*batch_buffer_size*2,fft_length/2+1);
        #endif
        
        compress(average_data, input_float, input_float+input_float_offset,compressed_data,(fft_length/2+1)*2,per_batch/step, step,begin_channel, compress_channel_num, window_size, thread_num);
        #ifdef PRINT_MEMORY
        print_data_binary(compressed_data,0,(compress_channel_num/8)*(per_batch/step),compress_channel_num/8);
        #endif
        #ifdef WRITE_DATA
        compressed_data_file_stream.write((const char *)compressed_data,(compress_channel_num/8)*(per_batch/step)*sizeof(char));
        #endif
        
        //注意,调用cudaMemcpy时要传入复制的内存大小,因此要乘sizeof
        printf("move fft data in gpu memory\n");
        cudaMemcpy(input_float,input_float+(fft_length/2+1)*2*per_batch,(fft_length/2+1)*2*window_size*sizeof(float),cudaMemcpyDeviceToDevice);
        #ifdef PRINT_MEMORY
        print_data_float(input_float,0,(fft_length/2+1)*batch_buffer_size*2,fft_length/2+1);
        #endif
    }
    
    
     if(remain_batch>0)
     {
         //把余下的数据补到最后一次读取的数据的后面
         printf("begin compress remained batch\n");
         
         readfile(input_char_cpu+fft_length*per_batch*2,file_list,file_size_list,fft_length*remain_batch*2);
         
         printf("copy input char to gpu memory\n");
         cudaMemcpy(input_char+fft_length*per_batch*2,input_char_cpu+fft_length*per_batch*2,fft_length*remain_batch*2*sizeof(char),cudaMemcpyHostToDevice);
         //read_char_array(input_char+fft_length*per_batch*2,simulate_input_char,fft_length*remain_batch*2);
         //先读取余下的数据
         char2float_interlace(input_char+fft_length*per_batch*2,input_half+input_half_offset,fft_length,remain_batch,thread_num);
         #ifdef PRINT_MEMORY
         print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
         #endif

         
         //由于缓冲区空间不足,无法一次性完成反射
         if(per_batch-remain_batch<window_size-step)
         {
             
             //反射读取数据
             printf("complete reflect in two steps\n");
            
             //第一次反射,读取了per_batch-remain_batch,还剩(window_size-step)-(per_batch-remain_batch),反射到window_size+remain_batch的位置
             printf("first step\n");
             char2float_interlace_reflect(input_char+(per_batch+remain_batch-step)*fft_length*2-1,input_half+(fft_length+2)*2*(window_size+remain_batch),fft_length,per_batch-remain_batch,thread_num);
             #ifdef PRINT_MEMORY
             print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
             #endif
             
             //由于缓冲区现在读满了数据,与循环中的处理方式相同
             fft_1d_half(input_half+input_half_offset,input_half+input_half_offset,fft_length,per_batch*2);
             #ifdef PRINT_MEMORY
             print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
             #endif
             
             complex_modulus_squared_interlace(input_half+input_half_offset,input_float+input_float_offset,fft_length/2,factor_A,factor_B,per_batch,thread_num);
             #ifdef PRINT_MEMORY
             print_data_float(input_float,0,(fft_length/2+1)*batch_buffer_size*2,fft_length/2+1);
             #endif
             
             compress(average_data, input_float, input_float+input_float_offset,compressed_data,(fft_length/2+1)*2,per_batch/step, step, begin_channel, compress_channel_num, window_size, thread_num);
             #ifdef PRINT_MEMORY
             print_data_binary(compressed_data,0,(compress_channel_num/8)*(per_batch/step),compress_channel_num/8);
             #endif
             #ifdef WRITE_DATA
             compressed_data_file_stream.write((const char *)compressed_data,(compress_channel_num/8)*(per_batch/step)*sizeof(char));
             #endif
             
             printf("move fft data in gpu memory\n");
             cudaMemcpy(input_float,input_float+(fft_length/2+1)*2*per_batch,(fft_length/2+1)*2*window_size*sizeof(float),cudaMemcpyDeviceToDevice);
             #ifdef PRINT_MEMORY
             print_data_float(input_float,0,(fft_length/2+1)*batch_buffer_size*2,fft_length/2+1);  
             #endif
             
             //第二次反射,还剩window_size-step-per_batch+remain_batch,char的偏移量为(per_batch+remain_batch-step)-(per_batch-remain_batch)=2*remain_batch-step
             printf("second step\n");
             char2float_interlace_reflect(input_char+(2*remain_batch-step)*fft_length*2-1,input_half+input_half_offset,fft_length,(window_size-step)-(per_batch-remain_batch),thread_num);
             #ifdef PRINT_MEMORY
             print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
             #endif
             
             //fft_1d_half(input_half+input_half_offset,input_half+input_half_offset,fft_length,((window_size-step)-(per_batch-remain_batch))*2);
             
             fft_1d_half(input_half+input_half_offset,input_half+input_half_offset,fft_length,per_batch*2);
             #ifdef PRINT_MEMORY
             print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
             #endif
             
             complex_modulus_squared_interlace(input_half+input_half_offset,input_float+input_float_offset,fft_length/2,factor_A,factor_B,((window_size-step)-(per_batch-remain_batch)),thread_num);
             #ifdef PRINT_MEMORY
             print_data_float(input_float,0,(fft_length/2+1)*batch_buffer_size*2,fft_length/2+1);
             #endif
             
             compress(average_data, input_float, input_float+input_float_offset,compressed_data,(fft_length/2+1)*2,((window_size-step)-(per_batch-remain_batch))/step+1, step, begin_channel, compress_channel_num, window_size, thread_num);
             #ifdef PRINT_MEMORY
             print_data_binary(compressed_data,0,(compress_channel_num/8)*((window_size-step)-(per_batch-remain_batch))/step+1,compress_channel_num/8);
             #endif
             #ifdef WRITE_DATA
             compressed_data_file_stream.write((const char *)compressed_data,(compress_channel_num/8)*(((window_size-step)-(per_batch-remain_batch))/step+1)*sizeof(char));
             #endif
         }
        //缓冲区空间够用,可以一次性完成反射
         else
         {
             printf("complete reflect in one step\n");
             
             //从char的数据末尾开始反射读取,per_batch+remain_batch-step为数据末尾回退一个step的位置
             //读取到float末尾位置,偏移量为window_size+remain_batch
             //读取量为window_size-step
             char2float_interlace_reflect(input_char+(per_batch+remain_batch-step)*fft_length*2-1,input_half+(fft_length+2)*2*(window_size+remain_batch),fft_length,window_size-step,thread_num);
             #ifdef PRINT_MEMORY
             print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
             #endif
             
             //计算量为remain_batch+window_size-step
             
             //fft_1d_half(input_half+input_half_offset,input_half+input_half_offset,fft_length,(remain_batch+window_size-step)*2);
             fft_1d_half(input_half+input_half_offset,input_half+input_half_offset,fft_length,per_batch*2);
             
             #ifdef PRINT_MEMORY
             print_data_half(input_half,0,(fft_length+2)*batch_buffer_size*2,fft_length+2);
             #endif
             
             //计算量为remain_batch+window_size-step
             complex_modulus_squared_interlace(input_half+input_half_offset,input_float+input_float_offset,fft_length/2,factor_A,factor_B,(remain_batch+window_size-step),thread_num);
             #ifdef PRINT_MEMORY
             print_data_float(input_float,0,(fft_length/2+1)*batch_buffer_size*2,fft_length/2+1);
             #endif
             
             //压缩量为(remain_batch+window_size-step)/step+1
             compress(average_data, input_float, input_float+input_float_offset,compressed_data,(fft_length/2+1)*2,(window_size-step+remain_batch)/step+1, step, begin_channel, compress_channel_num, window_size, thread_num);
             #ifdef PRINT_MEMORY
             print_data_binary(compressed_data,0,(compress_channel_num/8)*((window_size-step+remain_batch)/step+1),compress_channel_num/8);
             #endif
             #ifdef WRITE_DATA
             compressed_data_file_stream.write((const char *)compressed_data,(compress_channel_num/8)*((window_size-step+remain_batch)/step+1)*sizeof(char));
             #endif
         }
     }
    #ifdef WRITE_DATA
    compressed_data_file_stream.close();
    #endif
    //关闭可能仍处于打开状态的文件
    readfile(input_char_cpu,file_list,file_size_list,-1);
    cudaFree(input_char_cpu);
    cudaFree(input_char);
    cudaFree(input_half);
    cudaFree(average_data);
    cudaFree(compressed_data);
    
    return 0;
}
