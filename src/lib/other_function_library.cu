#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cuda_fp16.h>
//#include <unistd.h>
#include <stdlib.h>


//一个获取文件大小的函数
#include <sys/stat.h>
long long int GetFileSize(char* filename)
{
    struct stat stat_buf;
    int rc = stat(filename, &stat_buf);
    if(rc==0)
    {
        printf("File %s 's size is %lld.\n" , filename,(long long int)stat_buf.st_size);
    }
    else
    {
        printf("fail to open file %s , exit.\n" , filename);
        exit(-1);
    }
    return rc == 0 ? stat_buf.st_size : -1;
}

//读取数据的函数,其中ask_size以char为单位

void readfile(char *input_char,char** file_list,long long int ask_size)
{
    std::cout<<"read char data to cpu memory , size = "<<ask_size<<std::endl;
    static std::ifstream original_data;
    static int file_number=0;
    static long long int file_remain_size=0;
    long long int read_remain_size=ask_size;
    long long int read_size=0;
    //打开文件
 
    while(read_remain_size>0)
    {
        if(!original_data.is_open())
        {
            original_data.open(file_list[file_number],std::ios::in|std::ios::binary);
            if(original_data.is_open())
            {
                std::cout << "Succeed opening input file "<<file_list[file_number]<<std::endl;
                file_remain_size=GetFileSize(file_list[file_number]);
            }
            else
            {
                std::cout << "Fali to open input file "<<file_list[file_number]<<std::endl;
            }
        }

        if(read_remain_size<file_remain_size)
        {
            original_data.read((char *) input_char+read_size, read_remain_size* sizeof(char));
            file_remain_size-=read_remain_size;
            read_size+=read_remain_size;
            read_remain_size=0;
        }
        else
        {
            original_data.read((char *) input_char+read_size, file_remain_size* sizeof(char));
            read_remain_size-=file_remain_size;
            read_size+=file_remain_size;
            file_remain_size=0;
            original_data.close();
            std::cout << "Close file "<<file_list[file_number]<<std::endl;
            file_number++;
        }
    }
    
    //假设退出程序时有未关闭的文件,可向该函数传递参数-1以关闭文件
    if(read_remain_size==-1)
    {
         if(!original_data.is_open())
         {
             std::cout << "All input files are closed"<<file_list[file_number]<<std::endl;
         }
         else
         {
             original_data.close();
             std::cout << "Close file "<<file_list[file_number]<<std::endl;
         }
    }
} 

//该函数用于模拟readfile的行为,仅用于调试
void read_char_array(signed char *input_char,signed char * simulate_array ,long long int ask_size)
{
    static long long int read_size=0;
    std::memcpy(input_char,simulate_array+read_size,ask_size);
    read_size+=ask_size;
    //把数据从input_char复制到input_float,并分离两个通道的数据
    //cudaMemcpy(input_char_gpu,input_char_cpu,size*sizeof(signed char),cudaMemcpyHostToDevice);
    //cudaDeviceSynchronize();
}

//struct fft_half_1_channel_parameter_list 
//{
//    
//}

//该函数返回输入文件的总大小,便于主程序决定循环次数等参数
//注意，新的参数解析函数已不再使用该函数
long long int generate_file_list(int argc ,char *argv[],char* file_list[],long long int file_size_list[])
{
    long long int file_total_size=0;
    int file_num= argc - 1;
    for (int i = 0; i < file_num; i++){
        file_list[i] = argv[i+1];
        std::cout<<"Input file "<<i<<" is "<<file_list[i]<<std::endl;
        file_size_list[i]=GetFileSize(file_list[i]);
        file_total_size+=file_size_list[i];
    }
    return file_total_size;
}

void print_data_signed_char(char* data,long long int begin,long long int end,int break_num)
{
    std::cout<<"Print signed char data from "<<begin<<" to "<<end<<std::endl;
    char newline=0;
    for(int i=begin;i<end;i++)
    {
            std::cout<<i<<"\t"<<(int)*((signed char*)data+i)<<"\t";
            newline++;
            if(newline==break_num)
            {
                std::cout << std::endl;
                newline=0;
            }
    }
    if(newline!=break_num)
        std::cout << std::endl;
}


void print_data_half(short* data,long long int begin,long long int end, int break_num)
{
    std::cout<<"Print half data from "<<begin<<" to "<<end<<std::endl;
    std::cout.precision(2);
    std::cout<<std::fixed;
    char newline=0;
    for(int i=begin;i<end;i++)
    {
            std::cout<<i<<"\t"<<__half2float(*(half*)(data+i))<<"\t";
            newline++;
            if(newline==break_num)
            {
                std::cout << std::endl;
                newline=0;
            }
    }
    if(newline!=break_num)
        std::cout << std::endl;
}

void print_data_float(float* data,long long int begin,long long int end,int break_num)
{
    std::cout<<"Print float data from "<<begin<<" to "<<end<<std::endl;
    std::cout.precision(2);
    std::cout<<std::fixed;
    char newline=0;
    for(int i=begin;i<end;i++)
    {
            std::cout<<i<<"\t"<<data[i]<<"\t";
            newline++;
            if(newline==break_num)
            {
                std::cout << std::endl;
                newline=0;
            }
    }
    if(newline!=break_num)
        std::cout << std::endl;
}

void print_data_double(double* data,long long int begin,long long int end,int break_num)
{
    std::cout<<"Print double data from "<<begin<<" to "<<end<<std::endl;
    std::cout.precision(2);
    std::cout<<std::fixed;
    char newline=0;
    for(int i=begin;i<end;i++)
    {
            std::cout<<i<<"\t"<<data[i]<<"\t";
            newline++;
            if(newline==break_num)
            {
                std::cout << std::endl;
                newline=0;
            }
    }
    if(newline!=break_num)
        std::cout << std::endl;
}

void print_char_binary(char c) {
    for (char i = 0; i <=7 ; i++) {
        std::cout << ((c & (1 << i))? '1' : '0');
    }
}

void print_data_binary(char* data,long long int begin,long long int end,int break_num)
{
    std::cout<<"Print binary data from "<<begin<<" to "<<end<<std::endl;
    char newline=0;
    for(int i=begin;i<end;i++)
    {
        
        std::cout<<i<<"\t";
        print_char_binary(data[i]);
        std::cout<<"\t";
        newline++;
        if(newline==break_num)
        {
            std::cout << std::endl;
            newline=0;
        }
    }
    if(newline!=break_num)
        std::cout << std::endl;
}

void print_data_half_for_copy(short* data_short,long long int begin,long long int end, int break_num)
{
    half *data=(half*)data_short;
    std::cout<<"Print half data for copy from "<<begin<<" to "<<end<<std::endl;
    char newline=0;
    std::cout.precision(4);
    std::cout<<std::fixed;
    for(int i=begin;i<end;i++)
    {
        std::cout<<(float)data[i]<<",";
        newline++;
        if(newline==break_num)
        {
            std::cout << std::endl;
            newline=0;
        }
    }
    if(newline!=break_num)
        std::cout << std::endl;
}

void print_data_float_for_copy(float* data,long long int begin,long long int end)
{
    std::cout<<"Print float data for copy from "<<begin<<" to "<<end<<std::endl;
    std::cout.precision(2);
    std::cout<<std::fixed;
    for(int i=begin;i<end;i++)
    {
            std::cout<<data[i]<<",";
    }
    std::cout<<std::endl;
}

void float_2_half(void *input_float_void,void *output_half_void,long long int begin,long long int end)
{
    float *input_float=(float*)input_float_void;
    half *output_half=(half*)output_half_void;
    std::cout<<"Convert float to half , from "<<begin<<" to "<<end<<std::endl;
     for(int i=begin;i<end;i++)
    {
         output_half[i]=__float2half(input_float[i]);
    }
}


