#include <iostream>
#include <fstream>
#include <cuda_fp16.h>

#if __GNUC__ <= 15

#include <sys/stat.h>
long long int GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

#else

#include <filesystem>
long long int GetFileSize(std::string filename)
{
    return std::filesystem::file_size(filename);
}

#endif


void readfile(signed char *input_char,std::string * file_list,long long int size)
{
    static std::ifstream original_data;
    static int file_number=0;
    static unsigned long long int file_remain_size=0;
    unsigned long long int read_remain_size=size;
    unsigned long long int read_size=0;
    //打开文件

    while(read_remain_size>0)
    {
        if(!original_data.is_open())
        {
            original_data.open(file_list[file_number].c_str(),std::ios::in|std::ios::binary);
            if(original_data.is_open())
            {
                std::cout << "Succeed opening input file "<<file_list[file_number]<<"\n";
                
                file_remain_size=GetFileSize(file_list[file_number]);
                std::cout << "File size is "<<file_remain_size<<"\n";
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
            file_number++;
        }
    }

    //把数据从input_char复制到input_float,并分离两个通道的数据
    //cudaMemcpy(input_char_gpu,input_char_cpu,size*sizeof(signed char),cudaMemcpyHostToDevice);
    //cudaDeviceSynchronize();
} 


void generate_file_list(int argc ,char *argv[],std::string file_list[])
{
    int file_num= argc - 1;
    for (int i = 0; i < file_num; i++){
        file_list[i] = argv[i+1];
        std::cout<<"Input file "<<i<<" is "<<file_list[i]<<std::endl;
    }

}


void print_data_signed_char(signed char* data,long long int begin,long long int end)
{
    std::cout<<"Print signed char data from "<<begin<<" to "<<end<<std::endl;
    char newline=0;
    for(int i=begin;i<end;i++)
    {
            std::cout<<i<<"\t"<<(int)*(data+i)<<"\t";
            newline++;
            if(newline==8)
            {
                std::cout << std::endl;
                newline=0;
            }
    }
    if(newline!=8)
        std::cout << std::endl;
}


void print_data_half(short* data,long long int begin,long long int end)
{
    std::cout<<"Print half data from "<<begin<<" to "<<end<<std::endl;
    std::cout.precision(2);
    std::cout<<std::fixed;
    char newline=0;
    for(int i=begin;i<end;i++)
    {
            std::cout<<i<<"\t"<<__half2float(*(half*)(data+i))<<"\t";
            newline++;
            if(newline==8)
            {
                std::cout << std::endl;
                newline=0;
            }
    }
    if(newline!=8)
        std::cout << std::endl;
}

void print_data_half_for_copy(short* data,long long int begin,long long int end)
{
    std::cout<<"Print half data for copy from "<<begin<<" to "<<end<<std::endl;
    std::cout.precision(4);
    std::cout<<std::fixed;
    for(int i=begin;i<end;i++)
    {
            std::cout<<__half2float(*(half*)(data+i))<<",";
    }
    std::cout<<std::fixed;
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

