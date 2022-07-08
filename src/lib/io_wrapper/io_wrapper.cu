#ifdef DEBUG
constexpr bool debug_flag=1;
#else
constexpr bool debug_flag=0;
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <limits.h>
#include <unistd.h>
#include <cstring>
#include <cuda_fp16.h>
#include <sstream>
#include <vector>
#include <array>
#include <iterator>
#include <exception>
//#include <unistd.h>

//获取程序绝对路径的函数
std::string GetExePath(void)
{
    char result[ PATH_MAX ];
    ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
    return std::string( result, (count > 0) ? count : 0 );
};

std::string GetExeDirPath(void)
{
    //find_las_of:从后往前查找，找到参数字符串中的任意一个字符，则返回该字符的索引
    std::string str=GetExePath();
    auto location = str.find_last_of("/");
    location++;
    return str.substr(0, location);
};

//一个获取文件大小的函数
#include <sys/stat.h>
long long int GetFileSize(const char* filename)
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

//读取数据的函数,其中ask_size以char为单位,input_char指向读出数据的存储位置

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

//读取数据的函数,其中ask_size以char为单位
//C++风格，用于处理有多个独立的输入流的情况（例如信号有多路）
class read_stream
{
public:
    std::ifstream *original_data;
    //记录当前打开的文件编号
    int file_number;
    //记录文件的总大小和未读取部分的大小
    long long int stream_size;
    long long int stream_remain_size;
    //记录当前打开的文件的大小和未读取部分的大小
    long long int file_remain_size;
    long long int read_remain_size;
    long long int read_size;
    std::vector< std::string> filelist;
    read_stream(const std::string & filelist);
    read_stream(const std::vector<std::string> & filelist ,long long int stream_size);
    read_stream(read_stream& old_stream) = delete;
    read_stream(read_stream&& old_stream);
    int read(char *input_char, long long int ask_size);
    ~read_stream(void);
};

//按空格分割字符串的函数
std::vector<std::string> split_names_by_space(const std::string & file_names)
{
    if(debug_flag){std::cout<<"split string "<<file_names<<" by space"<<std::endl;}
    std::vector<std::string> file_list;
    std::stringstream names_stream(file_names);
    std::istream_iterator<std::string> name_iterator(names_stream),stream_end;
    while(name_iterator!=stream_end)
    {
        if(debug_flag){std::cout<<"get "<<*name_iterator<<std::endl;}
        file_list.emplace_back(*name_iterator);
        name_iterator++;
    }
    if(debug_flag){std::cout<<"string is splitted successfully"<<std::endl;}
    return file_list;
};

//用一个包含所有输入文件名的字符串初始化，文件名之间用空格分割
read_stream::read_stream(const std::string & filelist):filelist(split_names_by_space(filelist))
{
    this->original_data=new std::ifstream();
    this->file_number=0;
    this->file_remain_size=0;
    this->read_remain_size=0;
    this->read_size=0;
    this->stream_size=0;
    for(int serial=0 ; serial<this->filelist.size() ; serial++)
    {
        this->stream_size+=GetFileSize(this->filelist[serial].c_str());
    }
    this->stream_remain_size=this->stream_size;
};

read_stream::read_stream(const std::vector<std::string> & filelist ,long long int stream_size):filelist(filelist)
{
    if(debug_flag)
    {
        std::cout<<"create a new stream from a filelist\n";
        std::cout<<"list contains "<<filelist.size()<<" file(s)"<<std::endl;
        for(int serial=0 ; serial<filelist.size(); serial++)
        {
            std::cout<<"file "<<serial<<" is "<<this->filelist[serial]<<std::endl;
        }
    }
    this->original_data=new std::ifstream();
    this->file_number=0;
    this->file_remain_size=0;
    this->read_remain_size=0;
    this->read_size=0;
    this->stream_size=stream_size;
    this->stream_remain_size=stream_size;
    if(debug_flag){std::cout<<"stream is created"<<std::endl;}
};

int read_stream::read(char *input_char , long long int ask_size)
{
    if(ask_size > this->stream_remain_size)
    {
        throw std::runtime_error("asked size is too big");
    }
    else
    {
    this->read_remain_size=ask_size;
    this->read_size=0;
    std::cout<<"read char data to cpu memory , size = "<<ask_size<<std::endl;
    while(this->read_remain_size>0)
    {
        if(!this->original_data->is_open())
        {
            this->original_data->open(this->filelist[file_number].c_str(),std::ios::in|std::ios::binary);
            if(this->original_data->is_open())
            {
                std::cout << "Succeed opening input file "<<this->filelist[file_number]<<std::endl;
                this->file_remain_size=GetFileSize(this->filelist[file_number].c_str());
            }
            else
            {
                std::cout << "Fali to open input file "<<this->filelist[file_number]<<std::endl;
                return -1;
            }
        }

        if(this->read_remain_size<this->file_remain_size)
        {
            this->original_data->read( input_char+this->read_size, this->read_remain_size);
            this->file_remain_size-=this->read_remain_size;
            this->read_size+=this->read_remain_size;
            this->read_remain_size=0;
        }
        else
        {
            this->original_data->read( input_char+this->read_size, file_remain_size);
            this->read_remain_size-=this->file_remain_size;
            this->read_size+=this->file_remain_size;
            this->file_remain_size=0;
            this->original_data->close();
            std::cout << "Close file "<<this->filelist[file_number]<<std::endl;
            this->file_number++;
        }
    }
    this->stream_remain_size-=ask_size;
    return 0;
    }
};

read_stream::~read_stream(void)
{
    if(debug_flag){std::cout<<"try to delete a read_stream"<<std::endl;};
    if(this->original_data!=nullptr)
    {
        if(this->original_data->is_open())
        {
            this->original_data->close();
            std::cout << "Close file "<<this->filelist[this->file_number]<<std::endl;
        }
    }
    delete this->original_data;
    if(debug_flag){std::cout<<"read_stream is deleted"<<std::endl;};
    return;
};

read_stream::read_stream(read_stream&& old_stream)
{
    if(debug_flag){std::cout<<"read_stream class is being moved"<<std::endl;}
    this->original_data=old_stream.original_data;
    old_stream.original_data=nullptr;
    this->file_number=old_stream.file_number;
    this->stream_size=old_stream.stream_size;
    this->stream_remain_size=old_stream.stream_remain_size;
    this->file_remain_size=old_stream.file_remain_size;
    this->read_remain_size=old_stream.read_remain_size;
    this->read_size=old_stream.read_size;
    this->filelist=old_stream.filelist;
    if(debug_flag){std::cout<<"read_stream class is moved successfully"<<std::endl;}
}

//该函数返回输入文件的总大小,便于主程序决定循环次数等参数
long long int GetFilelistSize(int file_num , char** filelist)
{
    long long int file_total_size=0;
    for (int i = 0; i < file_num; i++){
        std::cout<<"Input file "<<i<<" is "<<filelist[i]<<std::endl;
        file_total_size+=GetFileSize(filelist[i]);
    }
    return file_total_size;
}

long long int GetFilelistSize(const std::vector<std::string> & filelist)
{
    long long int list_size=0;
    for(int serial=0 ; serial<filelist.size() ; serial++)
    {
        list_size+=GetFileSize(filelist[serial].c_str());
    }
    return list_size;
};

long long int GetFilelistSize(const std::string & filelist)
{
    return GetFilelistSize(split_names_by_space(filelist));
};

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
    std::cout.precision(6);
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
    std::cout.precision(10);
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
    std::cout.precision(10);
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
    std::cout.precision(6);
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
    std::cout.precision(10);
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


