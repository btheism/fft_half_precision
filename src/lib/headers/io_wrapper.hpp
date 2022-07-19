#include <string>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

long long int GetFileSize(const char* filename);
long long int GetFilelistSize(const std::vector<std::string> & filelist);
long long int GetFilelistSize(const std::string & filelist);
std::vector<std::string> split_names_by_space(const std::string & file_names);
std::string GetExeDirPath(void);
void readfile(char *input_char,char** file_list ,long long int ask_size);
void read_char_array(signed char *input_char,signed char * simulate_array ,long long int ask_size);
void print_data_signed_char(signed char* data,long long int begin,long long int end,int break_num);

//根据传入的flag的不同可以打印多种整型的函数
void print_data_multi_int(void* data,long long int begin,long long int end,int break_num ,int input_type_flag);

void print_data_half(short* data,long long int begin,long long int end,int break_num);
void print_data_float(float* data,long long int begin,long long int end,int break_num);
void print_data_double(double* data,long long int begin,long long int end,int break_num);
void print_data_binary(char* data,long long int begin,long long int end,int break_num);
void print_data_half_for_copy(short* data,long long int begin,long long int end, int break_num);
void print_data_float_for_copy(float* data,long long int begin,long long int end, int break_num);
void float_2_half(void *input_float_void,void *output_half_void,long long int begin,long long int end);

class read_stream
{
public:
    std::ifstream *original_data;
    int file_number;
    long long int stream_size;
    long long int stream_remain_size;
    long long int file_remain_size;
    long long int read_remain_size;
    long long int read_size;
    std::vector<std::string> filelist;
    read_stream(const std::string& filelist);
    read_stream(const std::vector<std::string>& filelist ,long long int stream_size);
    read_stream(read_stream& old_stream) = delete;
    read_stream(read_stream&& old_stream);
    int read(char *input_char , long long int ask_size);
    ~read_stream(void);
};
