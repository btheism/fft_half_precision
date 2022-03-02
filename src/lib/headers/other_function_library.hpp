#include <string>
#include <iostream>

long long int GetFileSize(char* filename);
long long int GetFilelistSize(int file_num , char** file_list);
void readfile(char *input_char,char** file_list ,long long int ask_size);
void read_char_array(signed char *input_char,signed char * simulate_array ,long long int ask_size);
void print_data_signed_char(char* data,long long int begin,long long int end,int break_num);
void print_data_half(short* data,long long int begin,long long int end,int break_num);
void print_data_float(float* data,long long int begin,long long int end,int break_num);
void print_data_double(double* data,long long int begin,long long int end,int break_num);
void print_data_binary(char* data,long long int begin,long long int end,int break_num);
void print_data_half_for_copy(short* data,long long int begin,long long int end, int break_num);
void print_data_float_for_copy(float* data,long long int begin,long long int end);
void float_2_half(void *input_float_void,void *output_half_void,long long int begin,long long int end);

