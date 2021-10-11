#include <string>
#include <iostream>
long long int generate_file_list(int argc ,char *argv[],std::string file_list[],long long int file_size[]);
void readfile(signed char *input_char,std::string * file_list, long long int file_size[],long long int size);
void print_data_signed_char(signed char* data,long long int begin,long long int end);
void print_data_half(short* data,long long int begin,long long int end);
void print_data_float(float* data,long long int begin,long long int end);
void print_data_double(double* data,long long int begin,long long int end);
void print_data_half_for_copy(short* data,long long int begin,long long int end);
void print_data_float_for_copy(float* data,long long int begin,long long int end);
void float_2_half(void *input_float_void,void *output_half_void,long long int begin,long long int end);
