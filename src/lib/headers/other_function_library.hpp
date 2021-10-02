#include <string>
#include <iostream>
void generate_file_list(int argc, char *argv[],std::string file_list[]);
void readfile(signed char *input_char_cpu,std::string file_list[],long long int size);
void print_data_signed_char(signed char* data,long long int begin,long long int end);
void print_data_half(short* data,long long int begin,long long int end);
void print_data_half_for_copy(short* data,long long int begin,long long int end);
void float_2_half(void *input_float_void,void *output_half_void,long long int begin,long long int end);
