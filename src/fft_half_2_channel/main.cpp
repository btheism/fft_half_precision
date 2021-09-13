#include <string>
#include <other_function_library.hpp>
#include <cuda_runtime.h>


int main(int argc, char *argv[]) {

    //生成文件列表
    std::string file_list[argc-1];
    generate_file_list(argc ,argv,file_list);
    //
    signed char *input_char;
    cudaMallocManaged((void**)&input_char,sizeof(unsigned char)*1024);
    readfile(input_char,file_list,1024);
    print_data_signed_char(input_char,0,15);

}

