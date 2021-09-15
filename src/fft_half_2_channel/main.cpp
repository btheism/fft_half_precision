#include <string>
#include <other_function_library.hpp>
#include <cuda_runtime.h>


int main(int argc, char *argv[]) {

    //生成文件列表
    std::string file_list[argc-1];
    generate_file_list(argc ,argv,file_list);
    //
    signed char *input_char;
    std::cout<<"parameter address before is "<<(void*)input_char<<std::endl;
    input_char=new signed char [1200];

    //cudaMallocManaged((void**)&input_char,sizeof(unsigned char)*1024);

    std::cout<<"parameter address after is "<<(void*)input_char<<std::endl;
    readfile(input_char,file_list,1024);
    print_data_signed_char(input_char,0,1024);
    delete input_char;
    return 0;
}

