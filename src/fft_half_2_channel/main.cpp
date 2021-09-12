#include <string>
#include <other_function_library.hpp>

int main(int argc, char *argv[]) {

    std::string file_list[argc-1];
    generate_file_list(argc ,argv,file_list);

}

