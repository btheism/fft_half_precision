#include<iostream>
#include<fstream>
#include<string>

#if __GNUC__ <= 8

#include <sys/stat.h>
long long int GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    std::cout<<"Using version A."<<std::endl;
    return rc == 0 ? stat_buf.st_size : -1;
}

#endif

#if __GNUC__ > 8

#include <filesystem>
long long int GetFileSize(std::string filename)
{
    std::cout<<"Using version B."<<std::endl;
    return std::filesystem::file_size(filename);
}

#endif


int main(int argc , char *argv[])
{
     std::ifstream original_data;
     original_data.open((std::string)argv[1],std::ios::in|std::ios::binary);
     char * input_char=new char[20];
     original_data.read((char *) input_char,16*sizeof(char));
     original_data.close();
     for(int i=0;i<16;i++)
         std::cout<<"Character "<<i<<" is "<<input_char[i]<<" ."<<std::endl;
     
     std::cout<<"File size is "<<GetFileSize((std::string)argv[1])<<" ."<<std::endl;
     delete input_char;
     return 0;
}
