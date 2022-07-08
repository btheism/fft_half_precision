#include <iostream>
#include <yaml2cpp.hpp>
int main(void)
{
    std::string filename = std::string("test.yaml");
    yaml_node root_node(filename);
    std::cout<<"test map :"<<std::endl;
    std::cout<<root_node("key")("key1")()<<std::endl;
    std::cout<<root_node("key")("key2")()<<std::endl;
    std::cout<<"test sequence :"<<std::endl;
    std::cout<<root_node("seq")(0)()<<std::endl;
    std::cout<<root_node("seq")(1)()<<std::endl;
    std::cout<<"test scalar :"<<std::endl;
    std::cout<<root_node("scalar")()<<std::endl;
    std::cout<<"test map with not existed key"<<std::endl;
    std::cout<<root_node("key")("keyx")()<<std::endl;
    std::cout<<"test sequence with not existed serial number"<<std::endl;
    std::cout<<root_node("seq")(114514)()<<std::endl;
    std::cout<<"test node copy"<<std::endl;
    std::cout<<"test empty scalar"<<std::endl;
    std::cout<<root_node("null_scalar")()<<std::endl;
    std::cout<<"compare null_scalar with null string :"<<
        (root_node("null_scalar")()==std::string(""))<<std::endl;
    
    auto copied_node = root_node;
    std::cout<<"test copied node sequence :"<<std::endl;
    std::cout<<copied_node("seq")(1)()<<std::endl;
    return 0;
}
  
