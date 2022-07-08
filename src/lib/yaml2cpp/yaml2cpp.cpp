#include <stdio.h>
#include <yaml.h>
#include <map>
#include <vector>
#include <stack>
#include <string>
#include <iostream>
#include <yaml2cpp.hpp>
#include <exception>

#ifdef DEBUG
    constexpr bool debug_flag = 1;
#else
    constexpr bool debug_flag = 0;
#endif

cpp_parser::cpp_parser(FILE *inputfile)
{
    /* Initialize parser */
    if(!yaml_parser_initialize(&(this->c_parser)))
    {
        throw std::runtime_error("Failed to initialize parser!");
    }
    /* Set input file */
    if(inputfile == NULL)
    {
        throw std::runtime_error("Failed to open file!");
    }
    yaml_parser_set_input_file(&(this->c_parser), inputfile);
    std::cout<<"create a cpp_parser with a input file"<<std::endl;
    this->state=TO_UPDATE;
};

cpp_parser::~cpp_parser(void)
{
    /* Cleanup */
    yaml_event_delete(&(this->event));
    yaml_parser_delete(&(this->c_parser));
};

void cpp_parser::get_next_event(void)
{
    if(debug_flag){std::cout<<"call function get_next_event"<<std::endl;}
    if (!yaml_parser_parse(&(this->c_parser), &(this->event)))
    {
        std::cerr<<"Parser error "<<this->c_parser.error<<std::endl;
        throw std::runtime_error("Parser error");
    }
    if(debug_flag){std::cout<<"get_next_event is deciding event type"<<std::endl;}
    switch(this->event.type)
    {
    case YAML_NO_EVENT:
        if(debug_flag){puts("No event!");}
        this->get_next_event();
        break;
    /* Stream start/end */
    case YAML_STREAM_START_EVENT:
        if(debug_flag){("STREAM START");}
        this->get_next_event();
        break;
    case YAML_STREAM_END_EVENT:
        if(debug_flag){puts("STREAM END");}
        break;
    /* Block delimeters */
    case YAML_DOCUMENT_START_EVENT: 
        if(debug_flag){puts("Start Document");}
        this->get_next_event();
        break;
    case YAML_DOCUMENT_END_EVENT:
        if(debug_flag){puts("End Document");}
        this->get_next_event();
        break;
    case YAML_ALIAS_EVENT:
        printf("Got alias (anchor %s)\n", event.data.alias.anchor);
        this->get_next_event();
        break;
    default:
        break;
    }
    return;
};

/*
namespace std
{
    template<> struct less<yaml_node>
    {
       bool operator() (const yaml_node& lhs, const yaml_node& rhs) const
       {
           return lhs.scalar < rhs.scalar;
       }
    };
};
*/

yaml_node::yaml_node(void)
{
    this->type = YAML_UNDEFINED;
    this->value = nullptr;
};

yaml_node::yaml_node(const std::string yaml_file_name)
{
    if(debug_flag){std::cout<<"use a file to construct yaml_node"<<std::endl;}
    FILE *yaml_file = fopen(yaml_file_name.c_str(), "r");
    cpp_parser yaml_parser(yaml_file);
    yaml_node tmp_yaml_node(yaml_parser);
    fclose(yaml_file);
    *this=std::move(tmp_yaml_node);
    return;
};

yaml_node::yaml_node(cpp_parser &parser)
{
    if(debug_flag){std::cout<<"use cpp_parser to construct yaml_node"<<std::endl;}
    if(parser.state==TO_UPDATE)
    {
        parser.get_next_event();
        parser.state=BEEN_UPDATED;
    }
    switch(parser.event.type)
    {
    case YAML_MAPPING_START_EVENT:
        if(debug_flag){puts("Start Mapping");}
        this->type = YAML_MAP;
        this->value= (void*)new yaml_node::map;
        parser.get_next_event();
        while(parser.event.type!=YAML_MAPPING_END_EVENT)
        {
            //add <key value> to map
            //为什么需要(char *)转换?文档上是unsigned char yaml_char_t
            std::string key_value=std::string((char *)(parser.event.data.scalar.value));
            if(debug_flag){std::cout<<"add a key "<<key_value<<" to map"<<std::endl;}
            parser.get_next_event();
            yaml_node* value_node=new yaml_node(parser);
            ((yaml_node::map*)(this->value))->insert(yaml_node::map_pair{key_value,value_node});
        };
        if(debug_flag){puts("End Mapping");}
        parser.get_next_event();
        break;
    case YAML_SEQUENCE_START_EVENT:
        if(debug_flag){puts("Start Sequence");}
        this->type = YAML_SEQUENCE;
        this->value=(void*)new yaml_node::seq;
        parser.get_next_event();
        while(parser.event.type!=YAML_SEQUENCE_END_EVENT)
        {
            yaml_node* sequence_node = new yaml_node(parser);
            ((yaml_node::seq*)(this->value))->push_back(sequence_node);
        };
        if(debug_flag){puts("End Sequence");}
        parser.get_next_event();
        break;
    case YAML_SCALAR_EVENT:
        if(debug_flag){printf("Got scalar (value %s)\n", (char *)(parser.event.data.scalar.value));}
        this->type = YAML_SCALAR;
        this->value = (void*)new yaml_node::scalar((char *)(parser.event.data.scalar.value));
        parser.get_next_event();
        break;
    default:
        throw std::runtime_error("Fail to create new yaml node ,received invalid yaml event");
        break;
    }
    /*
    if(event.type != YAML_STREAM_END_EVENT)
        yaml_event_delete(&event);
    } while(event.type != YAML_STREAM_END_EVENT);
    */
};

yaml_node::~yaml_node(void)
{
    switch(this->type)
    {
        case(YAML_MAP):
            for(auto iter = ((yaml_node::map*)(this->value))->begin();iter!=((yaml_node::map*)(this->value))->end();iter++)
            {
                delete iter->second;
            }
            if(debug_flag){std::cout<<"free map"<<std::endl;}
            delete (yaml_node::map*)(this->value); 
            break;
        case(YAML_SEQUENCE):
            for(auto iter = ((yaml_node::seq*)(this->value))->begin();iter!=((yaml_node::seq*)(this->value))->end();iter++)
            {
                delete *iter;
            }
            if(debug_flag){std::cout<<"free sequence"<<std::endl;}
            delete (yaml_node::seq*)(this->value);
            break;
        case(YAML_SCALAR):
            if(debug_flag){std::cout<<"free scalar , scalar value is "<<*(yaml_node::scalar*)(this->value)<<std::endl;}
            delete (yaml_node::scalar*)(this->value);
            break;
        default:
            break;
    }
};

yaml_node::yaml_node (yaml_node & old_node):type(old_node.type)
{
    switch(this->type)
    {
        case(YAML_MAP):
            if(debug_flag){std::cout<<"copy a map node"<<std::endl;}
            this->value=(void*)new yaml_node::map;
            for(auto iter = ((yaml_node::map*)(old_node.value))->begin();iter!=((yaml_node::map*)(old_node.value))->end();iter++)
            {
                if(debug_flag){std::cout<<"add a key "<<iter->first<<" to copied map"<<std::endl;}
                yaml_node* value_node=new yaml_node(*(iter->second));
                ((yaml_node::map*)(this->value))->insert(yaml_node::map_pair{iter->first,value_node}); 
            }            
            break;
        case(YAML_SEQUENCE):
            if(debug_flag){std::cout<<"copy a sequence node"<<std::endl;}
            this->value=(void*)new yaml_node::seq;
            for(auto iter = ((yaml_node::seq*)(old_node.value))->begin();iter!=((yaml_node::seq*)(old_node.value))->end();iter++)
            {
                yaml_node* sequence_node=new yaml_node(**iter);
                ((yaml_node::seq*)(this->value))->push_back(sequence_node);
            }
            break;
        case(YAML_SCALAR):
            if(debug_flag){std::cout<<"copy a scalar node"<<std::endl;}
            this->value = (void*)new yaml_node::scalar(*((yaml_node::scalar*)(old_node.value)));
            break;
        default:
            this->value = nullptr;
            break;
    }
};

yaml_node::yaml_node (yaml_node && old_node):type(old_node.type),value(old_node.value)
{
    old_node.type=YAML_UNDEFINED;
    old_node.value=nullptr;
};

yaml_node& yaml_node::operator = (yaml_node && old_node)
{
    this->type = old_node.type;
    this->value = old_node.value;
    old_node.type=YAML_UNDEFINED;
    old_node.value=nullptr;
    return *this;
};

yaml_node & yaml_node::operator ()(std::string input_key)
{
    if(debug_flag){std::cout<<"query a key of value "<<input_key<<std::endl;}
    switch(this->type)
    {
        case(YAML_MAP):
            if(((yaml_node::map*)(this->value))->count(input_key))
            {
                return *((*(yaml_node::map*)(this->value))[input_key]);
            }
            else
            {
                return (*this);
                std::cerr<<"input key does not exist"<<std::endl;
            }
            break;
        default:
            std::cerr<<"input node is not of type map"<<std::endl;
            return (*this);
    }
};
    
yaml_node & yaml_node::operator ()(int input_index)
{
    if(debug_flag){std::cout<<"query a sequence of index "<<input_index<<std::endl;}
    switch(this->type)
    {
        case(YAML_SEQUENCE):
            if(input_index<=(((yaml_node::seq*)(this->value))->size()))
            {
                return *((*(yaml_node::seq*)(this->value))[input_index]);
            }
            else
            {
                std::cerr<<"input index is too large"<<std::endl;
                return *this;
            }
            break;
        default:
            std::cerr<<"input node is not of type sequence"<<std::endl;
            return *this;
    }
};

std::string & yaml_node::operator ()(void)
    {
        //用于在发生错误时返回一个空字符串
        static std::string null_string = "error when use operator () to get a yaml value";
        switch(this->type)
        {
            case(YAML_SCALAR):
                return *((yaml_node::scalar*)(this->value));
                break;
            default:
                std::cerr<<"input node is not of type scalar"<<std::endl;
                return null_string;
        }
    };

