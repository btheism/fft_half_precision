#include <yaml.h>
#include <stdio.h>
#include <map>
#include <vector>

typedef enum{UNINITIALIZED , TO_UPDATE , BEEN_UPDATED} cpp_parser_state;
typedef enum{YAML_MAP , YAML_SEQUENCE , YAML_SCALAR , YAML_UNDEFINED} yaml_node_type;

class cpp_parser
{
public:
    cpp_parser_state state;
    yaml_parser_t c_parser;
    yaml_event_t  event; 
    void get_next_event(void);
    cpp_parser(FILE *inputfile);
    ~cpp_parser(void);
    cpp_parser(cpp_parser& old_parser) = delete;
    cpp_parser(cpp_parser&& old_parser) = delete;
};

class yaml_node
{
public:
    typedef std::map<std::string ,yaml_node*> map;
    typedef std::pair<std::string ,yaml_node*> map_pair;
    typedef std::string scalar;
    typedef std::vector<yaml_node*> seq;
    yaml_node_type type ;
    void * value;
    yaml_node(const std::string yaml_file_name);
    yaml_node(cpp_parser & parser);
    yaml_node (yaml_node & old_node);
    yaml_node (yaml_node && old_node);
    yaml_node(void);
    ~yaml_node(void);
    yaml_node & operator = (yaml_node && old_node);
    yaml_node & operator ()(std::string input_key);
    yaml_node & operator ()(int input_index);
    std::string & operator ()(void);
};
//也许应该尝试一下把yaml_node分成三个子类？
