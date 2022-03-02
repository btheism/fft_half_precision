#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <cstring>
#include <fstream>
#include <cuda_runtime.h>

#include<map>
#include<regex.h>
#include <getopt.h>

#include <other_function_library.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>
#include<filheader.h>

//定义了fft_half_1_channel程序所需的参数表,参数的功能在初始化函数initialize_2_channel_parameter_list中有详细说名
struct fft_half_1_channel_parameter_list 
{
    //####以下参量根据输入的文件直接产生####
    
    //文件列表
    char **file_list;
    
    //文件数
    int file_num;
    
    //输入数据的总长度
    long long int signal_length;
    
    //####以下参量为可自定义参数####
    
    char* output_name;
    long long int fft_length;
    long long int window_size;
    long long int step;
    long long int begin_channel;
    long long int compress_channel_num;
    long long int batch_buffer_size;
    double tsamp;
    long long int thread_num;
    
    //以下参数用于写入文件头
    char* source_name_in;
    int machine_id_in;
    int telescope_id_in;
    
    //####下列参数由程序自行计算####

    long long int signal_batch;
    long long int per_batch;
    long long int compress_num;
    long long int remain_batch;


    //####以下为一些标志位,用于程序的调试####
    int write_head;
    int print_memory;
    int write_data;

};

typedef struct fft_half_1_channel_parameter_list pars;

void generate_source_name_in(char* source_name_in,char* file_name)
{
    regex_t match_exp;
    regmatch_t match_location[1];
    size_t nmatch=1;
    //似乎不支持[::]风格？
    regcomp (&match_exp, "_[A-Za-z0-9]*_[A-Za-z0-9]*.dat$", 0);
    if(regexec(&match_exp,file_name,nmatch ,match_location,0)==REG_NOMATCH)
    {
        printf("Fail to set source_name_in by input file name ,source_name_in is set to rawdata\n");
        strcpy(source_name_in,"rawdata");
    }
    else
    {
        //这个函数竟然不会自动在字符串后面补0!必须使用calloc分配字符串空间
        strncpy(source_name_in,file_name,match_location[0].rm_so);
        
    }
    regfree(&match_exp);
}

//该函数解析命令行传递给fft_half_1_channel程序的参数列表，设置程序的各项参数
void initialize_1_channel_parameter_list (int argc ,char **argv ,pars *par)
{
    int option_index;
    int par_type;
    
    //可自定义的参数名称列表,之后会传递给getopt_long()函数
    struct option long_options[] = 
      {
          //参数位
          {"output_name" ,required_argument ,NULL,'s'},
          {"fft_length" ,required_argument ,NULL,'l'},
          {"window_size" ,required_argument ,NULL,'l'},
          {"step" ,required_argument ,NULL,'l'},
          {"begin_channel" ,required_argument ,NULL, 'l'},
          {"compress_channel_num" ,required_argument ,NULL, 'l'},
          {"batch_buffer_size" ,required_argument ,NULL, 'l'},
          {"tsamp",required_argument ,NULL, 'd'},
          {"thread_num",required_argument ,NULL, 'l'},
          
          //标志位
          
          {"write_head" ,required_argument ,NULL, 'i'},
          {"write_data" ,required_argument ,NULL, 'i'},
          {"print_memory" ,required_argument ,NULL, 'i'},
          
          //写入文件头的部分
          {"source_name_in" ,required_argument ,NULL,'s'},
          {"machine_id_in" ,required_argument ,NULL, 'i'},
          {"telescope_id_in" ,required_argument ,NULL, 'i'},
          
          //表示列表结束
          {NULL,0,NULL,0}
      };
    
    //参数的名称-地址表,用于修改参数的值
    /*std::map<std::string ,void *>par_dic={

        {"output_name" ,(void*)&par->output_name},
        {"fft_length",(void*)&par->fft_length},
        {"window_size",(void*)&par->window_size},
        {"step",(void*)&par->step},
        {"factor_A",(void*)&par->factor_A},
        {"factor_B",(void*)&par->factor_B},
        {"begin_channel",(void*)&par->begin_channel},
        {"compress_channel_num",(void*)&par->compress_channel_num},
        {"batch_buffer_size",(void*)&par->batch_buffer_size},
        {"tsamp",(void*)&par->tsamp},
        
        {"write_head" ,(void*)&par->write_head},
        {"write_data" ,(void*)&par->write_data},
        {"print_memory" ,(void*)&par->print_memory},
        
        {"source_name_in" ,(void*)&par->source_name_in},
        {"machine_id_in" ,(void*)&par->machine_id_in},
        {"telescope_id_in" ,(void*)&par->telescope_id_in},
        {"thread_num" ,(void*)&par->thread_num}
    };*/
    
    //C++98
    std::map<std::string ,void *>par_dic;
    par_dic.insert(std::pair<std::string ,void *>("output_name" ,(void*)&par->output_name));
    par_dic.insert(std::pair<std::string ,void *>("fft_length",(void*)&par->fft_length));
    par_dic.insert(std::pair<std::string ,void *>("window_size",(void*)&par->window_size));
    par_dic.insert(std::pair<std::string ,void *>("step",(void*)&par->step));
    par_dic.insert(std::pair<std::string ,void *>("begin_channel",(void*)&par->begin_channel));
    par_dic.insert(std::pair<std::string ,void *>("compress_channel_num",(void*)&par->compress_channel_num));
    par_dic.insert(std::pair<std::string ,void *>("batch_buffer_size",(void*)&par->batch_buffer_size));
    par_dic.insert(std::pair<std::string ,void *>("tsamp",(void*)&par->tsamp));
        
    par_dic.insert(std::pair<std::string ,void *>("write_head" ,(void*)&par->write_head));
    par_dic.insert(std::pair<std::string ,void *>("write_data" ,(void*)&par->write_data));
    par_dic.insert(std::pair<std::string ,void *>("print_memory" ,(void*)&par->print_memory));
        
    par_dic.insert(std::pair<std::string ,void *>("source_name_in" ,(void*)&par->source_name_in));
    par_dic.insert(std::pair<std::string ,void *>("machine_id_in" ,(void*)&par->machine_id_in));
    par_dic.insert(std::pair<std::string ,void *>("telescope_id_in" ,(void*)&par->telescope_id_in));      par_dic.insert(std::pair<std::string ,void *>("thread_num" ,(void*)&par->thread_num));

    //下列参数可以自行指定数值，先进行默认初始化
    
    //规定程序进行fft变换的长度,程序会把fft_length个采样点作为一个区间进行fft变换
    //由于程序使用了cufft库,该数必须为2的正数次幂
    par->fft_length=131072;
    
    //压缩数据时,滑动窗口的长度,单位为区间,假设window_size=4,fft_length=16,则滑动窗口会跨越16个fft变换区间，共64个采样点
    par->window_size=4096;
    
    //合并的采样点数量(若step=4,则程序会把时间上相邻的4个区间的功率谱相加)
    par->step=4;
    
    //输出结果的起始频率通道序号
    par->begin_channel=par->fft_length/4;
    
    //输出的结果包含的通道数,必须能被8整除(因为每8个相邻通道被压缩到了一个字节)
    par->compress_channel_num=par->fft_length/4;
    
    //规定可用缓冲区的大小,单位为fft区间长度的二倍,若规定fft_length=16,batch_buffer_size=8,则程序会分配16*8*2=256字节的缓冲区(之所以乘以系数2是因为数据有两个通道)
    //该大小由gpu的显存决定,太大会导致程序运行失败,缓冲区越大,程序一次性进行的fft可压缩的数据就越多
    //注意,由于压缩数据的函数一次读取step个数据,为避免内存访问越界,batch_buffer_size必须被step整除
    //此外,为了确保程序可以正确在buffer内移动数据,防止数据出错，要求batch_buffer_size>=2*window_size
    //cufft库的函数会自行分配缓冲区，故batch_buffer_size至多只能为显存的显卡一半
    //不知为何.当cufft一次进行的fft变换数为2的整数次幂时，所用的缓冲空间约为非整数次幂的一半,而程序在压缩数据时,每次会调用cufft库执行batch_buffer_size-window_size个cufft变换，故最好保证batch_buffer_size-window_size是2的整数次幂
    par->batch_buffer_size=8192;
    
    //规定采样点的时间间隔,单位为秒,该数据会被写入文件头
    par->tsamp=1.0/(2500000000/2);
    
    //初始化标志位
    par->write_head=1;
    par->print_memory=0;
    par->write_data=1;
    
    par->machine_id_in=0;
    par->telescope_id_in=0;
    
    //初始化一些只能在手动初始化后自动初始化的参数.需要用特殊的值标记
    par->output_name=NULL;
    par->source_name_in==NULL;
    par->thread_num=0;
    
    //利用命令行选项第二次初始化参数
    
    char* endptr;
    long long int* ll_tmp;
    int* i_tmp;
    float* f_tmp;
    double* d_tmp;
    char ** p_tmp;
    
    while(1)
    {
        option_index=0;
        par_type=getopt_long(argc, argv, "", long_options, &option_index);
        
        if(par_type==-1)
        {
            break;
        }
        switch(par_type)
        {
            case('l'):
                ll_tmp=(long long int*)par_dic[long_options[option_index].name];
                *ll_tmp=strtoll(optarg,&endptr,10);
                printf("Set long long int parameter %s to %lld \n",long_options[option_index].name ,*ll_tmp);
                break;
            case('i'):
                i_tmp=(int*)par_dic[long_options[option_index].name];
                *i_tmp=(int)strtoll(optarg,&endptr,10);
                //std::cout<<"Now optarg is "<<optarg<<std::endl;
                printf("Set int parameter %s to %d \n",long_options[option_index].name ,*i_tmp);
                break;
            case('f'):
                f_tmp=(float*)par_dic[long_options[option_index].name];
                *f_tmp=strtof(optarg,&endptr);
                printf("Set float parameter %s to %f \n",long_options[option_index].name ,*f_tmp);
                break;
            case('d'):
                d_tmp=(double*)par_dic[long_options[option_index].name];
                *d_tmp=strtof(optarg,&endptr);
                printf("Set double parameter %s to %f \n",long_options[option_index].name ,*d_tmp);
                break;
            case('s'):
                p_tmp=(char**)par_dic[long_options[option_index].name];
                *p_tmp=(char*)calloc(strlen(optarg),sizeof(char));
                strcpy(*p_tmp,optarg);
                printf("Set string parameter %s to %s \n",long_options[option_index].name ,*p_tmp);
                break;
            default:
                printf("Fail to find parameter %s\n" ,long_options[option_index].name);
        }
    }
  
    //####所有的可自定义参数确定后,下列参数由程序自行计算####    
    
    //对输入文件表和数据总长度进行初始化
    par->signal_length=0;
    par->file_num=0;
    par->file_list=argv+optind;

    
    for(;(optind+par->file_num)<argc;(par->file_num)++) 
        {
            
            std::cout<<"Input file "<<par->file_num<<" is "<<par->file_list[par->file_num]<<std::endl;
            par->signal_length+=GetFileSize(par->file_list[par->file_num]);
            //printf("par->file_num=%d\n,argc=%d\n",par->file_num,argc);
        }    
        
    //规定文件的输出路径,默认与第一个输入文件相同(把后缀名从.dat修改为.fil)
    //这两个参数也是可手动指定的参数，应该在最开始被自动初始化，但由于getopt库只有在初始化所有”--型“参数后才给出无选项参数，因此只能放在这里初始化
    if(par->output_name==NULL)
        {
            par->output_name=(char*)calloc(strlen(par->file_list[0])+10,sizeof(char));
            regex_t match_exp;
            regmatch_t match_location[1];
            size_t nmatch=1;
            regcomp (&match_exp, ".dat$",0);
            if(regexec(&match_exp,(par->file_list)[0],nmatch ,match_location,0)==REG_NOMATCH)
            {
                printf("Fail to set output_name by input file name ,output_name is set to first file name + .fil\n");
            }
            else
            {
                strncpy(par->output_name,par->file_list[0],match_location[0].rm_so);
                strcat(par->output_name,".fil");
            }
            regfree(&match_exp);
        }
    if(par->source_name_in==NULL)
        {
            par->source_name_in=(char*)calloc(strlen((par->file_list)[0])+10,sizeof(char));
            generate_source_name_in(par->source_name_in,(par->file_list)[0]);
        }

    //thread_num规定了程序在计算fft变换结果对应的功率谱以及压缩数据时使用的线程数
    //根据cuda规定,线程数不得超过1024,程序的压缩算法要求线程数最大为compress_channel_num/8,故取两个约束的最小值
    //也可手动指定
    if(par->thread_num==0)
    {
            par->thread_num=(par->compress_channel_num/8>1024)?1024:(par->compress_channel_num/8);

    }
    
    //计算数据被划分的区间数(即数据需要进行的fft变换的次数),双通道数据需要额外除以2,且要去除无法被step整除的部分
    par->signal_batch=(par->signal_length/par->fft_length/par->step)*par->step;
    
    if(par->signal_batch<par->window_size)
    {
        printf("input file is too small , exit.\n");
        exit(-1);
    }
    
    //缓冲区减去窗口大小是每次压缩的数据量
    par->per_batch=par->batch_buffer_size-par->window_size;
    
    //计算程序需要的循环次数
    par->compress_num=(par->signal_batch-par->window_size)/par->per_batch;
    
    //若signal_batch无法被compress_batch整除,则需要单独处理剩余的数据,计算这部分数据的区间数
    par->remain_batch=par->signal_batch-par->compress_num*par->per_batch-par->window_size;
}

void print_1_channel_parameter_list(pars *par)
{
    std::cout<<"output file = "<<par->output_name<<std::endl;
    std::cout<<"fft length = "<<par->fft_length<<std::endl;
    std::cout<<"signal length = "<<par->signal_length<<std::endl;
    std::cout<<"signal batch = "<<par->signal_batch<<std::endl;
    std::cout<<"window size = "<<par->window_size<<std::endl;
    std::cout<<"step = "<<par->step<<std::endl;
    std::cout<<"batch buffer size = "<<par->batch_buffer_size<<std::endl;
    std::cout<<"per batch ="<<par->per_batch<<std::endl;
    std::cout<<"compress num = "<<par->compress_num<<std::endl;
    std::cout<<"remain batch = "<<par->remain_batch<<std::endl;
    std::cout<<"source_name_in = "<<par->source_name_in<<std::endl;
    std::cout<<"compress num = "<<par->compress_num<<std::endl;
    std::cout<<"remain batch = "<<par->remain_batch<<std::endl;
    std::cout<<"write_head = "<<par->write_head<<std::endl;
    std::cout<<"write_data = "<<par->write_data<<std::endl;
    std::cout<<"print_memory = "<<par->print_memory<<std::endl;
    
}

int fft_half_1_channel(pars *par)
{
    //一些指针,之后为这些指针分配空间
    char *input_char_cpu;
    char *input_char;
    short *input_half;
    double *average_data;
    char* compressed_data;
    float *input_float;
    
    //计算需要为程序分配的内存空间大小(
    
    //根据对batch_buffer_size的要求,per_batch>=window_size
    long long int input_char_size=par->fft_length*par->per_batch;
    long long int input_half_size=(par->fft_length+(long long int)2)*par->batch_buffer_size;
    long long int input_float_size=input_half_size/(long long int)2;
    long long int average_data_size=par->fft_length/(long long int)2;
    long long int compressed_data_size=par->compress_channel_num/(long long int)8*par->per_batch/par->step;
    
    //写入压缩数据的文件流;
    std::ofstream output_stream;
    
    //分配cpu的内存
    cudaMallocHost((void**)&input_char_cpu,sizeof(char)*input_char_size);
    if(par->print_memory)
    {
        //所有内存均为unified memory
        cudaMallocManaged((void**)&input_char,sizeof(char)*input_char_size);
        cudaMallocManaged((void**)&input_half,sizeof(short)*input_half_size);
        cudaMallocManaged((void**)&compressed_data,sizeof(char)*compressed_data_size);
        cudaMallocManaged((void**)&average_data,sizeof(double)*average_data_size);
    }
    else
    {
        cudaMalloc((void**)&input_char,sizeof(char)*input_char_size);
        cudaMalloc((void**)&input_half,sizeof(short)*input_half_size);
        cudaMalloc((void**)&average_data,sizeof(double)*average_data_size);
        cudaMallocManaged((void**)&compressed_data,sizeof(char)*compressed_data_size);
    }
    input_float=(float*)input_half;
    
    //定义两个之后会经常用到的内存偏移量
    long long int half_window_size=(par->fft_length+(long long int)2)*par->window_size;
    long long int float_window_size=(par->fft_length/(long long int)2+(long long int)1)*par->window_size;
    long long int float_add_size=(par->fft_length/(long long int)2+(long long int)1)*par->per_batch;
    //根据以上偏移量计算指针
    
    //第一次读取数据后,后续读入数据的位置
    short *input_half_add=input_half+half_window_size;
    float *input_float_add=input_float+float_window_size;
    
    //每次压缩循环过后,把从该处开始的数据复制到缓冲区头部
    float *input_float_to_head=input_float+float_add_size;
    
    //计算填满一个window需要读取的原始数据大小
    long long int window_char_size=par->fft_length*par->window_size;
    
    //计算填满缓冲区除去window后需要读取的原始数据大小(即每次需要追加读取的数据量)
    long long int add_char_size=par->fft_length*par->per_batch;
    
    //指向fft变换后的数据末尾的倒数第一个区间的指针,便于之后对尾部数据进行反射
    float* input_float_reflect_tail=input_float_add-(par->fft_length/(long long int)2+(long long int)1);
    //指向需要反射的数据的头部的指针
    float* input_float_reflect_head=input_float;
    
    //执行程序
    if(par->write_head)
    {
        //该函数会根据传入的文件名创建一个新文件,写入并关闭
        write_header(par->output_name,par->source_name_in, 8, 21, par->compress_channel_num, 1, 1, 1, 58849., 0.0,  par->tsamp*par->fft_length*par->step, 1.0/par->tsamp/1.0e6*((par->begin_channel+par->compress_channel_num-1)/par->fft_length), -1.0/par->tsamp/1.0e6/par->fft_length, 0.0, 0.0, 0.0, 0.0);
        printf("Succeed writing file header .\n");
    }
    if(par->write_data)
    {
        if(par->write_head)
        {
            //此时写入模式为追加模式(写入文件已由write_header函数创建)
            output_stream.open(par->output_name,std::ios::app|std::ios::binary);
        }
        else
        {
            //此时写入模式为全新模式
            output_stream.open(par->output_name,std::ios::trunc|std::ios::binary);
        }
        if(output_stream.is_open()) 
        {
            std::cout << "Succeed opening output fft file "<<par->output_name<<std::endl;
        }
        else
        {
            std::cout << "Fail to open output fft file "<<par->output_name<<std::endl;
        }
    }
    
    readfile(input_char_cpu,par->file_list,window_char_size);
    printf("copy input char to gpu memory\n");
    cudaMemcpy(input_char,input_char_cpu,window_char_size,cudaMemcpyHostToDevice);
    char2float(input_char,input_half,par->fft_length,par->window_size,par->thread_num);
    
    if(par->print_memory)
    {print_data_half(input_half,0,input_half_size,par->fft_length+2);}
    
    fft_1d_half(input_half,input_half,par->fft_length,par->window_size);
    
    if(par->print_memory)
    {print_data_half(input_half,0,input_half_size,par->fft_length+2);}
    
    complex_modulus_squared(input_half,input_half,par->fft_length/2,par->window_size,par->thread_num);
    
    if(par->print_memory)
    {print_data_float(input_float,0,input_float_size,par->fft_length/2+1);}
    
    //求各通道的初始平均数
    channels_sum(input_float,average_data,par->window_size,par->window_size,(par->fft_length/2+1),par->fft_length/2,par->thread_num);
    
    if(par->print_memory)
    {print_data_double(average_data,0,average_data_size,par->fft_length/2);}

    for(int loop_num=0;loop_num<par->compress_num;loop_num++)
    {
        printf("compress_loop=%d\n",loop_num);
        
        readfile(input_char_cpu,par->file_list,add_char_size);
        
        printf("copy input char to gpu memory\n");
        cudaMemcpy(input_char,input_char_cpu,add_char_size,cudaMemcpyHostToDevice);      
        char2float(input_char,input_half_add,par->fft_length,par->per_batch,par->thread_num);
        
        if(par->print_memory)
        {print_data_half(input_half,0,input_half_size,par->fft_length+2);}
        
        fft_1d_half(input_half_add,input_half_add,par->fft_length,par->per_batch);
        
        if(par->print_memory)
        {print_data_half(input_half,0,input_half_size,par->fft_length+2);}
        
        complex_modulus_squared(input_half_add,input_float_add,par->fft_length/2,par->per_batch,par->thread_num);
        
        if(par->print_memory)
        {print_data_float(input_float,0,input_float_size,par->fft_length/2+1);}
        
        compress(average_data, input_float, input_float_add,compressed_data,(par->fft_length/2+1),par->per_batch/par->step, par->step,par->begin_channel, par->compress_channel_num, par->window_size, par->thread_num);
        
        if(par->print_memory)
        {print_data_binary(compressed_data,0,compressed_data_size,par->compress_channel_num/8);}
        
        if(par->write_data)
        {
            output_stream.write((const char *)compressed_data,compressed_data_size*sizeof(char));
        }
        
        //注意,调用cudaMemcpy时要传入复制的内存大小,因此要乘sizeof
        printf("move fft data in gpu memory\n");
        cudaMemcpy(input_float,input_float_to_head,float_window_size*sizeof(float),cudaMemcpyDeviceToDevice);
        
        if(par->print_memory)
        {print_data_float(input_float,0,input_float_size,par->fft_length/2+1);}
        
    }
    
    if(par->remain_batch>0)
    {
        long long int remain_char_size=par->fft_length*par->remain_batch;
        
        //把余下的数据补到最后一次读取的数据的后面
        printf("begin compress remained batch\n");

        readfile(input_char_cpu,par->file_list,remain_char_size);
        printf("copy input char to gpu memory\n");
        cudaMemcpy(input_char,input_char_cpu,remain_char_size*sizeof(char),cudaMemcpyHostToDevice);

        //read_char_array(input_char+fft_length*per_batch*2,simulate_input_char,fft_length*remain_batch*2);
        //先读取余下的数据
        char2float(input_char,input_half_add,par->fft_length,par->remain_batch,par->thread_num);
        
        if(par->print_memory)
        {print_data_half(input_half,0,input_half_size,par->fft_length+2);}
        
        //对余下的数据进行fft变换等处理(之所以使用per_batch,是为了保证batch数为2的整数次幂,减小cufft库使用的显存)
        fft_1d_half(input_half_add,input_half_add,par->fft_length,par->per_batch);
        
        if(par->print_memory)
        {print_data_half(input_half,0,input_half_size,par->fft_length+2);}
        
        complex_modulus_squared(input_half_add,input_float_add,par->fft_length/2,par->remain_batch,par->thread_num);
        
        if(par->print_memory)
        {print_data_float(input_float,0,input_float_size,par->fft_length/2+1);}
        
        long long int compressed_remain_data_size=par->compress_channel_num/(long long int)8*par->remain_batch/par->step;
        
        compress(average_data, input_float, input_float_add,compressed_data,(par->fft_length/2+1),par->remain_batch/par->step, par->step,par->begin_channel, par->compress_channel_num, par->window_size, par->thread_num);

        if(par->print_memory)
        {print_data_binary(compressed_data,0,compressed_remain_data_size,par->compress_channel_num/8);}
        
        if(par->write_data)
        {
            output_stream.write((const char *)compressed_data,compressed_remain_data_size*sizeof(char));
        }
        
        //input_float_tail和head需要修改
        input_float_reflect_head+=(par->fft_length/2+1)*par->remain_batch;
        input_float_reflect_tail+=(par->fft_length/2+1)*par->remain_batch;
    }
    
    //清除cufft的plan
    fft_1d_half(NULL,NULL,-1,-1);
    
    //余下的window_size大小的数据需要经过反射处理,因此调用另一个压缩函数(head指针和tail指针反向移动)
    
    printf("begin reflect tail data\n");
    compress_reflect(average_data, input_float_reflect_head, input_float_reflect_tail,compressed_data,(par->fft_length/2+1),par->window_size/par->step, par->step,par->begin_channel, par->compress_channel_num, par->window_size, par->thread_num);
    
    long long int compressed_reflect_data_size=par->compress_channel_num/(long long int)8*par->window_size/par->step;
    
    if(par->print_memory)
    {print_data_binary(compressed_data,0,compressed_reflect_data_size,par->compress_channel_num/8);}
    
    if(par->write_data)
    {

        output_stream.write((const char *)compressed_data,compressed_reflect_data_size*sizeof(char));
        //关闭输出文件 
        output_stream.close();
    }
    
    //关闭可能仍处于打开状态的文件
    readfile(input_char_cpu,par->file_list,-1);
    
    //释放内存
    cudaFreeHost(input_char_cpu);
    printf("free memory of input_char_cpu\n");
    cudaFree(input_char);
    printf("free memory of input_char\n");
    cudaFree(input_half);
    printf("free memory of input_half\n");
    cudaFree(average_data);
    printf("free memory of average_data\n");
    cudaFree(compressed_data);
    printf("free memory of compressed_data\n");
    
    return 0;
}

int main(int argc, char *argv[]) {
    
    //初始化程序用到的参数
    pars* program_par= (pars*)calloc(1,sizeof(pars));
    initialize_1_channel_parameter_list (argc ,argv ,program_par);
    print_1_channel_parameter_list(program_par);
    //完成计算
    return fft_half_1_channel(program_par);
}

