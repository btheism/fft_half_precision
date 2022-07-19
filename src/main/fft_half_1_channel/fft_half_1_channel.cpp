#ifdef DEBUG
constexpr bool debug_flag = 1;
#else
constexpr bool debug_flag = 0;
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <fstream>
#include <exception>
#include <cuda_runtime.h>
#include <cuda_macros.hpp>
#include <io_wrapper.hpp>
#include <kernal_wrapper.hpp>
#include <cufft_wrapper.hpp>
#include <yaml2cpp.hpp>
#include<filheader.h>

class stream_info
{
public:
    //数据流的文件名列表,文件名之间以空格分开
    std::vector<std::string> filelist;
    //数据流的长度
    long long int stream_length;
    void set(const std::string & filelist)
    {
        this->filelist=split_names_by_space(filelist);
        if(debug_flag){std::cout<<"create a new stream info"<<std::endl;};
        this->stream_length=GetFilelistSize(this->filelist);
        return;
    };
};

std::ostream & operator<<(std::ostream & output, const stream_info & input)
{
    output<<"\n filelist : \n";
    for(int serial=0 ; serial < input.filelist.size() ; serial++)
    {
        output<<input.filelist[serial]<<"\n";
    };
    output<<"stream length : "<<input.stream_length<<"\n";
    return output;
};



//定义了fft_half_1_channel程序所需的参数表,参数的功能在初始化函数initialize_2_channel_parameter_list中有详细说名
struct fft_half_1_channel_parameter_list 
{
    //文件列表
    stream_info stream;
    long long int signal_length;
    
    //输入数据的类型
    int input_type_flag;
    
    std::string output_name;
    long long int fft_length;
    long long int window_size;
    long long int step;
    long long int begin_channel;
    long long int compress_channel_num;
    long long int batch_buffer_size;
    double tsamp;
    int thread_num;
    
    //以下参数用于写入文件头
    std::string source_name_in;
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

//该函数解析命令行传递给fft_half_1_channel程序的参数列表，设置程序的各项参数
void initialize_1_channel_parameter_list (pars &par, int argc ,char *argv[])
{
    std::string config_file_name;
    //打开配置文件
    if(argc>1)
    {
        config_file_name=std::string(argv[1]);
    }
    else
    {
        config_file_name=GetExeDirPath()+"fft_half_1_channel.yaml";
    }
    std::cout<<"config file is "<<config_file_name<<std::endl;
    yaml_node config(config_file_name);
    std::cout<<"parse config file successfully"<<std::endl;
    
    //规定程序进行fft变换的长度,程序会把fft_length个采样点作为一个区间进行fft变换
    //由于程序使用了cufft库,该数必须为2的正数次幂
    par.fft_length=std::stoll(config("fft_length")());
    std::cout<<"set fft_length to "<<par.fft_length<<std::endl;
    
    //压缩数据时,滑动窗口的长度,单位为区间,假设window_size=4,fft_length=16,则滑动窗口会跨越16个fft变换区间，共64个采样点
    par.window_size=std::stoll(config("window_size")());
    std::cout<<"set window_size to "<<par.window_size<<std::endl;
    
    //合并的采样点数量(若step=4,则程序会把时间上相邻的4个区间的功率谱相加)
    par.step=std::stoll(config("step")());
    std::cout<<"set step to "<<par.step<<std::endl;
    
    //输入数据类型,0表示8位int,1表示16位int
    par.input_type_flag=std::stoi(config("input_type_flag")());
    std::cout<<"set input_type_flag to "<<par.input_type_flag<<std::endl;
    
    //输出结果的起始频率通道序号
    par.begin_channel=std::stoll(config("begin_channel")());
    std::cout<<"set begin_channel to "<<par.begin_channel<<std::endl;
    
    //输出的结果包含的通道数,必须能被8整除(因为每8个相邻通道被压缩到了一个字节)
    par.compress_channel_num=std::stoll(config("compress_channel_num")());
    std::cout<<"set compress_channel_num to "<<par.compress_channel_num<<std::endl;
    
    //规定可用缓冲区的大小,单位为fft区间长度的二倍,若规定fft_length=16,batch_buffer_size=8,则程序会分配16*8*2=256字节的缓冲区(之所以乘以系数2是因为数据有两个通道)
    //该大小由gpu的显存决定,太大会导致程序运行失败,缓冲区越大,程序一次性进行的fft可压缩的数据就越多
    //注意,由于压缩数据的函数一次读取step个数据,为避免内存访问越界,batch_buffer_size必须被step整除
    //此外,为了确保程序可以正确在buffer内移动数据,防止数据出错，要求batch_buffer_size>=2*window_size
    //cufft库的函数会自行分配缓冲区，故batch_buffer_size至多只能为显存的显卡一半
    //不知为何.当cufft一次进行的fft变换数为2的整数次幂时，所用的缓冲空间约为非整数次幂的一半,而程序在压缩数据时,每次会调用cufft库执行batch_buffer_size-window_size个cufft变换，故最好保证batch_buffer_size-window_size是2的整数次幂
    par.batch_buffer_size=std::stoll(config("batch_buffer_size")());
    std::cout<<"set batch_buffer_size to "<<par.batch_buffer_size<<std::endl;
    
    //规定采样点的时间间隔,单位为秒,该数据会被写入文件头
    par.tsamp=std::stod(config("tsamp")());
    std::cout<<"set tsamp to "<<par.tsamp<<std::endl;
    
    //初始化标志位
    par.write_head=std::stoi(config("write_head")());
    par.print_memory=std::stoi(config("print_memory")());
    par.write_data=std::stoi(config("write_data")());
    
    par.machine_id_in=std::stoi(config("machine_id_in")());
    par.telescope_id_in=std::stoi(config("telescope_id_in")());
    
    //初始化写入文件名与头文件中的文件名
    par.output_name=config("output_name")();
    par.source_name_in=config("source_name_in")();
    
    //初始化线程数,线程数若设为0,则自动选择
    par.thread_num=std::stoi(config("thread_num")());
    
    //thread_num规定了程序在计算fft变换结果对应的功率谱以及压缩数据时使用的线程数
    //根据cuda规定,线程数不得超过1024,程序的压缩算法要求线程数最大为compress_channel_num/8,故取两个约束的最小值
    //也可手动指定
    if(par.thread_num==0)
    {
        par.thread_num=(par.compress_channel_num/8>1024)?1024:(par.compress_channel_num/8);
    }
    if(par.thread_num==0)
    {
        throw std::runtime_error("channel num is too small, exit.");
    }
    
    //设定输入流信息
    par.stream.set(config("stream")());
    
    //设定信号长度
    long long int manual_signal_length = std::stoll(config("signal_length")());
    if(manual_signal_length==0)
    {
        par.signal_length=par.stream.stream_length;
    }
    else if(manual_signal_length > par.stream.stream_length)
    {
        throw std::runtime_error("signal_length is larger than actual file stream size");
    }
    else
    {
        par.signal_length=manual_signal_length;
    }
    
    //计算数据被划分的区间数(即数据需要进行的fft变换的次数),双通道数据需要额外除以2,且要去除无法被step整除的部分
    par.signal_batch=(par.signal_length/input_type_size(par.input_type_flag)/par.fft_length/par.step)*par.step;
    
    if(par.signal_batch<par.window_size)
    {
        throw std::runtime_error("input file is too small, exit.");
    }
    //缓冲区减去窗口大小是每次压缩的数据量
    par.per_batch=par.batch_buffer_size-par.window_size;
    
    //计算程序需要的循环次数
    par.compress_num=(par.signal_batch-par.window_size)/par.per_batch;
    
    //若signal_batch无法被compress_batch整除,则需要单独处理剩余的数据,计算这部分数据的区间数
    par.remain_batch=par.signal_batch-par.compress_num*par.per_batch-par.window_size;
    
    std::cout<<"set program parameters successfully"<<std::endl;
}

void print_1_channel_parameter_list(const pars &par)
{
    std::cout<<"output file = "<<par.output_name<<std::endl;
    std::cout<<"fft length = "<<par.fft_length<<std::endl;
    std::cout<<"signal length = "<<par.signal_length<<std::endl;
    std::cout<<"signal batch = "<<par.signal_batch<<std::endl;
    std::cout<<"window size = "<<par.window_size<<std::endl;
    std::cout<<"step = "<<par.step<<std::endl;
    std::cout<<"batch buffer size = "<<par.batch_buffer_size<<std::endl;
    std::cout<<"per batch ="<<par.per_batch<<std::endl;
    std::cout<<"compress num = "<<par.compress_num<<std::endl;
    std::cout<<"source_name_in = "<<par.source_name_in<<std::endl;
    std::cout<<"remain batch = "<<par.remain_batch<<std::endl;
    std::cout<<"write_head = "<<par.write_head<<std::endl;
    std::cout<<"write_data = "<<par.write_data<<std::endl;
    std::cout<<"print_memory = "<<par.print_memory<<std::endl;
    
}

inline void get_new_data(read_stream& input_stream, void *input_int, void *input_int_cpu, short* input_half, long long int fft_length, long long int ask_batch, int thread_num, int input_type_flag, size_t sizeof_input_int ,int print_memory)
{
    long long int batch_half_size=(fft_length+2)*ask_batch;
    std::cout<<"read input int to cpu memory"<<std::endl;
    input_stream.read((char*)input_int_cpu, fft_length*ask_batch*sizeof_input_int);
    if(print_memory)
    {
        print_data_multi_int(input_int_cpu, 0, fft_length*ask_batch, fft_length, input_type_flag);
    }
    std::cout<<"read input int to gpu memory"<<std::endl;
    cudaMemcpy(input_int, input_int_cpu, fft_length*ask_batch*sizeof_input_int,cudaMemcpyHostToDevice);
    int2float(input_int, input_half, fft_length, ask_batch, thread_num ,input_type_flag);
    if(print_memory)
    {
        print_data_half(input_half, 0, batch_half_size, fft_length+2);
        print_data_half_for_copy(input_half, 0, batch_half_size, fft_length+2);
    }
    return;
}

int fft_half_1_channel(const pars &par)
{
    //一些指针,之后为这些指针分配空间
    
    //由于输入数据的类型未定,因此采用void表示
    //这里用short表示half类型,因为half在cufft的库中定义,而主程序并不直接调用cufft,因此未包含该头文件,而short正好与half的大小一致
    void *input_int_cpu;
    void *input_int;
    
    short *input_half;
    double *average_data;
    char* compressed_data;
    float *input_float;
    
    //计算输入数据的类型大小
    size_t sizeof_input_int=input_type_size(par.input_type_flag);
    
    //计算需要为程序分配的内存空间大小
    
    //根据对batch_buffer_size的要求,per_batch>=window_size
    size_t input_int_size=par.fft_length*par.per_batch;
    size_t input_half_size=(par.fft_length+2)*par.batch_buffer_size;
    size_t input_float_size=input_half_size/2;
    size_t average_data_size=par.fft_length/2;
    size_t compressed_data_size=par.compress_channel_num/8*par.per_batch/par.step;
    
    //写入压缩数据的文件流;
    std::ofstream output_stream;
    
    //分配cpu的内存
    gpuErrchk(cudaMallocHost((void**)&input_int_cpu,sizeof_input_int*input_int_size));
    
    if(par.print_memory)
    {
        //所有内存均为unified memory
        gpuErrchk(cudaMallocManaged((void**)&input_int,sizeof_input_int*input_int_size));
        gpuErrchk(cudaMallocManaged((void**)&input_half,sizeof(short)*input_half_size));
        gpuErrchk(cudaMallocManaged((void**)&average_data,sizeof(double)*average_data_size));
        gpuErrchk(cudaMallocManaged((void**)&compressed_data,sizeof(char)*compressed_data_size));
    }
    else
    {
        gpuErrchk(cudaMalloc((void**)&input_int,sizeof_input_int*input_int_size));
        gpuErrchk(cudaMalloc((void**)&input_half,sizeof(short)*input_half_size));
        gpuErrchk(cudaMalloc((void**)&average_data,sizeof(double)*average_data_size));
        gpuErrchk(cudaMallocManaged((void**)&compressed_data,sizeof(char)*compressed_data_size));
    }
    input_float=(float*)input_half;
    
    //定义两个之后会经常用到的内存偏移量
    long long int half_window_size=(par.fft_length+2)*par.window_size;
    long long int float_window_size=(par.fft_length/2+1)*par.window_size;
    long long int float_add_size=(par.fft_length/2+1)*par.per_batch;
    //根据以上偏移量计算指针
    
    //第一次读取数据后,后续读入数据的位置
    short *input_half_add=input_half+half_window_size;
    float *input_float_add=input_float+float_window_size;
    
    //每次压缩循环过后,把从该处开始的数据复制到缓冲区头部
    float *input_float_to_head=input_float+float_add_size;
    
    //指向fft变换后的数据末尾的倒数第一个区间的指针,便于之后对尾部数据进行反射
    float* input_float_reflect_tail=input_float_add-(par.fft_length/2+1);
    //指向需要反射的数据的头部的指针
    float* input_float_reflect_head=input_float;
    
    //执行程序
    if(par.write_head)
    {
        //该函数会根据传入的文件名创建一个新文件,写入并关闭
        write_header(par.output_name.c_str(),par.source_name_in.c_str(), 8, 21, par.compress_channel_num, 1, 1, 1, 58849., 0.0,  par.tsamp*par.fft_length*par.step, 1.0/par.tsamp/1.0e6*((par.begin_channel+par.compress_channel_num-1)/par.fft_length), -1.0/par.tsamp/1.0e6/par.fft_length, 0.0, 0.0, 0.0, 0.0);
        printf("Succeed writing file header .\n");
    }
    if(par.write_data)
    {
        if(par.write_head)
        {
            //此时写入模式为追加模式(写入文件已由write_header函数创建)
            output_stream.open(par.output_name,std::ios::app|std::ios::binary);
        }
        else
        {
            //此时写入模式为全新模式
            output_stream.open(par.output_name,std::ios::trunc|std::ios::binary);
        }
        if(output_stream.is_open()) 
        {
            std::cout << "Succeed opening output fft file "<<par.output_name<<std::endl;
        }
        else
        {
            std::cout << "Fail to open output fft file "<<par.output_name<<std::endl;
        }
    }
    
    //创建输入流
    read_stream input_stream(par.stream.filelist ,par.stream.stream_length);
    
    get_new_data(input_stream, input_int, input_int_cpu, input_half, par.fft_length, par.window_size,  par.thread_num, par.input_type_flag, sizeof_input_int, par.print_memory);
    
    fft_1d_half(input_half,input_half,par.fft_length,par.window_size);
    
    if(par.print_memory)
    {print_data_half(input_half,0,input_half_size,par.fft_length+2);}
    
    complex_modulus_squared(input_half,input_half,par.fft_length/2,par.window_size,par.thread_num);
    
    if(par.print_memory)
    {print_data_float(input_float,0,input_float_size,par.fft_length/2+1);}
    
    //求各通道的初始平均数
    channels_sum(input_float,average_data,par.window_size,par.window_size,(par.fft_length/2+1),par.fft_length/2,par.thread_num);
    
    if(par.print_memory)
    {print_data_double(average_data,0,average_data_size,par.fft_length/2);}

    for(int loop_num=0;loop_num<par.compress_num;loop_num++)
    {
        printf("compress_loop=%d\n",loop_num);
        
        get_new_data(input_stream, input_int, input_int_cpu, input_half_add, par.fft_length, par.per_batch,  par.thread_num, par.input_type_flag, sizeof_input_int, par.print_memory);
        
        fft_1d_half(input_half_add,input_half_add,par.fft_length,par.per_batch);
        
        if(par.print_memory)
        {print_data_half(input_half,0,input_half_size,par.fft_length+2);}
        
        complex_modulus_squared(input_half_add,input_float_add,par.fft_length/2,par.per_batch,par.thread_num);
        
        if(par.print_memory)
        {print_data_float(input_float,0,input_float_size,par.fft_length/2+1);}
        
        compress(average_data, input_float, input_float_add,compressed_data,(par.fft_length/2+1),par.per_batch/par.step, par.step,par.begin_channel, par.compress_channel_num, par.window_size, par.thread_num);
        
        if(par.print_memory)
        {print_data_binary(compressed_data,0,compressed_data_size,par.compress_channel_num/8);}
        
        if(par.write_data)
        {
            output_stream.write((const char *)compressed_data,compressed_data_size*sizeof(char));
        }
        
        //注意,调用cudaMemcpy时要传入复制的内存大小,因此要乘sizeof
        printf("move fft data in gpu memory\n");
        cudaMemcpy(input_float,input_float_to_head,float_window_size*sizeof(float),cudaMemcpyDeviceToDevice);
        
        if(par.print_memory)
        {print_data_float(input_float,0,input_float_size,par.fft_length/2+1);}
        
    }
    
    if(par.remain_batch>0)
    {        
        //把余下的数据补到最后一次读取的数据的后面
        printf("begin compress remained batch\n");

        get_new_data(input_stream, input_int, input_int_cpu, input_half_add, par.fft_length, par.remain_batch,  par.thread_num, par.input_type_flag, sizeof_input_int, par.print_memory);
        
        //对余下的数据进行fft变换等处理(之所以使用per_batch,是为了保证batch数为2的整数次幂,减小cufft库使用的显存)
        fft_1d_half(input_half_add,input_half_add,par.fft_length,par.per_batch);
        
        if(par.print_memory)
        {print_data_half(input_half,0,input_half_size,par.fft_length+2);}
        
        complex_modulus_squared(input_half_add,input_float_add,par.fft_length/2,par.remain_batch,par.thread_num);
        
        if(par.print_memory)
        {print_data_float(input_float,0,input_float_size,par.fft_length/2+1);}
        
        long long int compressed_remain_data_size=par.compress_channel_num/(long long int)8*par.remain_batch/par.step;
        
        compress(average_data, input_float, input_float_add,compressed_data,(par.fft_length/2+1),par.remain_batch/par.step, par.step,par.begin_channel, par.compress_channel_num, par.window_size, par.thread_num);

        if(par.print_memory)
        {print_data_binary(compressed_data,0,compressed_remain_data_size,par.compress_channel_num/8);}
        
        if(par.write_data)
        {
            output_stream.write((const char *)compressed_data,compressed_remain_data_size*sizeof(char));
        }
        
        //input_float_tail和head需要修改
        input_float_reflect_head+=(par.fft_length/2+1)*par.remain_batch;
        input_float_reflect_tail+=(par.fft_length/2+1)*par.remain_batch;
    }
    
    //清除cufft的plan
    fft_1d_half(NULL,NULL,-1,-1);
    
    //余下的window_size大小的数据需要经过反射处理,因此调用另一个压缩函数(head指针和tail指针反向移动)
    
    printf("begin reflect tail data\n");
    compress_reflect(average_data, input_float_reflect_head, input_float_reflect_tail,compressed_data,(par.fft_length/2+1),par.window_size/par.step, par.step,par.begin_channel, par.compress_channel_num, par.window_size, par.thread_num);
    
    long long int compressed_reflect_data_size=par.compress_channel_num/(long long int)8*par.window_size/par.step;
    
    if(par.print_memory)
    {print_data_binary(compressed_data,0,compressed_reflect_data_size,par.compress_channel_num/8);}
    
    if(par.write_data)
    {

        output_stream.write((const char *)compressed_data,compressed_reflect_data_size*sizeof(char));
        //关闭输出文件 
        output_stream.close();
    }
    
    //释放内存
    gpuErrchk(cudaFreeHost(input_int_cpu));
    gpuErrchk(cudaFree(input_int));
    gpuErrchk(cudaFree(input_half));
    gpuErrchk(cudaFree(average_data));
    gpuErrchk(cudaFree(compressed_data));
    
    return 0;
}

int main(int argc, char *argv[]) {
    
    //初始化程序用到的参数
    pars program_par;
    initialize_1_channel_parameter_list (program_par, argc, argv);
    print_1_channel_parameter_list(program_par);
    //完成计算
    return fft_half_1_channel(program_par);
}

