//读取数据的函数
void char2float_interlace(void* input_char_void,void* input_half_void,int fft_length,int batch,int thread_num);
void char2float_interlace_reflect(void* input_char_void,void* input_half_void,int fft_length,int batch,int thread_num);
void char2float(void* input_char_void,void* input_half_void,int fft_length,int batch,int thread_num);
void char2float_reflect(void* input_char_void,void* input_half_void,int fft_length,int batch,int thread_num);


//进行相应变换的参数
void complex_modulus_squared_interlace(void *complex_number_void,void *float_number_void, int channel_num, int factor_A, int factor_B, int batch,int thread_num);
void complex_modulus_squared(void *complex_number_void,void *float_number_void, int channel_num,int batch,int thread_num);

//这两个函数对于单通道和双通道数据均适用,只是传入的batch_interval不同
void channels_sum(void *input_data_void,void *average_data_void,int window_size,double coefficient ,int batch_interval,int channel_num,int thread_num);
void compress(void *average_data_void,float *uncompressed_data_head_void ,float *uncompressed_data_tail_void ,void *compressed_data_void, int batch_interval ,int batch, int step, int channel_num ,int window_size,int thread_num);


//测试内核是否工作的函数
void kernal_add_test(void *input_A_void,void *input_B_void,void *output_void ,long long int input_length);
void kernal_parameter_pass_test(signed char a,short b,int c,long long int d,float e,double f );
void kernal_call_test(void);


