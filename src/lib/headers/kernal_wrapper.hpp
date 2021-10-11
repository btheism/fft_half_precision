void char2float_interlace(void* input_char_void,void* input_half_void,int fft_length,int batch,int thread_num);
void char2float(void* input_char_void,void* input_half_void,int fft_length,int batch,int thread_num);
void complex_modulus_squared_interlace(void *complex_number_void,void *float_number_void, int channel_num, int factor_A, int factor_B, int batch,int thread_num);
void complex_modulus_squared(void *complex_number_void,void *float_number_void, int channel_num,int fft_num,int thread_num);
void channels_average(void *input_data_void,void *average_data_void,int window_size,int interval,int channel_num,int thread_num);


void kernal_add_test(void *input_A_void,void *input_B_void,void *output_void ,long long int input_length);
void kernal_parameter_pass_test(signed char a,short b,int c,long long int d,float e,double f );
void kernal_call_test(void);


