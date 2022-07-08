void fft_1d_half(void* input_half_void, void* output_half_complex_void, long long int fft_length, long long int fft_num); 
void ifft_1d_half(void* input_half_complex_void, void* output_half_void, long long int ifft_length, long long int ifft_num);
long long int test_fft_plan_memory_size( long long int fft_length, long long int fft_num);
