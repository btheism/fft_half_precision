#include<other_function_library.hpp>
typedef struct fft_half_2_channel_parameter_list pars;

int main(int argc ,char *argv[])
{
pars* program_par= (pars*)calloc(1,sizeof(pars));
initialize_2_channel_parameter_list (argc ,argv ,program_par);
print_2_channel_parameter_list(program_par);
return 0;
}

