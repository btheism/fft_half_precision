#include<string>

#include<filheader.h>

double tsamp=1.0/(2560000000/2);
long long int fft_length=131072;
long long int step=4;


int main(void) {

std::string compressed_data_file="/tmp/cctv";
    
//void write_header(char *outptr_in, char *source_name_in, int machine_id_in, int telescope_id_in, int nchans_in, int nbits_in, int nbeams_in, int ibeam_in, double tstart_in, double start_time_in,  double tsamp_in, double fch1_in, double foff_in, double az_start_in, double za_start_in, double src_raj_in, double src_dej_in)
write_header((char *)compressed_data_file.c_str(), (char *)compressed_data_file.c_str(), 8, 21, 32768, 1, 1, 1, 58849., 0.0,  tsamp*fft_length*step, 625.0, -0.0095367431640625, 0.0, 0.0, 0.0, 0.0); 

return 0;
}
