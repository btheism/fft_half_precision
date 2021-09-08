NVCC := nvcc
SRC_DIR :=./src/
BIN_DIR :=./bin/
LIB_DIR :=./bin/lib/
CUFFT_WRAPPER :=cufft_wrapper
KERNEL_WRAPPER :=kernel_wrapper
FIL_HEADER_WRITER :=fil_header_writer
FFT_HALF_2_CHANNEL :=fft_half_2_channel
all: main
$(CUFFT_WRAPPER):
	$(NVCC) -fPIC -shared -o $(LIB_DIR)lib$(CUFFT_WRAPPER).so -c $(SRC_DIR)$(CUFFT_WRAPPER)*.cu
$(KERNEL_WRAPPER):
	$(NVCC) -fPIC -shared -o $(LIB_DIR)lib$(KERNEL_WRAPPERS).so -c $(SRC_DIR)$(KERNEL_WRAPPER)*.cu
$(FIL_HEADER_WRITER):
	$(NVCC) -fPIC -shared -o $(LIB_DIR)lib$(FIL_HEADER_WRITER).so -c $(SRC_DIR)$(FIL_HEADER_WRITER)*.c
wrappers: $(CUFFT_WRAPPER) $(KERNEL_WRAPPER) $(FIL_HEADER_WRITER)
main: wrappers
	$(NVCC) -o  $(BIN_DIR)$(FFT_HALF_2_CHANNEL) -c $(SRC_DIR)$(FFT_HALF_2_CHANNEL)*.cpp -L $(LIB_DIR) -l$(CUFFT_WRAPPE) -l$(KERNEL_WRAPPER) -l$(FIL_HEADER_WRITER)
clean:
	rm -f -r $(BIN_DIR)
