NVCC := nvcc
G++ := g++
SRC_DIR :=./src/
HEADER_DIR :=./src/headers/
BIN_DIR :=./bin/
LIB_DIR :=./bin/lib/
CUFFT_WRAPPER :=cufft_wrapper
KERNEL_WRAPPER :=kernel_wrapper
FIL_HEADER_WRITER :=fil_header_writer
FFT_HALF_2_CHANNEL :=fft_half_2_channel
OTHER_FUNCTION_LIBRARY :=other_function_library
all: main
$(CUFFT_WRAPPER):
	$(NVCC) -fPIC -shared -o $(LIB_DIR)lib$(CUFFT_WRAPPER).so  $(SRC_DIR)$(CUFFT_WRAPPER)/*.cu
$(KERNEL_WRAPPER):
	$(NVCC) -fPIC -shared -o $(LIB_DIR)lib$(KERNEL_WRAPPERS).so  $(SRC_DIR)$(KERNEL_WRAPPER)/*.cu
$(FIL_HEADER_WRITER):
	$(NVCC) -fPIC -shared -o $(LIB_DIR)lib$(FIL_HEADER_WRITER).so  $(SRC_DIR)$(FIL_HEADER_WRITER)/*.c
$(OTHER_FUNCTION_LIBRARY):
	$(G++) -fPIC -shared -o $(LIB_DIR)lib$(OTHER_FUNCTION_LIBRARY).so $(SRC_DIR)$(OTHER_FUNCTION_LIBRARY)/*.cpp
wrappers: $(CUFFT_WRAPPER) $(KERNEL_WRAPPER) $(FIL_HEADER_WRITER)
main: wrappers
	$(NVCC) -o  $(BIN_DIR)$(FFT_HALF_2_CHANNEL)  $(SRC_DIR)$(FFT_HALF_2_CHANNEL)*.cpp -I $(HEADER_DIR) -l$(CUFFT_WRAPPE) -l$(KERNEL_WRAPPER) -l$(FIL_HEADER_WRITER)
clean:
	rm -f -r $(BIN_DIR)
