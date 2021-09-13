SRC_DIR :=./src/
HEADER_DIR :=./src/headers/
BIN_DIR :=./bin/
LIB_DIR :=./bin/lib/
LIB_DIR_SHORT :=./lib
CUFFT_WRAPPER :=cufft_wrapper
KERNEL_WRAPPER :=kernel_wrapper
FIL_HEADER_WRITER :=fil_header_writer
FFT_HALF_2_CHANNEL :=fft_half_2_channel
OTHER_FUNCTION_LIBRARY :=other_function_library

NVCC := nvcc 
NVCC_SHARED_OPTION := --compiler-options="-fPIC -shared" --linker-options="-shared"
NVCC_FIND_LIB := --linker-options="-rpath=$(LIB_DIR_SHORT)"
G++ := g++
GCC := gcc

all: main
$(CUFFT_WRAPPER):
	$(NVCC) $(NVCC_SHARED_OPTION) -o $(LIB_DIR)lib$(CUFFT_WRAPPER).so  $(SRC_DIR)$(CUFFT_WRAPPER)/*.cu
$(KERNEL_WRAPPER):
	$(NVCC) $(NVCC_SHARED_OPTION) -o $(LIB_DIR)lib$(KERNEL_WRAPPER).so  $(SRC_DIR)$(KERNEL_WRAPPER)/*.cu
$(FIL_HEADER_WRITER):
	$(GCC) -fPIC -shared -o $(LIB_DIR)lib$(FIL_HEADER_WRITER).so  $(SRC_DIR)$(FIL_HEADER_WRITER)/*.c
$(OTHER_FUNCTION_LIBRARY):
	$(G++) -fPIC -shared -o $(LIB_DIR)lib$(OTHER_FUNCTION_LIBRARY).so $(SRC_DIR)$(OTHER_FUNCTION_LIBRARY)/*.cpp
librarys: $(CUFFT_WRAPPER) $(KERNEL_WRAPPER) $(FIL_HEADER_WRITER) $(OTHER_FUNCTION_LIBRARY)
main: librarys
	$(NVCC) -o  $(BIN_DIR)$(FFT_HALF_2_CHANNEL)  $(SRC_DIR)$(FFT_HALF_2_CHANNEL)/*.cpp -I $(HEADER_DIR) -L $(LIB_DIR) $(NVCC_FIND_LIB) -l$(CUFFT_WRAPPER) -l$(KERNEL_WRAPPER) -l$(FIL_HEADER_WRITER) -l$(OTHER_FUNCTION_LIBRARY)
clean:
	rm -f -r $(LIB_DIR)/*.so $(BIN_DIR)$(FFT_HALF_2_CHANNEL)
