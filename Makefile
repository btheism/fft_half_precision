#define source file path
LIB_SRC_DIR :=src/lib/
LIB_HEADER_SRC_DIR :=src/lib/headers/
TEST_SRC_DIR :=src/test/
MAIN_SRC_DIR :=src/

#define binary filw path
LIB_BIN_DIR :=bin/lib/

#use relative path for dynamic library
#LIB_DIR_FOR_MAIN_LINK :=./lib/
#LIB_DIR_FOR_TEST_LINK :=../lib/
#use absolute path for dynamic library
LIB_DIR_FOR_MAIN_LINK :=$(realpath $(LIB_BIN_DIR))
LIB_DIR_FOR_TEST_LINK :=$(realpath $(LIB_BIN_DIR))

TEST_BIN_DIR :=bin/test/
MAIN_BIN_DIR :=bin/

#define some file lists

#lists of headers
HEADER :=$(wildcard $(LLIB_HEADER_SRC_DIR)/*)

#lists of library
LIBRARY :=cufft_wrapper kernel_wrapper other_function_library
#LINK_LIB_OPTION :=$(addprefix -l, $(LIBRARY) ) -lcufft
LINK_LIB_OPTION :=$(addprefix -l, $(LIBRARY) ) -lcufft -lfil_header_writer
LIBRARY_WITH_PATH :=$(addsuffix .so, $(addprefix $(LIB_BIN_DIR)lib, $(LIBRARY) ) )

#lists of test programs , add new test programs to here to compile them
TEST :=test_2_channel_with_reflect test_2_channel_with_reflect_and_loop test_1_channel_with_reflect test_cufft_plan_memory_size test_half_fft de_interlace test_write_header
TEST_WITH_PATH :=$(addprefix $(TEST_BIN_DIR), $(TEST) )

#lists of main program
MAIN := fft_half_2_channel fft_half_1_channel
MAIN_WITH_PATH :=$(addprefix $(MAIN_BIN_DIR), $(MAIN) )

#list of all binary files
ALL_TARGET :=$(LIBRARY_WITH_PATH) $(TEST_WITH_PATH) $(MAIN_WITH_PATH)

#define some compilers
#NVCC_CODE_FLAG :=-gencode arch=compute_75,code=sm_75
NVCC_LIB := nvcc --compiler-options="-fPIC -shared" --linker-options="-shared" $(NVCC_CODE_FLAG) -DPRINT_INFO
NVCC_TEST :=nvcc --linker-options="-rpath=$(LIB_DIR_FOR_TEST_LINK)" -I $(LIB_HEADER_SRC_DIR) -L $(LIB_BIN_DIR) $(LINK_LIB_OPTION) $(NVCC_CODE_FLAG)
NVCC_MAIN :=nvcc --linker-options="-rpath=$(LIB_DIR_FOR_MAIN_LINK)" -I $(LIB_HEADER_SRC_DIR) -L $(LIB_BIN_DIR) $(LINK_LIB_OPTION) $(NVCC_CODE_FLAG) -DWRITE_HEADER -DWRITE_DATA
GCC_LIB :=gcc -fPIC -shared

#cufft_wrapper:
#	$(NVCC_LIB) -o $(LIB_BIN_DIR)lib$(CUFFT_WRAPPER).so  $(LIB_SRC_DIR)$(CUFFT_WRAPPER)/*.cu -lcufft
#fil_header_writer:
#	$(GCC_LIB) -o $(LIB_BIN_DIR)lib$(FIL_HEADER_WRITER).so  $(LIB_SRC_DIR)$(FIL_HEADER_WRITER)/*.c

#define targets , set main as default target
main: $(MAIN_WITH_PATH)
test: $(TEST_WITH_PATH)
#library: $(LIBRARY_WITH_PATH)
library: $(LIBRARY_WITH_PATH) fil_header_writer

#依赖项必须为文件明，若依赖项为虚拟目标，则make总会重新执行编译
$(LIBRARY_WITH_PATH):$(LIB_BIN_DIR)lib%.so:$(LIB_SRC_DIR)%.cu
	$(NVCC_LIB) -o $@ $<

#此处利用“%”匹配$(TEST_WITH_PATH)所列出的目标
$(TEST_WITH_PATH): $(TEST_BIN_DIR)%:$(TEST_SRC_DIR)%.cpp $(LIBRARY_WITH_PATH) $(LIB_BIN_DIR)libfil_header_writer.so
	$(NVCC_TEST) -o $@ $<

$(MAIN_WITH_PATH):$(MAIN_BIN_DIR)%:$(MAIN_SRC_DIR)%.cpp $(LIBRARY_WITH_PATH) $(LIB_BIN_DIR)libfil_header_writer.so
	$(NVCC_MAIN) -o $@ $<

#fil_header_writer:模块有多个源文件，故在此单独编译
fil_header_writer:$(LIB_BIN_DIR)libfil_header_writer.so

$(LIB_BIN_DIR)libfil_header_writer.so:$(LIB_SRC_DIR)fil_header_writer/swap_bytes.c $(LIB_SRC_DIR)fil_header_writer/send_stuff.c $(LIB_SRC_DIR)fil_header_writer/filheader.c
	$(NVCC_LIB) -o $@ $^

#delete all binary files
clean:
	rm -f $(ALL_TARGET) $(LIB_BIN_DIR)libfil_header_writer.so

#generate ignore file list for git
gitignore: .gitignore

.gitignore:
	rm -f .gitignore
	for name in $(ALL_TARGET);do echo $${name}>>.gitignore;done
	echo $(LIB_BIN_DIR)libfil_header_writer.so>>.gitignore

.PHONY: main library test clean gitignore fil_header_writer .gitignore
	
	
