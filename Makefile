#define source file path
LIB_SRC_DIR :=src/lib/
LIB_HEADER_SRC_DIR :=src/lib/headers/
TEST_SRC_DIR :=src/test/
MAIN_SRC_DIR :=src/

#define binary filw path
LIB_BIN_DIR :=bin/lib/
LIB_DIR_FOR_MAIN_LINK :=./lib
LIB_DIR_FOR_TEST_LINK :=../lib
TEST_BIN_DIR :=bin/test/
MAIN_BIN_DIR :=bin/

#define some file lists

#lists of library
LIBRARIES :=cufft_wrapper kernel_wrapper other_function_library
LINK_LIBS_OPTION :=$(addprefix -l, $(LIBRARIES) ) -lcufft
LIBRARIES_WITH_PATH :=$(addsuffix .so, $(addprefix $(LIB_BIN_DIR)lib, $(LIBRARIES) ) )

#lists of test program
TESTS :=test_half_fft
TESTS_WITH_PATH :=$(addprefix $(TEST_BIN_DIR), $(TESTS) )

#lists of main program
MAIN :=fft_half_2_channel
MAIN_WITH_PATH :=$(addprefix $(MAIN_BIN_DIR), $(MAIN) )

#list of all binary files
ALL_TARGETS :=$(LIBRARIES_WITH_PATH) $(TESTS_WITH_PATH) $(MAIN_WITH_PATH)

#define some compilers

NVCC_LIB := nvcc --compiler-options="-fPIC -shared" --linker-options="-shared"
NVCC_TEST :=nvcc --linker-options="-rpath=$(LIB_DIR_FOR_TEST_LINK)" -I $(LIB_HEADER_SRC_DIR) -L $(LIB_BIN_DIR) $(LINK_LIBS_OPTION)
NVCC_MAIN :=nvcc --linker-options="-rpath=$(LIB_DIR_FOR_MAIN_LINK)" -I $(LIB_HEADER_SRC_DIR) -L $(LIB_BIN_DIR) $(LINK_LIBS_OPTION)

GCC_LIB :=gcc -fPIC -shared

#compile libraries
#cufft_wrapper:
#	$(NVCC_LIB) -o $(LIB_BIN_DIR)lib$(CUFFT_WRAPPER).so  $(LIB_SRC_DIR)$(CUFFT_WRAPPER)/*.cu -lcufft
#fil_header_writer:
#	$(GCC_LIB) -o $(LIB_BIN_DIR)lib$(FIL_HEADER_WRITER).so  $(LIB_SRC_DIR)$(FIL_HEADER_WRITER)/*.c

#define targets , set main as default target
main: $(MAIN)
libraries: $(LIBRARIES)
test: $(TESTS)

$(LIBRARIES):
	$(NVCC_LIB) -o $(LIB_BIN_DIR)lib$@.so  $(LIB_SRC_DIR)$@/*

$(TESTS):libraries
	$(NVCC_TEST) -o $(TEST_BIN_DIR)$@  $(TEST_SRC_DIR)$@/*

$(MAIN):libraries
	$(NVCC_MAIN) -o  $(MAIN_BIN_DIR)$@  $(MAIN_SRC_DIR)$@/*

#delete all binary files
clean:
	rm -f $(ALL_TARGETS)

#generate ignore file list for git
gitignore:
	rm -f .gitignore
	for name in $(ALL_TARGETS);do echo $${name}>>.gitignore;done


	
	
