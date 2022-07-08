#define source file path
LIB_SRC_DIR :=src/lib/
LIB_HEADER_SRC_DIR :=src/lib/headers/
TEST_SRC_DIR :=src/test/
MAIN_SRC_DIR :=src/main/

#define binary file path
LIB_BIN_DIR :=bin/lib/
TEST_BIN_DIR :=bin/test/
MAIN_BIN_DIR :=bin/main/

#use relative path for dynamic library 
#LIB_DIR_FOR_TEST_LINK :=../lib/
#use absolute path for dynamic library
LIB_BIN_DIR_FOR_LINK :=$(realpath $(LIB_BIN_DIR))

#define some file lists

#lists of headers
#HEADERS:=$(wildcard $(LIB_HEADER_SRC_DIR)/*)

#lists of library
NVCC_LIBRARY := cufft_wrapper kernel_wrapper io_wrapper
C_LIBRARY := fil_header_writer
CPP_LIBRARY := yaml2cpp
#libraries that are already installed in system
SYSTEM_NVCC_LIBRARY := cufft 
SYSTEM_C_CPP_LIBRARY := yaml

#all project libraries
ALL_LIBRARY:= $(NVCC_LIBRARY) $(C_LIBRARY) $(CPP_LIBRARY)

#libraries that are used by nvcc
NVCC_LINK_LIBRARY := $(ALL_LIBRARY) $(SYSTEM_NVCC_LIBRARY) $(SYSTEM_C_CPP_LIBRARY)
#libraries that are used by gcc g++
C_CPP_LINK_LIBRARY := $(ALL_LIBRARY) $(SYSTEM_C_CPP_LIBRARY)

#genetate link option
NVCC_LINK_LIB_OPTION :=$(addprefix -l, $(NVCC_LINK_LIBRARY) )
C_CPP_LINK_LIB_OPTION :=$(addprefix -l, $(C_CPP_LINK_LIBRARY) )

NVCC_LIBRARY_WITH_PATH :=$(addsuffix .so, $(addprefix $(LIB_BIN_DIR)lib, $(NVCC_LIBRARY) ) )
C_LIBRARY_WITH_PATH :=$(addsuffix .so, $(addprefix $(LIB_BIN_DIR)lib, $(C_LIBRARY) ) )
CPP_LIBRARY_WITH_PATH :=$(addsuffix .so, $(addprefix $(LIB_BIN_DIR)lib, $(CPP_LIBRARY) ) )
LIBRARY_WITH_PATH:= $(NVCC_LIBRARY_WITH_PATH) $(C_LIBRARY_WITH_PATH) $(CPP_LIBRARY_WITH_PATH)

#lists of test programs , add new test programs to here to compile them
NVCC_TEST := test_half_fft de_interlace test_cufft_plan_memory_size output_arch_code test_1_channel_with_reflect
C_TEST := 
CPP_TEST := test_yaml_parser

NVCC_TEST_WITH_PATH :=$(addprefix $(TEST_BIN_DIR), $(NVCC_TEST) )
C_TEST_WITH_PATH :=$(addprefix $(TEST_BIN_DIR), $(C_TEST) )
CPP_TEST_WITH_PATH :=$(addprefix $(TEST_BIN_DIR), $(CPP_TEST) )
TEST_WITH_PATH :=$(NVCC_TEST_WITH_PATH) $(C_TEST_WITH_PATH) $(CPP_TEST_WITH_PATH)

#lists of main program
NVCC_MAIN := fft_half_2_channel fft_half_1_channel fft_phase_shift
C_MAIN :=
CPP_MAIN :=

NVCC_MAIN_WITH_PATH :=$(addprefix $(MAIN_BIN_DIR), $(NVCC_MAIN) )
C_MAIN_WITH_PATH :=$(addprefix $(MAIN_BIN_DIR), $(C_MAIN) )
CPP_MAIN_WITH_PATH :=$(addprefix $(MAIN_BIN_DIR), $(CPP_MAIN) )
MAIN_WITH_PATH :=$(NVCC_MAIN_WITH_PATH) $(C_MAIN_WITH_PATH) $(CPP_MAIN_WITH_PATH)

#list of all binary files
ALL_TARGETS :=$(LIBRARY_WITH_PATH) $(TEST_WITH_PATH) $(MAIN_WITH_PATH)

#define some compilers

#define universal flag for nvcc compiler
NVCC_CODE_FLAG :=-gencode arch=compute_61,code=sm_61

#define compilers for library
NVCC_LIB := nvcc --compiler-options="-fPIC -shared" --linker-options="-shared" --shared -I $(LIB_HEADER_SRC_DIR) $(NVCC_CODE_FLAG) -DPRINT_INFO -DDEBUG -g
C_LIB := gcc -fPIC -shared -I $(LIB_HEADER_SRC_DIR) -DPRINT_INFO -DDEBUG -g
CPP_LIB :=g++ -fPIC -shared -I $(LIB_HEADER_SRC_DIR) -DPRINT_INFO -g

#define compilers for main and test program
NVCC_PROGRAM :=nvcc --linker-options="-rpath=$(LIB_BIN_DIR_FOR_LINK)" -I $(LIB_HEADER_SRC_DIR) -L $(LIB_BIN_DIR) $(NVCC_CODE_FLAG) -DDEBUG -g
C_PROGRAM :=gcc -Wl,-rpath=$(LIB_BIN_DIR_FOR_LINK) -I $(LIB_HEADER_SRC_DIR) -L $(LIB_BIN_DIR) -DDEBUG -g
CPP_PROGRAM :=g++ -Wl,-rpath=$(LIB_BIN_DIR_FOR_LINK) -I $(LIB_HEADER_SRC_DIR) -L $(LIB_BIN_DIR) -DDEBUG -g

MAKEFILE_JUSTNAME := $(firstword $(MAKEFILE_LIST))
#MAKEFILE_COMPLETE := $(realpath $(MAKEFILE_JUSTNAME))
MAKEFILE_COMPLETE := $(MAKEFILE_JUSTNAME)
MAKE := make -f $(MAKEFILE_COMPLETE)

#set main as default target
main:
	$(MAKE) $(LIBRARY_WITH_PATH)
	$(MAKE) $(MAIN_WITH_PATH)
test:
	$(MAKE) $(LIBRARY_WITH_PATH)
	$(MAKE) $(TEST_WITH_PATH)

library:$(LIBRARY_WITH_PATH)

#需要对依赖项进行二次展开，一次展开后到文件夹，二次展开后到文件
.SECONDEXPANSION:
#这之后定义的目标的依赖项都会被二次展开(似乎除了依赖项之外的其他字段不会被二次展开)
#$$被转义为$，以便用于在二次展开中调用函数
#之所以用二次展开，是因为如果使用$(...)，那么%会被当作常规字符，因此需要在展开%后再展开$(...)

#依赖项必须为文件名，若依赖项为虚拟目标，则make总会重新执行编译
#利用%匹配变量的字段
$(NVCC_LIBRARY_WITH_PATH):$(LIB_BIN_DIR)lib%.so:$$(wildcard $(LIB_SRC_DIR)%/*)
	$(NVCC_LIB) -o $@ $^
$(C_LIBRARY_WITH_PATH):$(LIB_BIN_DIR)lib%.so:$$(wildcard $(LIB_SRC_DIR)%/*)
	$(C_LIB) -o $@ $^
$(CPP_LIBRARY_WITH_PATH):$(LIB_BIN_DIR)lib%.so:$$(wildcard $(LIB_SRC_DIR)%/*)
	$(CPP_LIB) -o $@ $^

$(NVCC_TEST_WITH_PATH):$(TEST_BIN_DIR)%:$$(wildcard $(TEST_SRC_DIR)%/*)
	$(NVCC_PROGRAM) -o $@ $^ $(NVCC_LINK_LIB_OPTION)
$(C_TEST_WITH_PATH):$(TEST_BIN_DIR)%:$$(wildcard $(TEST_SRC_DIR)%/*)
	$(C_PROGRAM) -o $@ $^ $(C_CPP_LINK_LIB_OPTION)
$(CPP_TEST_WITH_PATH):$(TEST_BIN_DIR)%:$$(wildcard $(TEST_SRC_DIR)%/*)
	$(CPP_PROGRAM) -o $@ $^ $(C_CPP_LINK_LIB_OPTION)

$(NVCC_MAIN_WITH_PATH):$(MAIN_BIN_DIR)%:$$(wildcard $(MAIN_SRC_DIR)%/*)
	$(NVCC_PROGRAM) -o $@ $^ $(NVCC_LINK_LIB_OPTION)
$(C_MAIN_WITH_PATH):$(MAIN_BIN_DIR)%:$$(wildcard $(MAIN_SRC_DIR)%/*)
	$(C_PROGRAM) -o $@ $^ $(C_CPP_LINK_LIB_OPTION)
$(CPP_MAIN_WITH_PATH):$(MAIN_BIN_DIR)%:$$(wildcard $(MAIN_SRC_DIR)%/*)
	$(CPP_PROGRAM) -o $@ $^ $(C_CPP_LINK_LIB_OPTION)

#delete all binary files
clean:
	rm -f $(ALL_TARGETS)

#generate ignore file list for git
gitignore: .gitignore

.gitignore:
	rm -f .gitignore
	for name in $(ALL_TARGETS);do echo $${name}>>.gitignore;done

.PHONY: main library test clean gitignore fil_header_writer .gitignore

