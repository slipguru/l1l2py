################################################################################
#
# Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

# Location of the CUDA Toolkit

#CUDA_PATH       ?= /usr/local/cuda-7.5
CUDA_PATH       ?= /usr/local/cuda

##############################
# start deprecated interface #
##############################
ifeq ($(x86_64),1)
    $(info WARNING - x86_64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=x86_64 instead)
    TARGET_ARCH ?= x86_64
endif
ifeq ($(ARMv7),1)
    $(info WARNING - ARMv7 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=armv7l instead)
    TARGET_ARCH ?= armv7l
endif
ifeq ($(aarch64),1)
    $(info WARNING - aarch64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=aarch64 instead)
    TARGET_ARCH ?= aarch64
endif
ifeq ($(ppc64le),1)
    $(info WARNING - ppc64le variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=ppc64le instead)
    TARGET_ARCH ?= ppc64le
endif
ifneq ($(GCC),)
    $(info WARNING - GCC variable has been deprecated)
    $(info WARNING - please use HOST_COMPILER=$(GCC) instead)
    HOST_COMPILER ?= $(GCC)
endif
ifneq ($(abi),)
    $(error ERROR - abi variable has been removed)
endif
############################
# end deprecated interface #
############################

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
    TARGET_SIZE := 64
else ifeq ($(TARGET_ARCH),armv7l)
    TARGET_SIZE := 32
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler
ifeq ($(TARGET_OS),darwin)
    ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
        HOST_COMPILER ?= clang++
    endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        ifeq ($(TARGET_OS),linux)
            HOST_COMPILER ?= arm-linux-gnueabihf-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
        else ifeq ($(TARGET_OS),android)
            HOST_COMPILER ?= arm-linux-androideabi-g++
        endif
    else ifeq ($(TARGET_ARCH),aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS), android)
            HOST_COMPILER ?= aarch64-linux-android-g++
        endif
    else ifeq ($(TARGET_ARCH),ppc64le)
        HOST_COMPILER ?= powerpc64le-linux-gnu-g++
    endif
endif
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     := 
LDFLAGS     :=
#LDFLAGS     := -Wl,-soname,test

# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
    CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
    LDFLAGS += -pie
    CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif
    endif
endif

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I../common/inc
#INCLUDES  := -I./common/inc
LIBRARIES :=

LIBRARIES += -lcublas -lcusolver

################################################################################

SAMPLE_ENABLED := 1

# Gencode arguments
ifeq ($(TARGET_ARCH),armv7l)
SMS ?= 20 30 32 35 37 50 52
else
SMS ?= 20 30 35 37 50 52
endif

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################

# Target rules
all: build

build: simpleCublas

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

simpleCublas.o:simpleCublas.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

simpleCublas: simpleCublas.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p ../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
	$(EXEC) cp $@ ../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

csv.o: csv.cpp
	$(NVCC) -o $@ -c $<
	
utils.o: utils.cpp
	$(NVCC) -o $@ -c $<

#algorithms.o: algorithms.cu
	#$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ -c $< $(LIBRARIES)
	
l1l2_regularization.o: l1l2_regularization.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ -c $< $(LIBRARIES)
	
ridge_regression.o: ridge_regression.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ -c $< $(LIBRARIES)
	
test_l1l2C.o: test_l1l2C.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ -c $< $(LIBRARIES)
	
test_ridge_regression.o: test_ridge_regression.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ -c $< $(LIBRARIES)
	
test_ridge_regression: test_ridge_regression.o csv.o ridge_regression.so utils.o
	$(EXEC) $(NVCC) -o $@ $+ $(LIBRARIES)
	
test_l1l2C: test_l1l2C.o csv.o l1l2_regularization.so ridge_regression.so utils.o
	$(EXEC) $(NVCC) -o $@ $+ $(LIBRARIES)
	
test_svd.o: test_svd.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ -c $< $(LIBRARIES)

test_svd: test_svd.o
	$(EXEC) $(NVCC) -o $@ $+ $(LIBRARIES)
	
test_kern.o: test_kern.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ -c $< $(LIBRARIES)

test_kern: test_kern.o
	$(EXEC) $(NVCC) -o $@ $+ $(LIBRARIES)
	
pinv.o: pinv.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ -c $< $(LIBRARIES)
	
test_pinv.o: test_pinv.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
test_pinv: test_pinv.o pinv.o utils.o
	$(EXEC) $(NVCC) -o $@ $+ $(LIBRARIES)
	
test_float: test_float.cpp
	g++ -o $@ $+
	
#test_l1l2C: test_l1l2C.o csv.o algorithms.o utils.o
	#$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

pinv.so: pinv.cu
	$(NVCC) $(INCLUDES) -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -Xcompiler -fPIC -shared -o pinv.so pinv.cu -lcublas -lcusolver
	
l1l2_regularization.so: l1l2_regularization.cu
	$(NVCC) $(INCLUDES) -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -Xcompiler -fPIC -shared -o l1l2_regularization.so l1l2_regularization.cu -lcublas -lcusolver
	
l1l2_path.so: l1l2_path.cu  l1l2_regularization.so
	$(NVCC) $(INCLUDES) -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -Xcompiler -fPIC -shared -o l1l2_path.so l1l2_path.cu l1l2_regularization.so -lcublas -lcusolver
	
ridge_regression.so: ridge_regression.cu pinv.so
	$(NVCC) $(INCLUDES) -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -Xcompiler -fPIC -shared -o ridge_regression.so ridge_regression.cu pinv.so -lcublas -lcusolver
	
#pinv_bridge.o: pinv_bridge.cpp
	#$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ -c $< $(LIBRARIES)

run: build
	$(EXEC) ./simpleCublas

clean:
	rm -f simpleCublas simpleCublas.o
	rm -rf ../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)/simpleCublas
	rm -f csv.o test_l1l2C.o l1l2_regularization.*o ridge_regression.*o test_l1l2C
	rm -f test_svd.o test_svd
	rm -f test_kern.o test_kern
	rm -f test_pinv.o test_pinv pinv.o pinv.so
	rm -f test_ridge_regression test_ridge_regression.*o
	rm -f l1l2_path.*o

clobber: clean
