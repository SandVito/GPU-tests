# Cuda directory : you may have to change it
CUDADIR = /usr/local/cuda
# Compilator
NVCC = ${CUDADIR}/bin/nvcc
# Generate code adapted to different architectures
NVOPTS = -gencode arch=compute_20,code=sm_20 \
         -gencode arch=compute_30,code=sm_30 \
         -gencode arch=compute_35,code=sm_35 \
         -gencode arch=compute_35,code=compute_35 \
		 --ptxas-options=-v
NVCCFLAGS = -I. -I/${CUDADIR}/include

CUCODE = compareHardwareSoftwareTexInterp.cu
CUBIN = compareHardwareSoftwareTexInterp

# Target rules
all : build

build : ${CUBIN}

compareHardwareSoftwareTexInterp : ${CUCODE}
	${NVCC} ${NVOPTS} ${NVCCFLAGS} -o ${CUBIN} ${CUCODE}

run :
	./${CUBIN}

clean :
	rm -rf ${CUBIN}

.PHONY: clean
