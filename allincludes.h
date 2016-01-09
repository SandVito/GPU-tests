#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Texture dimensions
#define SIZE_X 64
#define SIZE_Y 64
#define SIZE_Z 64
#define SIZE_YZ  (SIZE_Y*SIZE_Z)
#define SIZE_XYZ (SIZE_X*SIZE_Y*SIZE_Z)
// Number of blocks in each dimension
#define BCKX   2
#define BCKY   4
#define BCKZ   32
// Block size in each dimension
#define NBX   32
#define NBY   16
#define NBZ    2
// Denormalization coefficient
#define DENORM16 65535.0f

#define CUDA_SAFE_CALL( call) do {                                    \
    cudaError err = call;                                             \
    if( cudaSuccess != err) {                                         \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString( err) );       \
        exit(EXIT_FAILURE);                                           \
    } } while (0)


