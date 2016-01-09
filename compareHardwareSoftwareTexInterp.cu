//
//
// MAIN CODE
//
//

#include "allincludes.h"
#include "texHwItp.cuh"
#include "texSwItp.cuh"

/*
  FILL BUFFER USED TO BIND 3D GPU TEXTURE
 */
void fillTextBuffer( ushort * texBuff ) {
    int ix, iy, iz;
    int i1, i2, i3;
    ushort texel;

    for ( ix = 0 ; ix < SIZE_X ; ++ix ) {
        i1 = ix*SIZE_YZ;

        for ( iy = 0 ; iy < SIZE_Y ; ++iy ) {
            i2 = i1 + iy*SIZE_Z;
            texel = 0;

            for ( iz = 0 ; iz < SIZE_Z ; ++iz ) {

                if (iz % 2 == 0) {
                    texel += 20;
                } else if (iz % 3 == 0) {
                    texel += 30;
                } else if (iz % 5 == 0) {
                    texel += 50;
                } else if (iz % 7 == 0) {
                    texel += 70;
                } else {
                    texel += 10;
                }

                i3 = i2 + iz;
                texBuff[i3] = texel;
            }
        }
    }
}

/*
  COMPUTE L2 NORM
 */
float computeL2Norm( float * vect1, float * vect2 ) {
    int ix, iy, iz;
    int i1, i2, i3;
    float errRel = 0.0f;

    for ( ix = 0 ; ix < SIZE_X ; ++ix ) {
        i1 = ix*SIZE_YZ;

        for ( iy = 0 ; iy < SIZE_Y ; ++iy ) {
            i2 = i1 + iy*SIZE_Z;

            for ( iz = 0 ; iz < SIZE_Z ; ++iz ) {
                i3 = i2 + iz;
                errRel += pow((vect1[i3] - vect2[i3]),2);
            }
        }
    }

    return sqrt(errRel);
}

// Hw main
int main(int argc, char **argv) {

    /*****************
      DEVICE BUFFERS
    *****************/
    // allocate memory on the GPU for output buffer
    float * d_devBuff_hw;
    float * d_devBuff_sw;
    CUDA_SAFE_CALL( cudaMalloc( ( void**)&d_devBuff_hw, SIZE_XYZ*sizeof(float) ) );
    CUDA_SAFE_CALL( cudaMalloc( ( void**)&d_devBuff_sw, SIZE_XYZ*sizeof(float) ) );

    /*****************
      HOST BUFFERS
    *****************/
    // output buffer
    float h_devBuff_hw[SIZE_XYZ];
    float h_devBuff_sw[SIZE_XYZ];

    // buffer used to fill 3D texture
    ushort h_texBuff_hw[SIZE_XYZ];
    ushort h_texBuff_sw[SIZE_XYZ];
    fillTextBuffer( &h_texBuff_hw[0] );
    fillTextBuffer( &h_texBuff_sw[0] );

    /*****************
      3D TEXTURE (HW)
    *****************/
    cudaExtent texSize_hw = make_cudaExtent(SIZE_X, SIZE_Y, SIZE_Z);
    cudaArray *cuArray_hw = 0;

    // initialize "Tex3D_hw" with a 3D array "cuArray_hw"
    cudaChannelFormatDesc channelDesc_hw = cudaCreateChannelDesc<ushort>();
    CUDA_SAFE_CALL ( cudaMalloc3DArray( &cuArray_hw, &channelDesc_hw, texSize_hw ) );

    // copy h_texBuff to 3DArray
    cudaMemcpy3DParms copyParams_hw = {0};
    copyParams_hw.srcPtr = make_cudaPitchedPtr( (void*)h_texBuff_hw, SIZE_X*sizeof(ushort), SIZE_X, SIZE_Y ); 
    copyParams_hw.dstArray = cuArray_hw;
    copyParams_hw.extent = texSize_hw;
    copyParams_hw.kind = cudaMemcpyHostToDevice;
    CUDA_SAFE_CALL( cudaMemcpy3D( &copyParams_hw ) );

    // set texture parameters
    Tex3D_hw.addressMode[0] = cudaAddressModeClamp;
    Tex3D_hw.addressMode[1] = cudaAddressModeClamp;
    Tex3D_hw.addressMode[2] = cudaAddressModeClamp;
    Tex3D_hw.filterMode = cudaFilterModeLinear;
    Tex3D_hw.normalized = false;

    // bind texture
    CUDA_SAFE_CALL( cudaBindTextureToArray ( Tex3D_hw, cuArray_hw, channelDesc_hw ) );

    /*****************
      3D TEXTURE (SW)
    *****************/
    cudaExtent texSize_sw = make_cudaExtent(SIZE_X, SIZE_Y, SIZE_Z);
    cudaArray *cuArray_sw = 0;

    // initialize "Tex3D_sw" with a 3D array "cuArray_sw"
    cudaChannelFormatDesc channelDesc_sw = cudaCreateChannelDesc<ushort>();
    CUDA_SAFE_CALL ( cudaMalloc3DArray( &cuArray_sw, &channelDesc_sw, texSize_sw ) );

    // copy h_texBuff to 3DArray
    cudaMemcpy3DParms copyParams_sw = {0};
    copyParams_sw.srcPtr = make_cudaPitchedPtr( (void*)h_texBuff_sw, SIZE_X*sizeof(ushort), SIZE_X, SIZE_Y ); 
    copyParams_sw.dstArray = cuArray_sw;
    copyParams_sw.extent = texSize_sw;
    copyParams_sw.kind = cudaMemcpyHostToDevice;
    CUDA_SAFE_CALL( cudaMemcpy3D( &copyParams_sw ) );

    // set texture parameters
    Tex3D_sw.addressMode[0] = cudaAddressModeClamp;
    Tex3D_sw.addressMode[1] = cudaAddressModeClamp;
    Tex3D_sw.addressMode[2] = cudaAddressModeClamp;
    Tex3D_sw.filterMode = cudaFilterModePoint;
    Tex3D_sw.normalized = false;

    // bind texture
    CUDA_SAFE_CALL( cudaBindTextureToArray ( Tex3D_sw, cuArray_sw, channelDesc_sw ) );

    /*****************
      LAUNCH KERNEL
    *****************/
    testTexHwInterp<<<dim3(BCKX, BCKY, BCKZ), dim3(NBX, NBY, NBZ)>>>(d_devBuff_hw);
    testTexSwInterp<<<dim3(BCKX, BCKY, BCKZ), dim3(NBX, NBY, NBZ)>>>(d_devBuff_sw);

    /*****************
      GET OUTPUT
    *****************/
    // copy the array from the GPU to the CPU
    CUDA_SAFE_CALL( cudaMemcpy( h_devBuff_hw, d_devBuff_hw, SIZE_XYZ*sizeof(float), cudaMemcpyDeviceToHost ) );
    CUDA_SAFE_CALL( cudaMemcpy( h_devBuff_sw, d_devBuff_sw, SIZE_XYZ*sizeof(float), cudaMemcpyDeviceToHost ) );

    /*****************
      COMPARISON
    *****************/
    float l2norm = computeL2Norm( h_devBuff_hw, h_devBuff_sw );
    printf("L2 norm of difference : %f.\n", l2norm);    

    /*****************
      FREE GPU MEMORY
    *****************/
    cudaUnbindTexture( Tex3D_hw );
    cudaUnbindTexture( Tex3D_sw );
    cudaFree( d_devBuff_hw );
    cudaFree( d_devBuff_sw );
    cudaFreeArray( cuArray_hw );
    cudaFreeArray( cuArray_sw );

    return 0;
}



