//
//
//  HARWDARE INTERPOLATION OF TEXTURES
//
//

// Cuda texture
texture<ushort,3,cudaReadModeNormalizedFloat> Tex3D_hw;

/*
  GPU KERNEL
 */
__global__ void testTexHwInterp( float * devTex ) {

    int ix, iy, iz;
    int id;
    float fx, fy, fz;

    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;
    iz = blockIdx.z * blockDim.z + threadIdx.z;

    id = ix*SIZE_YZ + iy*SIZE_Z + iz;

    if ( id < SIZE_XYZ ) {
        fx = (float)ix+0.3;
        fy = (float)iy+0.3;
        fz = (float)iz+0.3;
  
        // handle the data at this index
        devTex[id] = tex3D(Tex3D_hw,fx,fy,fz)*DENORM16;
    }
}

