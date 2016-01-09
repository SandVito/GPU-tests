//
//
//  SOFTWARE INTERPOLATION OF TEXTURES
//
//

// Cuda texture
texture<ushort,3,cudaReadModeElementType> Tex3D_sw;


/*
  SOFTWARE INTERPOLATION 
 */
__device__ float compSwItp(float x, // coordinate in the width direction
                           float y, // coordinate in the height direction
                           float z) // coordinate in the depth direction
{
    float xb, yb, zb;
    int i0, i1, j0, j1, k0, k1;
    float a, inva, b, invb, c, invc;
    ushort tex000, tex100, tex010, tex110, tex001, tex101, tex011, tex111;
    float cel;

    // compute needed variables
    xb = x - 0.5f;
    yb = y - 0.5f;
    zb = z - 0.5f;
    
    i0 = (int)floorf(xb);
    j0 = (int)floorf(yb);
    k0 = (int)floorf(zb);
    i1 = i0 + 1;
    j1 = j0 + 1;
    k1 = k0 + 1;

    // compute weights
    a = xb - (float)floorf(xb);
    b = yb - (float)floorf(yb);
    c = zb - (float)floorf(zb);
    inva = 1.0f - a;
    invb = 1.0f - b;
    invc = 1.0f - c;

    // read values at pixels center
    tex000 = tex3D(Tex3D_sw, i0, j0, k0);
    tex100 = tex3D(Tex3D_sw, i1, j0, k0);
    tex010 = tex3D(Tex3D_sw, i0, j1, k0);
    tex110 = tex3D(Tex3D_sw, i1, j1, k0);
    tex001 = tex3D(Tex3D_sw, i0, j0, k1);
    tex101 = tex3D(Tex3D_sw, i1, j0, k1);
    tex011 = tex3D(Tex3D_sw, i0, j1, k1);
    tex111 = tex3D(Tex3D_sw, i1, j1, k1);

    // interpolation
    cel = (inva*invb*invc)*(float)tex000
        + (   a*invb*invc)*(float)tex100
        + (inva*   b*invc)*(float)tex010
        + (   a*   b*invc)*(float)tex110
        + (inva*invb*   c)*(float)tex001
        + (   a*invb*   c)*(float)tex101
        + (inva*   b*   c)*(float)tex011
        + (   a*   b*   c)*(float)tex111;

    return cel;
}

/*
  GPU KERNEL
 */
__global__ void testTexSwInterp( float * devTex ) {

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
        devTex[id] = compSwItp(fx,fy,fz);
    } 
}
