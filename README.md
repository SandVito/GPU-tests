# GPU tests

Some tests on Nvidia GPUs.

## Summary
This code uses two types of 3D textures. Both of them contains unsigned short (16 bits) values.
The first one will output normalized float. Output values are interpolated by the hardware.
The second one will output unsigned short values. We don't use the hardware interpolation as we 
perform it by reading the 8 needed pixel values.

Then we compare both results. The software interpolation is more precise so we can evaluate
the loss of precision when using hardware interpolation.

## Supported languages
  * C
  * CUDA

## Example

Here are the output you should expect on an Nvidia GTX580.
```
$ make all
$ make run
./compareHardwareSoftwareTexInterp
L2 norm of difference : 0.015639.
$ make clean
```
## Remark

You can also profile both kernels to compare performance between the two approachs.

