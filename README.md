# KCF tracker-FFT versions
This project aims to test multiple implementation of calculating Fast Fourier Transform(FFT) and Inverse Fourier Transform(IFFT) in KFC tracker and to see how the performance changes depending on the implementation.

C++ implementation of KCF tracker was written by Tomas Vojir [here](https://github.com/vojirt/kcf/blob/master/README.md) and is reimplementation of algorithm presented in "High-Speed Tracking with Kernelized Correlation Filters" paper[1].

### Prerequisites
The code depends on OpenCV 2.4+ library and build via cmake toolchain. Depending on the version selected you have to have installed [FFTW](http://www.fftw.org/), [CUDA](https://developer.nvidia.com/cuda-downloads) or [OpenMP](http://www.openmp.org/).

SSE instructions were used in the original code and these are only supported on x86 architecture. Thanks to the [SSE2NEON](https://github.com/jratcliff63367/sse2neon) code, we now support both ARM and x86 architecture.

The CuFFT version is set up to run on NVIDIA Jetson TX2. If you want to run it on different architecture, change the `--gpu-architecture sm_62` NVCC flag in **/src/CMakeLists.txt** to your architecture of NVIDIA GPU. To find what SM variation you architecure has please visit http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/.

## Getting Started
Open terminal in the directory with the code:
Using cmake gui:
________________
```
$ mkdir build
$ cmake-gui //Select the directory path and build directory. After which you can choose desired build option. 
            //Each option has comment explaining briefly what it does.
$ make
```
Without cmake gui:
___________________
```
$ mkdir build
$ cd build
$ cmake {OPTIONS} ..
```

The following table shows multiple options how to run cmake to get different version of the tracker. Following table shows FFT options available.

|Option| Description |
| --- | --- |
| `-DFFT=OpenCV` | OpenCV version of FFT.|
| `-DFFT=fftw` | Fftw version of FFT using plan many and new execute functions.|
| `-DFFT=cuFFTW` | CuFFT version using FFTW interface.|
| `-DFFT=cuFFT` | CuFFT version of FFT. This version also uses pure CUDA for ComplexMat class and Gaussian correlation.|

With all of these FFT version aditional options can be added:

|Option| Description |
| --- | --- |
| `-DASYNC=ON` | Adds C++ async directive. It is used to compute response maps for all scales in parallel. Doesn't work with big batch mode.|
| `-DOPENMP=ON` | Fftw version of FFT using plan many and new execute functions. If not selected with big batch mode, it also is used to compute all reposne maps for all scales in parallel. If used with big batch mode it parallelize extraction of feaures for all scales and selection of scale with highest reponse.|
| `-DBIG_BATCH=ON` | Enables big batch mode, which creates one big matrix from all scales and computes them together. This mode doesn't work for OpenCV FFT.|
| `-DCUDA_DEBUG=ON` | This mode adds CUDA error checking for all kernels and CUDA runtime libraries. Only works with cuFFT version.|

Finally call make:
```
$ make
```

There is also Makefile located in main directory. It will build all possible versions, with all possible combination of additional options (except for CUDA_DEBUG). Each version will have build directory named after it. Default build system used by this Makefile is [Ninja](https://ninja-build.org/). If you want to build only specific version use `make [version]`, where inplace of `version` use one the following:

|Version| Description |
| --- | --- |
| `opencvfft-st` | OpenCV FFT single threaded version|
| `opencvfft-async` | OpenCV FFT multithreaded version using C++ async directive|
| `fftw` | FFTW FFT single threaded version|
| `fftw_openmp` | FFTW FFT multithreaded version using OpenMP|
| `fftw_async` | FFTW FFT multithreaded version using C++ async directive|
| `fftw_big` | FFTW FFT single threaded version using Big batch mode|
| `fftw_big_openmp` | FFTW FFT multithreaded version using Big batch mode|
| `cufftw` | CuFFTW version|
| `cufftw_big` | CuFFTW version using Big batch mode|
| `cufftw_big_openmp` | CuFFTW version using Big batch mode and OpenMP multithreading to get maximal response|
| `cufft` | CuFFT FFT version|
| `cufft_big` | CuFFT FFT version using Big batch mode|
| `cufft_big_openmp` | CuFFT FFT version using Big batch mode and OpenMP multithreading to get maximal response|


This code compiles into binary **kcf_vot**

## Usage
`./kcf_vot [options] <path/to/region.txt or groundtruth.txt> <path/to/images.txt> [path/to/output.txt]`
- using [VOT 2014 methodology](http://www.votchallenge.net/)
- to get dataset used in VOT go [here](http://www.votchallenge.net/vot2016/dataset.html)
 - INPUT : expecting two files, images.txt (list of sequence images with absolute path) and
           region.txt with initial bounding box in the first frame in format "top_left_x, top_left_y, width, height" or
           four corner points listed clockwise starting from bottom left corner.
 - OUTPUT : output.txt containing the bounding boxes in the format "top_left_x, top_left_y, width, height"
 -There are also multiple additional terminal `[options]`, which you can use:
|Option| Description |
| --- | --- |
| `--visualize | -v [delay_ms]` | Visualize the output with specified delay. If the delay is set to 0 the output file will stay until the user presses any button.|
| `---output | -o <outout.txt>` | Specifies output file.|
| `--debug | -d` | Additional debugging output.|
| `--fit | -f [dimensions]` | Specifies dimension to which the extracted patch should be scaled. It should be divisible by 4, which is the size of the HOG cell. You can either input single dimension, which will result in in the other both dimensions being the same. Or both dimensions in the form of: `[X dimension]X[Y dimension]`.|

## Author
* **Karafiát Vít**, **Sojka Michal**

## References

[1] João F. Henriques, Rui Caseiro, Pedro Martins, Jorge Batista, “High-Speed Tracking with Kernelized Correlation Filters“,
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015
_____________________________________
Copyright (c) 2014, Tomáš Vojíř

Permission to use, copy, modify, and distribute this software for research
purposes is hereby granted, provided that the above copyright notice and
this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
