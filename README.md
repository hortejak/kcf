# KCF tracker-FFT versions
This project aims to test multiple implementation of calculating Fast Fourier Transform(FFT) and Inverse Fourier Transform(IFFT) in KFC tracker and to see how the performance changes depending on the implementation.

C++ implementation of KCF tracker was written by Tomas Vojir [here](https://github.com/vojirt/kcf/blob/master/README.md) and is reimplementation of algorithm presented in "High-Speed Tracking with Kernelized Correlation Filters" paper[1].

### Prerequisites
The code depends on OpenCV 2.4+ library and build via cmake toolchain. Depending on the version selected you have to have installed [FFTW](http://www.fftw.org/), [CUDA](https://developer.nvidia.com/cuda-downloads) or [OpenMP](http://www.openmp.org/).

SSE instructions were used in the original code and these are only supported on x86 architecture. Thanks to the [SSE2NEON](https://github.com/jratcliff63367/sse2neon) code, we now support both ARM and x86 architecture.

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
| `-DFFT=cuFFTW` | cuFFT version using FFTW interface.|
| `-DFFT=cuFFT` | cuFFT version of FFT. This version also uses pure CUDA for ComplexMat class and Gaussian correlation.|

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

There is also Makefile located in main directory. It will build all possible versions, with all possible combination of additional options (except for CUDA_DEBUG). Each version will have build directory named after it. Build system used is [Ninja](https://ninja-build.org/).

This code compiles into binary **kcf_vot**

./kcf_vot [-v[delay\_ms]]
- using [VOT 2014 methodology](http://www.votchallenge.net/)
- to get dataset used in VOT go [here](http://www.votchallenge.net/vot2016/dataset.html)
 - INPUT : expecting two files, images.txt (list of sequence images with absolute path) and
           region.txt with initial bounding box in the first frame in format "top_left_x, top_left_y, width, height" or
           four corner points listed clockwise starting from bottom left corner.
 - OUTPUT : output.txt containing the bounding boxes in the format "top_left_x, top_left_y, width, height"

 

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
