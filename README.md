# KCF tracker parallel and PREM implementations
This project aims to test multiple implementation of calculating Fast Fourier Transform(FFT) and Inverse Fourier Transform(IFFT) in KFC tracker and tosee how the performance changes depending on the implementation.

C++ implementation of KCF tracker was written by Tomas Vojir [here](https://github.com/vojirt/kcf/blob/master/README.md) and is reimplementation of algorithm presented in "High-Speed Tracking with Kernelized Correlation Filters" paper[1].

### Prerequisites
The code depends on OpenCV 2.4+ library and is build via cmake toolchain. Depending on the implementation selected you have to have installed [FFTW](http://www.fftw.org/), [CUDA](https://developer.nvidia.com/cuda-downloads) or [OpenMP](http://www.openmp.org/).

SSE instructions are used in code which are supported only by x86 architectecture thanks to the [SSE2NEON](https://github.com/jratcliff63367/sse2neon) code now supports both ARM and x86 architecture. 

## Getting Started
Open terminal in the directory with the code:
Using cmake gui:
________________
```
$ mkdir build
$ cmake-gui
```
Then select the directory path and build directory. After which you can choose desired build option. Each option has comment explaining briefly what it does.

Without cmake gui:
___________________
```
$ mkdir build
$ cd build
$ cmake -D"option",-D"option" ..
$ make
```

Where "option" is one of the options from this table:

| Option| Description |
| --- | --- |
| `OPENCV_CUFFT`**WIP** | If OFF CPU implementation using OpenCV implementation of FFT will be used. If ON Nvidia CUFFT implemented in OpenCV will be used. Together with Hostmem from OpenCV. Default value is OFF.|
| `FFTW` | Use FFTW implementation of FFT. If selected together with `OPENCV_CUFFT` then this option will not be used. Default value is OFF.|
| `ASYNC` | Works only if OPENCV_CUFFT is not ON. Will enable C++ async directive. Default value is OFF.|
| `VISULIZE_RESULT` | Check if you want to visulize the result. Default value is OFF. |
| `FFTW` | Use FFTW implementation of FFT. Default value is OFF.|
| `DEBUG_MODE` | Debug terminal output. Default value is OFF.)|
| `DEBUG_MODE_DETAILED` |Additional terminal outputs and screens. Default value is OFF.|

This code compiles into binary **kcf_vot**

./kcf_vot
- using VOT 2014 methodology (http://www.votchallenge.net/)
 - INPUT : expecting two files, images.txt (list of sequence images with absolute path) and
           region.txt with initial bounding box in the first frame in format "top_left_x, top_left_y, width, height" or
           four corner points listed clockwise starting from bottom left corner.
 - OUTPUT : output.txt containing the bounding boxes in the format "top_left_x, top_left_y, width, height"

 

## Author
* **Karafiát Vít**

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
