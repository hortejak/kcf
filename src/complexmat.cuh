#pragma once

#include <opencv2/opencv.hpp>

#include "cuda_runtime.h"
#include "cufft.h"

class ComplexMat
{
public:
    int cols;
    int rows;
    int n_channels;
    int n_scales = 1;
    
    ComplexMat() : cols(0), rows(0), n_channels(0) {}
    ComplexMat(int _rows, int _cols, int _n_channels) : cols(_cols), rows(_rows), n_channels(_n_channels)
    {
        cudaHostAlloc((void **)&p_data, n_channels*cols*rows*sizeof(cufftComplex), cudaHostAllocMapped);
        cudaHostGetDevicePointer((void **)&p_data_d,  (void *) p_data , 0);
    }
    
    ComplexMat(int _rows, int _cols, int _n_channels, int _n_scales) : cols(_cols), rows(_rows), n_channels(_n_channels), n_scales(_n_scales)
    {
        cudaHostAlloc((void **)&p_data, n_channels*cols*rows*sizeof(cufftComplex), cudaHostAllocMapped);
        cudaHostGetDevicePointer((void **)&p_data_d,  (void *) p_data , 0);
    }
    
    ComplexMat(ComplexMat &&other)
    {
        cols = other.cols;
        rows = other.rows;
        n_channels = other.n_channels;
        n_scales = other.n_scales;
        p_data = other.p_data;
        
        other.p_data = nullptr;
        other.p_data_d = nullptr;
    }
    
    ~ComplexMat()
    {
        if(p_data != nullptr) cudaFreeHost(p_data);
    }

    void create(int _rows, int _cols, int _n_channels)
    {
        rows = _rows;
        cols = _cols;
        n_channels = _n_channels;
        cudaHostAlloc((void **)&p_data, n_channels*cols*rows*sizeof(cufftComplex), cudaHostAllocMapped);
        cudaHostGetDevicePointer((void **)&p_data_d,  (void *) p_data , 0);
    }

    void create(int _rows, int _cols, int _n_channels, int _n_scales)
    {
        rows = _rows;
        cols = _cols;
        n_channels = _n_channels;
        n_scales = _n_scales;
        cudaHostAlloc((void **)&p_data, n_channels*cols*rows*sizeof(cufftComplex), cudaHostAllocMapped);
        cudaHostGetDevicePointer((void **)&p_data_d,  (void *) p_data , 0);
    }
    // cv::Mat API compatibility
    cv::Size size() { return cv::Size(cols, rows); }
    int channels() { return n_channels; }
    int channels() const { return n_channels; }

    float* sqr_norm() const;
    
    ComplexMat sqr_mag() const;

    ComplexMat conj() const;

    ComplexMat sum_over_channels() const;

    cufftComplex* get_p_data() const;

    //element-wise per channel multiplication, division and addition
    ComplexMat operator*(const ComplexMat & rhs) const;
    ComplexMat operator/(const ComplexMat & rhs) const;
    ComplexMat operator+(const ComplexMat & rhs) const;
    
    //multiplying or adding constant
    ComplexMat operator*(const float & rhs) const;
    ComplexMat operator+(const float & rhs) const;

    //multiplying element-wise multichannel by one channel mats (rhs mat is with one channel)
    ComplexMat mul(const ComplexMat & rhs) const;

    //multiplying element-wise multichannel by one channel mats (rhs mat is with multiple channel)
    ComplexMat mul2(const ComplexMat & rhs) const;
    //text output
    friend std::ostream & operator<<(std::ostream & os, const ComplexMat & mat)
    {
        //for (int i = 0; i < mat.n_channels; ++i){
        for (int i = 0; i < 1; ++i){
            os << "Channel " << i << std::endl;
            for (int j = 0; j < mat.rows; ++j) {
                for (int k = 0; k < 2*mat.cols-2; k+=2)
                    os << "(" << mat.p_data[j*2*mat.cols + k] << "," << mat.p_data[j*2*mat.cols + (k+1)] << ")" << ", ";
                os << "(" << mat.p_data[j*2*mat.cols + 2*mat.cols-2] << "," << mat.p_data[j*2*mat.cols + 2*mat.cols-1] << ")" <<  std::endl;
            }
        }
        return os;
    }
    
    void operator=(ComplexMat & rhs);
    void operator=(ComplexMat && rhs);


private:
    mutable float *p_data = nullptr, *p_data_d = nullptr;
};