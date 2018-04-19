
#ifndef FFT_H
#define FFT_H

#include <opencv2/opencv.hpp>
#include <vector>

#ifdef CUFFT
    #include "complexmat.cuh"
#else
    #include "complexmat.hpp"
#endif

class Fft
{
public:
    virtual void init(unsigned width, unsigned height,unsigned num_of_feats, unsigned num_of_scales, bool big_batch_mode) = 0;
    virtual void set_window(const cv::Mat &window) = 0;
    virtual ComplexMat forward(const cv::Mat &input) = 0;
    virtual ComplexMat forward_raw(float *input) = 0;
    virtual ComplexMat forward_window(const std::vector<cv::Mat> &input) = 0;
    virtual cv::Mat inverse(const ComplexMat &input) = 0;
    virtual float* inverse_raw(const ComplexMat &input) = 0;
    virtual ~Fft() = 0;
};

#endif // FFT_H
