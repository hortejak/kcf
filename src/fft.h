
#ifndef FFT_H
#define FFT_H

#include <opencv2/opencv.hpp>
#include "complexmat.hpp"
#include <vector>

class Fft
{
public:
    virtual void init(unsigned width, unsigned height) = 0;
    virtual void set_window(const cv::Mat &window) = 0;
    virtual ComplexMat forward(const cv::Mat &input) = 0;
    virtual ComplexMat forward_window(const std::vector<cv::Mat> &input) = 0;
    virtual cv::Mat inverse(const ComplexMat &input) = 0;
    virtual ~Fft() = 0;
};

#endif // FFT_H
