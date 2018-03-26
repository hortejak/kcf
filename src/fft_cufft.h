#ifndef FFT_CUDA_H
#define FFT_CUDA_H

#include "fft.h"

#include <cufft.h>

class cuFFT : public Fft
{
public:
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales) override;
    void set_window(const cv::Mat &window) override;
    ComplexMat forward(const cv::Mat &input) override;
    ComplexMat forward_window(const std::vector<cv::Mat> &input) override;
    cv::Mat inverse(const ComplexMat &inputf) override;
    ~cuFFT() override;
private:
    cv::Mat m_window;
    unsigned m_width, m_height, m_num_of_feats,m_num_of_scales;
};

#endif // FFT_CUDA_H
