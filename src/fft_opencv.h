
#ifndef FFTOPENCV_H
#define FFTOPENCV_H

#include "fft.h"

class FftOpencv : public Fft
{
public:
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales, bool big_batch_mode) override;
    void set_window(const cv::Mat &window) override;
    ComplexMat forward(const cv::Mat &input) override;
    ComplexMat forward_window(const std::vector<cv::Mat> &input) override;
    cv::Mat inverse(const ComplexMat &input) override;
    float* inverse_raw(const ComplexMat &input) override;
    ~FftOpencv() override;
private:
    cv::Mat m_window;
};

#endif // FFTOPENCV_H
