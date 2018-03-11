
#ifndef FFT_FFTW_H
#define FFT_FFTW_H

#include "fft.h"

class Fftw : public Fft
{
public:
    Fftw();
    void init(unsigned width, unsigned height) override;
    void set_window(const cv::Mat &window) override;
    ComplexMat forward(const cv::Mat &input) override;
    ComplexMat forward_window(const std::vector<cv::Mat> &input) override;
    cv::Mat inverse(const ComplexMat &input) override;
    ~Fftw() override;
private:
    unsigned m_width, m_height;
    cv::Mat m_window;
};

#endif // FFT_FFTW_H
