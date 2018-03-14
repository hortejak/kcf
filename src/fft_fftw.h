
#ifndef FFT_FFTW_H
#define FFT_FFTW_H

#include "fft.h"

#if defined(ASYNC)
#include <mutex>
#endif

#ifndef CUFFTW
  #include <fftw3.h>
#else
  #include <cufftw.h>
#endif //CUFFTW

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
    fftwf_plan plan_f, plan_fw, plan_if, plan_ir;
#if defined(ASYNC)
    std::mutex fftw_mut;
#endif
};

#endif // FFT_FFTW_H
