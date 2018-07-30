
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

struct Scale_vars;

class Fftw : public Fft
{
public:
    Fftw();
    Fftw(int num_of_threads);
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales, bool big_batch_mode) override;
    void set_window(const cv::Mat & window) override;
    ComplexMat forward(const cv::Mat & input) override;
    void forward(Scale_vars & vars) override;
    ComplexMat forward_raw(float *input, bool all_scales) override;
    ComplexMat forward_window(const std::vector<cv::Mat> & input) override;
    void forward_window(Scale_vars & vars) override;
    cv::Mat inverse(const ComplexMat & input) override;
    void inverse(Scale_vars & vars) override;
    float* inverse_raw(const ComplexMat & input) override;
    ~Fftw() override;
private:
    unsigned m_num_threads = 6;
    unsigned m_width, m_height, m_num_of_feats, m_num_of_scales;
    bool m_big_batch_mode;
    cv::Mat m_window;
    fftwf_plan plan_f, plan_f_all_scales, plan_fw, plan_fw_all_scales, plan_i_features,
	plan_i_features_all_scales, plan_i_1ch, plan_i_1ch_all_scales;
};

#endif // FFT_FFTW_H
