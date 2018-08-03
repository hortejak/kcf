
#ifndef FFTOPENCV_H
#define FFTOPENCV_H

#include "fft.h"

struct Scale_vars;

class FftOpencv : public Fft
{
public:
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales, bool big_batch_mode) override;
    void set_window(const cv::Mat & window) override;
    void forward(Scale_vars & vars) override;
    void forward_window(Scale_vars & vars) override;
    void inverse(Scale_vars & vars) override;
    ~FftOpencv() override;
private:
    cv::Mat m_window;
};

#endif // FFTOPENCV_H
