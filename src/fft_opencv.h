
#ifndef FFTOPENCV_H
#define FFTOPENCV_H

#include "fft.h"

class FftOpencv : public Fft
{
public:
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales);
    void set_window(const MatDynMem &window);
    template<uint CH, uint S>
    void forward(const MatScales &real_input, ComplexMat<CH,S> &complex_result);
    template<uint CH, uint S>
    void forward_window(MatScaleFeats &patch_feats_in, ComplexMat<CH,S> &complex_result, MatScaleFeats &tmp);
    template<uint CH, uint S>
    void inverse(ComplexMat<CH,S> &complex_input, MatScales &real_result);
    ~FftOpencv();
private:
    cv::Mat m_window;
};

#endif // FFTOPENCV_H
