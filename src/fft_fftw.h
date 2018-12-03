#ifndef FFT_FFTW_H
#define FFT_FFTW_H

#include "fft.h"

#ifndef CUFFTW
  #include <fftw3.h>
#else
  #include <cufftw.h>
#endif //CUFFTW

class Fftw : public Fft
{
  public:
    Fftw();
    template <unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales>
    void init();
    void set_window(const MatDynMem &window);
    template <int CH, int S>
    void forward(const MatScales &real_input, ComplexMat<CH, S> &complex_result);
    template <int CH, int S>
    void forward_window(MatScaleFeats &patch_feats_in, ComplexMat<CH, S> &complex_result, MatScaleFeats &tmp);
    template <int CH, int S>
    void inverse(ComplexMat<CH, S> &complex_input, MatScales &real_result);
    ~Fftw();

protected:
    template <uint howmany>
    fftwf_plan create_plan_fwd() const;
    template <uint howmany>
    fftwf_plan create_plan_inv() const;

private:
    cv::Mat m_window;
    fftwf_plan plan_f = 0, plan_fw = 0, plan_i_1ch = 0;
#ifdef BIG_BATCH
    fftwf_plan plan_f_all_scales = 0, plan_fw_all_scales = 0, plan_i_all_scales = 0;
#endif
};

#endif // FFT_FFTW_H
