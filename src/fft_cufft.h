
#ifndef FFT_CUDA_H
#define FFT_CUDA_H

#include "fft.h"
#include "cuda/cuda_error_check.cuh"

#if CV_MAJOR_VERSION == 2
  #include <opencv2/gpu/gpu.hpp>
  #define CUDA cv::gpu
#else
  #include "opencv2/opencv.hpp"
  #define CUDA cv::cuda
#endif

#include <cufft.h>
#include <cuda_runtime.h>

struct Scale_vars;

class cuFFT : public Fft
{
public:
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales, bool big_batch_mode) override;
    void set_window(const cv::Mat & window) override;
    void forward(Scale_vars & vars) override;
    void forward_window(Scale_vars & vars) override;
    void inverse(Scale_vars & vars) override;
    ~cuFFT() override;
private:
    cv::Mat m_window;
    unsigned m_width, m_height, m_num_of_feats, m_num_of_scales;
    bool m_big_batch_mode;
    cufftHandle plan_f, plan_f_all_scales, plan_fw, plan_fw_all_scales, plan_i_features,
     plan_i_features_all_scales, plan_i_1ch, plan_i_1ch_all_scales;
    float *data_f = nullptr, *data_f_all_scales = nullptr, *data_fw = nullptr, *data_fw_d = nullptr, *data_fw_all_scales = nullptr, *data_fw_all_scales_d = nullptr, *data_i_features = nullptr, *data_i_features_d = nullptr, *data_i_features_all_scales = nullptr, *data_i_features_all_scales_d = nullptr, *data_i_1ch = nullptr, *data_i_1ch_d = nullptr, *data_i_1ch_all_scales = nullptr, *data_i_1ch_all_scales_d = nullptr;
};

#endif // FFT_CUDA_H
