
#ifndef FFT_CUDA_H
#define FFT_CUDA_H

#include "fft.h"

#if CV_MAJOR_VERSION == 2
  #include <opencv2/gpu/gpu.hpp>
  #define CUDA cv::gpu
#else
  #include "opencv2/opencv.hpp"
  #define CUDA cv::cuda
#endif

#include <cufft.h>
#include <cuda_runtime.h>

class cuFFT : public Fft
{
public:
    cuFFT();
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales, bool big_batch_mode) override;
    void set_window(const cv::Mat &window) override;
    ComplexMat forward(const cv::Mat &input) override;
    ComplexMat forward_window(const std::vector<cv::Mat> &input) override;
    cv::Mat inverse(const ComplexMat &inputf) override;
    ~cuFFT() override;
private:
    cv::Mat m_window;
    unsigned m_width, m_height, m_num_of_feats, m_num_of_scales, m_num_of_streams;
    bool m_big_batch_mode;
    cudaStream_t streams[4];
    cufftHandle plan_f, plan_f_all_scales, plan_fw, plan_fw_all_scales, plan_i_features,
     plan_i_features_all_scales, plan_i_1ch, plan_i_1ch_all_scales;
    float *data_f, *data_f_all_scales, *data_fw, *data_fw_all_scales, *data_i_features, *data_i_features_all_scales, *data_i_1ch, *data_i_1ch_all_scales;
};

#endif // FFT_CUDA_H
