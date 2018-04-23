#include "fft_opencv.h"

void FftOpencv::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales, bool big_batch_mode)
{
    (void)width;
    (void)height;
    (void)num_of_feats;
    (void)num_of_scales;
    (void)big_batch_mode;
    std::cout << "FFT: OpenCV" << std::endl;
}

void FftOpencv::set_window(const cv::Mat &window)
{
     m_window = window;
}

ComplexMat FftOpencv::forward(const cv::Mat &input)
{
    cv::Mat complex_result;
    cv::dft(input, complex_result, cv::DFT_COMPLEX_OUTPUT);
    return ComplexMat(complex_result);
}

ComplexMat FftOpencv::forward_raw(float *input, bool all_scales)
{
    ComplexMat dummy;
    return dummy;
}

ComplexMat FftOpencv::forward_window(const std::vector<cv::Mat> &input)
{
    int n_channels = input.size();
    ComplexMat result(input[0].rows, input[0].cols, n_channels);

    for (int i = 0; i < n_channels; ++i) {
        cv::Mat complex_result;
        cv::dft(input[i].mul(m_window), complex_result, cv::DFT_COMPLEX_OUTPUT);
        result.set_channel(i, complex_result);
    }
    return result;
}

cv::Mat FftOpencv::inverse(const ComplexMat &input)
{
    cv::Mat real_result;
    if (input.n_channels == 1) {
        cv::dft(input.to_cv_mat(), real_result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    } else {
        std::vector<cv::Mat> mat_channels = input.to_cv_mat_vector();
        std::vector<cv::Mat> ifft_mats(input.n_channels);
        for (int i = 0; i < input.n_channels; ++i) {
            cv::dft(mat_channels[i], ifft_mats[i], cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        }
        cv::merge(ifft_mats, real_result);
    }
    return real_result;
}

float* FftOpencv::inverse_raw(const ComplexMat &input)
{
    return nullptr;
}

FftOpencv::~FftOpencv()
{

}
