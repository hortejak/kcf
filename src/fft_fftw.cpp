#include "fft_fftw.h"

#include "fft.h"
#ifndef CUFFTW
  #include <fftw3.h>
#else
  #include <cufftw.h>
#endif //CUFFTW

#if defined(OPENMP)
  #include <omp.h>
#endif

Fftw::Fftw()
{
}

void Fftw::init(unsigned width, unsigned height)
{
    m_width = width;
    m_height = height;

#if defined(OPENMP)
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
#endif //OPENMP

#ifndef CUFFTW
    std::cout << "FFT: FFTW" << std::endl;
#else
    std::cout << "FFT: cuFFTW" << std::endl;
#endif
}

void Fftw::set_window(const cv::Mat &window)
{
    m_window = window;
}

ComplexMat Fftw::forward(const cv::Mat &input)
{
    cv::Mat complex_result(m_height, m_width / 2 + 1, CV_32FC2);
    fftwf_plan plan = fftwf_plan_dft_r2c_2d(m_height, m_width,
                                            reinterpret_cast<float*>(input.data),
                                            reinterpret_cast<fftwf_complex*>(complex_result.data),
                                            FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    return ComplexMat(complex_result);
}

ComplexMat Fftw::forward_window(const std::vector<cv::Mat> &input)
{
    int n_channels = input.size();
    ComplexMat result(input[0].rows, input[0].cols/2 + 1, n_channels);

    for (int i = 0; i < n_channels; ++i) {
        cv::Mat in = input[i].mul(m_window);
        cv::Mat complex_result(result.rows, result.cols, CV_32FC2);
        fftwf_plan plan = fftwf_plan_dft_r2c_2d(m_height, m_width,
                                                reinterpret_cast<float*>(in.data),
                                                reinterpret_cast<fftwf_complex*>(complex_result.data),
                                                FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);

        result.set_channel(i, complex_result);
    }
    return result;
}

cv::Mat Fftw::inverse(const ComplexMat &inputf)
{
    cv::Mat real_result;

    std::vector<cv::Mat> mat_channels = inputf.to_cv_mat_vector();
    std::vector<cv::Mat> ifft_mats(inputf.n_channels);
    for (int i = 0; i < inputf.n_channels; ++i) {
        ifft_mats[i].create(m_height, m_width, CV_32F);
        fftwf_plan plan = fftwf_plan_dft_c2r_2d(m_height, m_width,
                                                reinterpret_cast<fftwf_complex*>(mat_channels[i].data),
                                                reinterpret_cast<float*>(ifft_mats[i].data),
                                                FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    }
    cv::merge(ifft_mats, real_result);

    return real_result/(m_width*m_height);
}

Fftw::~Fftw()
{
}
