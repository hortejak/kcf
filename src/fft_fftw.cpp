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

#if defined(ASYNC) || defined(OPENMP)
    fftw_init_threads();
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
#ifdef ASYNC
    std::unique_lock<std::mutex> lock(fftw_mut);
    fftw_plan_with_nthreads(2);
#endif
    fftwf_plan plan = fftwf_plan_dft_r2c_2d(m_height, m_width,
                                            reinterpret_cast<float*>(input.data),
                                            reinterpret_cast<fftwf_complex*>(complex_result.data),
                                            FFTW_ESTIMATE);
#ifdef ASYNC
    lock.unlock();
#endif
    fftwf_execute(plan);
#ifdef ASYNC
    lock.lock();
#endif
    fftwf_destroy_plan(plan);
#ifdef ASYNC
    lock.unlock();
#endif
    return ComplexMat(complex_result);
}

ComplexMat Fftw::forward_window(const std::vector<cv::Mat> &input)
{
    int n_channels = input.size();
    cv::Mat in_all(m_height * n_channels, m_width, CV_32F);
    for (int i = 0; i < n_channels; ++i) {
        cv::Mat in_roi(in_all, cv::Rect(0, i*m_height, m_width, m_height));
        in_roi = input[i].mul(m_window);
    }
    cv::Mat complex_result(n_channels*m_height, m_width/2+1, CV_32FC2);

    int rank = 2;
    int n[] = {(int)m_height, (int)m_width};
    int howmany = n_channels;
    int idist = m_height*m_width, odist = m_height*(m_width/2+1);
    int istride = 1, ostride = 1;
    int *inembed = NULL, *onembed = NULL;
    float *in = reinterpret_cast<float*>(in_all.data);
    fftwf_complex *out = reinterpret_cast<fftwf_complex*>(complex_result.data);
#ifdef ASYNC
    std::unique_lock<std::mutex> lock(fftw_mut);
    fftw_plan_with_nthreads(2);
#endif
    fftwf_plan plan = fftwf_plan_many_dft_r2c(rank, n, howmany,
                                              in,  inembed, istride, idist,
                                              out, onembed, ostride, odist,
                                              FFTW_ESTIMATE);
#ifdef ASYNC
    lock.unlock();
#endif
    fftwf_execute(plan);
#ifdef ASYNC
    lock.lock();
#endif
    fftwf_destroy_plan(plan);
#ifdef ASYNC
    lock.unlock();
#endif

    ComplexMat result(m_height, m_width/2 + 1, n_channels);
    for (int i = 0; i < n_channels; ++i)
        result.set_channel(i, complex_result(cv::Rect(0, i*m_height, m_width/2+1, m_height)));

    return result;
}

cv::Mat Fftw::inverse(const ComplexMat &inputf)
{
    int n_channels = inputf.n_channels;
    cv::Mat real_result(m_height, m_width, CV_32FC(n_channels));
    cv::Mat complex_vconcat = inputf.to_vconcat_mat();

    int rank = 2;
    int n[] = {(int)m_height, (int)m_width};
    int howmany = n_channels;
    int idist = m_height*(m_width/2+1), odist = 1;
    int istride = 1, ostride = n_channels;
#ifndef CUFFTW
    int *inembed = NULL, *onembed = NULL;
#else
    int inembed[2];
    int onembed[2];
    inembed[1] = m_width/2+1, onembed[1] = m_width;
#endif
    fftwf_complex *in = reinterpret_cast<fftwf_complex*>(complex_vconcat.data);
    float *out = reinterpret_cast<float*>(real_result.data);
#if defined(ASYNC) || defined(OPENMP)
    std::unique_lock<std::mutex> lock(fftw_mut);
    fftw_plan_with_nthreads(2);
#endif
    fftwf_plan plan = fftwf_plan_many_dft_c2r(rank, n, howmany,
                                              in,  inembed, istride, idist,
                                              out, onembed, ostride, odist,
                                              FFTW_ESTIMATE);
#ifdef ASYNC
    lock.unlock();
#endif
    fftwf_execute(plan);
#ifdef ASYNC
    lock.lock();
#endif
    fftwf_destroy_plan(plan);
#ifdef ASYNC
    lock.unlock();
#endif
    return real_result/(m_width*m_height);
}

Fftw::~Fftw()
{
}
