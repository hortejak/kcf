#include "fft_fftw.h"

#include "fft.h"

#ifdef OPENMP
  #include <omp.h>
#endif

#ifdef ASYNC
#define FFTW_PLAN_WITH_THREADS() fftw_plan_with_nthreads(m_num_threads);
#elif OPENMP
#define FFTW_PLAN_WITH_THREADS() fftw_plan_with_nthreads(omp_get_max_threads());
#else
#define FFTW_PLAN_WITH_THREADS()
#endif

Fftw::Fftw()
    : m_num_threads(1)
{
}

Fftw::Fftw(int num_threads)
    : m_num_threads(num_threads)
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

    {
    cv::Mat real_input = cv::Mat::zeros(m_height, m_width, CV_32FC1);
    ComplexMat complex_result(m_height, m_width / 2 + 1, 1);
    plan_f = fftwf_plan_dft_r2c_2d(m_height, m_width,
                                   reinterpret_cast<float*>(real_input.data),
                                   reinterpret_cast<fftwf_complex*>(complex_result.get_p_data()),
                                   FFTW_ESTIMATE);
    }

    {
        cv::Mat feats_input = cv::Mat::zeros(m_height * 44, m_width, CV_32F);
        ComplexMat complex_result(m_height, m_width / 2 + 1, 44);
        float *in = reinterpret_cast<float*>(feats_input.data);
        fftwf_complex *out = reinterpret_cast<fftwf_complex*>(complex_result.get_p_data());
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = 44;
        int idist = m_height*m_width, odist = m_height*(m_width/2+1);
        int istride = 1, ostride = 1;
        int *inembed = NULL, *onembed = NULL;

        FFTW_PLAN_WITH_THREADS();
        plan_fw = fftwf_plan_many_dft_r2c(rank, n, howmany,
                                          in, inembed, istride, idist,
                                          out, onembed, ostride, odist,
                                          FFTW_ESTIMATE);
    }

    {
        cv::Mat feats_input = cv::Mat::zeros(m_height * 308, m_width, CV_32F);
        ComplexMat complex_result(m_height, m_width / 2 + 1, 308);
        float *in = reinterpret_cast<float*>(feats_input.data);
        fftwf_complex *out = reinterpret_cast<fftwf_complex*>(complex_result.get_p_data());
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = 308;
        int idist = m_height*m_width, odist = m_height*(m_width/2+1);
        int istride = 1, ostride = 1;
        int *inembed = NULL, *onembed = NULL;

        FFTW_PLAN_WITH_THREADS();
        plan_fw_all_scales = fftwf_plan_many_dft_r2c(rank, n, howmany,
                                                     in,  inembed, istride, idist,
                                                     out, onembed, ostride, odist,
                                                     FFTW_ESTIMATE);
    }

    {
        ComplexMat input(m_height,m_width,44);
        cv::Mat real_result = cv::Mat::zeros(m_height, m_width, CV_32FC1);
        fftwf_complex *in = reinterpret_cast<fftwf_complex*>(input.get_p_data());
        float *out = reinterpret_cast<float*>(real_result.data);
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = 44;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = 44;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        FFTW_PLAN_WITH_THREADS();
        plan_i_features = fftwf_plan_many_dft_c2r(rank, n, howmany,
                                                  in,  inembed, istride, idist,
                                                  out, onembed, ostride, odist,
                                                  FFTW_ESTIMATE);
    }

    {
        ComplexMat input(m_height,m_width,1);
        cv::Mat real_result = cv::Mat::zeros(m_height, m_width, CV_32FC1);
        fftwf_complex *in = reinterpret_cast<fftwf_complex*>(input.get_p_data());
        float *out = reinterpret_cast<float*>(real_result.data);
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = 1;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = 1;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        FFTW_PLAN_WITH_THREADS();
        plan_i_1ch = fftwf_plan_many_dft_c2r(rank, n, howmany,
                                             in,  inembed, istride, idist,
                                             out, onembed, ostride, odist,
                                             FFTW_ESTIMATE);
    }
}

void Fftw::set_window(const cv::Mat &window)
{
    m_window = window;
}

ComplexMat Fftw::forward(const cv::Mat &input)
{
    cv::Mat complex_result(m_height, m_width / 2 + 1, CV_32FC2);

    fftwf_execute_dft_r2c(plan_f,reinterpret_cast<float*>(input.data),reinterpret_cast<fftwf_complex*>(complex_result.data));

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
    ComplexMat result(m_height, m_width/2 + 1, n_channels);

    float *in = reinterpret_cast<float*>(in_all.data);
    fftwf_complex *out = reinterpret_cast<fftwf_complex*>(result.get_p_data());

    if (n_channels <= 44)
        fftwf_execute_dft_r2c(plan_fw, in, out);
    else
        fftwf_execute_dft_r2c(plan_fw_all_scales, in, out);

    return result;
}

cv::Mat Fftw::inverse(const ComplexMat &inputf)
{
    int n_channels = inputf.n_channels;
    cv::Mat real_result(m_height, m_width, CV_32FC(n_channels));
    fftwf_complex *in = reinterpret_cast<fftwf_complex*>(inputf.get_p_data());
    float *out = reinterpret_cast<float*>(real_result.data);

    if(n_channels != 1)
        fftwf_execute_dft_c2r(plan_i_features, in, out);
    else
        fftwf_execute_dft_c2r(plan_i_1ch, in, out);

    return real_result/(m_width*m_height);
}

Fftw::~Fftw()
{
  fftwf_destroy_plan(plan_f);
  fftwf_destroy_plan(plan_fw);
  fftwf_destroy_plan(plan_fw_all_scales);
  fftwf_destroy_plan(plan_i_features);
  fftwf_destroy_plan(plan_i_1ch);
}
