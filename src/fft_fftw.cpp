#include "fft_fftw.h"

#include "fft.h"

#ifdef OPENMP
  #include <omp.h>
#endif

#if !defined(ASYNC) && !defined(OPENMP) && !defined(CUFFTW)
#define FFTW_PLAN_WITH_THREADS() fftw_plan_with_nthreads(m_num_threads);
#else
#define FFTW_PLAN_WITH_THREADS()
#endif

Fftw::Fftw()
    : m_num_threads(2)
{
}

Fftw::Fftw(int num_threads)
    : m_num_threads(num_threads)
{
}

void Fftw::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales, bool big_batch_mode)
{
    m_width = width;
    m_height = height;
    m_num_of_feats = num_of_feats;
    m_num_of_scales = num_of_scales;
    m_big_batch_mode = big_batch_mode;

#if (!defined(ASYNC) && !defined(CUFFTW))|| defined(OPENMP)
    fftw_init_threads();
#endif //OPENMP

#ifndef CUFFTW
    std::cout << "FFT: FFTW" << std::endl;
#else
    std::cout << "FFT: cuFFTW" << std::endl;
#endif
    //FFT forward one scale
    {
        cv::Mat in_f = cv::Mat::zeros(m_height, m_width, CV_32FC1);
        ComplexMat out_f(m_height, m_width / 2 + 1, 1);
        plan_f = fftwf_plan_dft_r2c_2d(m_height, m_width,
                                       reinterpret_cast<float*>(in_f.data),
                                       reinterpret_cast<fftwf_complex*>(out_f.get_p_data()),
                                       FFTW_PATIENT);
    }
    //FFT forward all scales
    if (m_num_of_scales > 1 && m_big_batch_mode) {
        cv::Mat in_f_all = cv::Mat::zeros(m_height*m_num_of_scales, m_width, CV_32F);
        ComplexMat out_f_all(m_height, m_width / 2 + 1, m_num_of_scales);
        float *in = reinterpret_cast<float*>(in_f_all.data);
        fftwf_complex *out = reinterpret_cast<fftwf_complex*>(out_f_all.get_p_data());
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_scales;
        int idist = m_height*m_width, odist = m_height*(m_width/2+1);
        int istride = 1, ostride = 1;
        int *inembed = NULL, *onembed = NULL;

        FFTW_PLAN_WITH_THREADS();
        plan_f_all_scales = fftwf_plan_many_dft_r2c(rank, n, howmany,
                                                    in, inembed, istride, idist,
                                                    out, onembed, ostride, odist,
                                                    FFTW_PATIENT);
    }
    //FFT forward window one scale
    {
        cv::Mat in_fw = cv::Mat::zeros(m_height * m_num_of_feats, m_width, CV_32F);
        ComplexMat out_fw(m_height, m_width / 2 + 1, m_num_of_feats);
        float *in = reinterpret_cast<float*>(in_fw.data);
        fftwf_complex *out = reinterpret_cast<fftwf_complex*>(out_fw.get_p_data());
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_feats;
        int idist = m_height*m_width, odist = m_height*(m_width/2+1);
        int istride = 1, ostride = 1;
        int *inembed = NULL, *onembed = NULL;

        FFTW_PLAN_WITH_THREADS();
        plan_fw = fftwf_plan_many_dft_r2c(rank, n, howmany,
                                          in, inembed, istride, idist,
                                          out, onembed, ostride, odist,
                                          FFTW_PATIENT);
    }
    //FFT forward window all scales all feats
    if (m_num_of_scales > 1 && m_big_batch_mode) {
        cv::Mat in_all = cv::Mat::zeros(m_height * (m_num_of_scales*m_num_of_feats), m_width, CV_32F);
        ComplexMat out_all(m_height, m_width / 2 + 1, m_num_of_scales*m_num_of_feats);
        float *in = reinterpret_cast<float*>(in_all.data);
        fftwf_complex *out = reinterpret_cast<fftwf_complex*>(out_all.get_p_data());
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_scales*m_num_of_feats;
        int idist = m_height*m_width, odist = m_height*(m_width/2+1);
        int istride = 1, ostride = 1;
        int *inembed = NULL, *onembed = NULL;

        FFTW_PLAN_WITH_THREADS();
        plan_fw_all_scales = fftwf_plan_many_dft_r2c(rank, n, howmany,
                                                     in,  inembed, istride, idist,
                                                     out, onembed, ostride, odist,
                                                     FFTW_PATIENT);
    }
    //FFT inverse one scale
    {
        ComplexMat in_i(m_height,m_width,m_num_of_feats);
        cv::Mat out_i = cv::Mat::zeros(m_height, m_width, CV_32FC(m_num_of_feats));
        fftwf_complex *in = reinterpret_cast<fftwf_complex*>(in_i.get_p_data());
        float *out = reinterpret_cast<float*>(out_i.data);
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_feats;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = m_num_of_feats;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        FFTW_PLAN_WITH_THREADS();
        plan_i_features = fftwf_plan_many_dft_c2r(rank, n, howmany,
                                                  in,  inembed, istride, idist,
                                                  out, onembed, ostride, odist,
                                                  FFTW_PATIENT);
    }
    //FFT inverse all scales
    if (m_num_of_scales > 1 && m_big_batch_mode) {
        ComplexMat in_i_all(m_height,m_width,m_num_of_feats*m_num_of_scales);
        cv::Mat out_i_all = cv::Mat::zeros(m_height, m_width, CV_32FC(m_num_of_feats*m_num_of_scales));
        fftwf_complex *in = reinterpret_cast<fftwf_complex*>(in_i_all.get_p_data());
        float *out = reinterpret_cast<float*>(out_i_all.data);
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_feats*m_num_of_scales;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = m_num_of_feats*m_num_of_scales;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        FFTW_PLAN_WITH_THREADS();
        plan_i_features_all_scales = fftwf_plan_many_dft_c2r(rank, n, howmany,
                                                             in,  inembed, istride, idist,
                                                             out, onembed, ostride, odist,
                                                             FFTW_PATIENT);
    }
    //FFT inver one channel one scale
    if(!m_big_batch_mode) {
        ComplexMat in_i1(m_height,m_width,1);
        cv::Mat out_i1 = cv::Mat::zeros(m_height, m_width, CV_32FC1);
        fftwf_complex *in = reinterpret_cast<fftwf_complex*>(in_i1.get_p_data());
        float *out = reinterpret_cast<float*>(out_i1.data);
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
                                             FFTW_PATIENT);
    }
    //FFT inver one channel all scales
    if (m_num_of_scales > 1 && m_big_batch_mode) {
        ComplexMat in_i1_all(m_height,m_width,m_num_of_scales);
        cv::Mat out_i1_all = cv::Mat::zeros(m_height, m_width, CV_32FC(m_num_of_scales));
        fftwf_complex *in = reinterpret_cast<fftwf_complex*>(in_i1_all.get_p_data());
        float *out = reinterpret_cast<float*>(out_i1_all.data);
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_scales;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = m_num_of_scales;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        FFTW_PLAN_WITH_THREADS();
        plan_i_1ch_all_scales = fftwf_plan_many_dft_c2r(rank, n, howmany,
                                                        in,  inembed, istride, idist,
                                                        out, onembed, ostride, odist,
                                                        FFTW_PATIENT);
    }
}

void Fftw::set_window(const cv::Mat &window)
{
    m_window = window;
}

ComplexMat Fftw::forward(const cv::Mat &input)
{
    ComplexMat complex_result;
    if(m_big_batch_mode && input.rows == (int)(m_height*m_num_of_scales)){
        complex_result.create(m_height, m_width / 2 + 1, m_num_of_scales);
        fftwf_execute_dft_r2c(plan_f_all_scales, reinterpret_cast<float*>(input.data),
                              reinterpret_cast<fftwf_complex*>(complex_result.get_p_data()));
    } else {
        complex_result.create(m_height, m_width / 2 + 1, 1);
        fftwf_execute_dft_r2c(plan_f, reinterpret_cast<float*>(input.data),
                              reinterpret_cast<fftwf_complex*>(complex_result.get_p_data()));
    }
    return complex_result;
}

ComplexMat Fftw::forward_raw(float *input, bool all_scales)
{
    ComplexMat dummy;
    return dummy;
}

ComplexMat Fftw::forward_window(const std::vector<cv::Mat> &input)
{
    int n_channels = input.size();
    cv::Mat in_all(m_height * n_channels, m_width, CV_32F);
    for (int i = 0; i < n_channels; ++i) {
        cv::Mat in_roi(in_all, cv::Rect(0, i*m_height, m_width, m_height));
        in_roi = input[i].mul(m_window);
    }
    ComplexMat result;
    if(n_channels > (int) m_num_of_feats)
        result.create(m_height, m_width/2 + 1, n_channels,m_num_of_scales);
    else
        result.create(m_height, m_width/2 + 1, n_channels);

    float *in = reinterpret_cast<float*>(in_all.data);
    fftwf_complex *out = reinterpret_cast<fftwf_complex*>(result.get_p_data());

    if (n_channels <= (int) m_num_of_feats)
        fftwf_execute_dft_r2c(plan_fw, in, out);
    else
        fftwf_execute_dft_r2c(plan_fw_all_scales, in, out);

    return result;
}

cv::Mat Fftw::inverse(const ComplexMat &input)
{
    int n_channels = input.n_channels;
    cv::Mat real_result(m_height, m_width, CV_32FC(n_channels));
    fftwf_complex *in = reinterpret_cast<fftwf_complex*>(input.get_p_data());
    float *out = reinterpret_cast<float*>(real_result.data);

    if(n_channels == 1)
        fftwf_execute_dft_c2r(plan_i_1ch, in, out);
    else if(m_big_batch_mode && n_channels == (int) m_num_of_scales)
        fftwf_execute_dft_c2r(plan_i_1ch_all_scales, in, out);
    else if(m_big_batch_mode && n_channels == (int) m_num_of_feats * (int) m_num_of_scales)
        fftwf_execute_dft_c2r(plan_i_features_all_scales, in, out);
    else
        fftwf_execute_dft_c2r(plan_i_features, in, out);

    return real_result/(m_width*m_height);
}

float* Fftw::inverse_raw(const ComplexMat &input)
{
    return nullptr;
}

Fftw::~Fftw()
{
    fftwf_destroy_plan(plan_f);
    fftwf_destroy_plan(plan_fw);
    fftwf_destroy_plan(plan_i_features);
    fftwf_destroy_plan(plan_i_1ch);
    
    if (m_big_batch_mode) {
        fftwf_destroy_plan(plan_f_all_scales);
        fftwf_destroy_plan(plan_i_features_all_scales);
        fftwf_destroy_plan(plan_fw_all_scales);
        fftwf_destroy_plan(plan_i_1ch_all_scales);
    }
}
