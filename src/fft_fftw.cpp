#include "fft_fftw.h"

#include "fft.h"

#ifdef OPENMP
  #include <omp.h>
#endif

Fftw::Fftw()
{
}

void Fftw::init(unsigned width, unsigned height)
{
    m_width = width;
    m_height = height;
    plan_f = NULL;
    plan_fw = NULL;
    plan_fwh = NULL;
    plan_if = NULL;
    plan_ir = NULL;

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

    if(!plan_f){
#ifdef ASYNC
        std::unique_lock<std::mutex> lock(fftw_mut);
        fftw_plan_with_nthreads(2);
#elif OPENMP
#pragma omp critical
        fftw_plan_with_nthreads(omp_get_max_threads());
#endif
#pragma omp critical
        plan_f = fftwf_plan_dft_r2c_2d(m_height, m_width,
                                                  reinterpret_cast<float*>(input.data),
                                                  reinterpret_cast<fftwf_complex*>(complex_result.data),
                                                  FFTW_ESTIMATE);
        fftwf_execute(plan_f);
    }else{fftwf_execute_dft_r2c(plan_f,reinterpret_cast<float*>(input.data),reinterpret_cast<fftwf_complex*>(complex_result.data));}

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

    float *in = reinterpret_cast<float*>(in_all.data);
    fftwf_complex *out = reinterpret_cast<fftwf_complex*>(complex_result.data);
    if(n_channels <= 44){
        if(!plan_fw){
            int rank = 2;
            int n[] = {(int)m_height, (int)m_width};
            int howmany = n_channels;
            int idist = m_height*m_width, odist = m_height*(m_width/2+1);
            int istride = 1, ostride = 1;
            int *inembed = NULL, *onembed = NULL;
#pragma omp critical
#ifdef ASYNC
            std::unique_lock<std::mutex> lock(fftw_mut);
            fftw_plan_with_nthreads(2);
#elif OPENMP
#pragma omp critical
            fftw_plan_with_nthreads(omp_get_max_threads());
#endif
            plan_fw = fftwf_plan_many_dft_r2c(rank, n, howmany,
                                                        in,  inembed, istride, idist,
                                                        out, onembed, ostride, odist,
                                                        FFTW_ESTIMATE);
            fftwf_execute(plan_fw);
        }else{fftwf_execute_dft_r2c(plan_fw,in,out);}
    } else {
        if(!plan_fwh){
            int rank = 2;
            int n[] = {(int)m_height, (int)m_width};
            int howmany = n_channels;
            int idist = m_height*m_width, odist = m_height*(m_width/2+1);
            int istride = 1, ostride = 1;
            int *inembed = NULL, *onembed = NULL;
#pragma omp critical
#ifdef ASYNC
            std::unique_lock<std::mutex> lock(fftw_mut);
            fftw_plan_with_nthreads(2);
#elif OPENMP
#pragma omp critical
            fftw_plan_with_nthreads(omp_get_max_threads());
#endif
            plan_fwh = fftwf_plan_many_dft_r2c(rank, n, howmany,
                                                        in,  inembed, istride, idist,
                                                        out, onembed, ostride, odist,
                                                        FFTW_ESTIMATE);
            fftwf_execute(plan_fwh);
        }else{fftwf_execute_dft_r2c(plan_fwh,in,out);}
    }
    ComplexMat result(m_height, m_width/2 + 1, n_channels);
    for (int i = 0; i < n_channels; ++i)
        result.set_channel(i, complex_result(cv::Rect(0, i*m_height, m_width/2+1, m_height)));
    return result;
}

cv::Mat Fftw::inverse(const ComplexMat &inputf)
{
    int n_channels = inputf.n_channels;
    cv::Mat real_result(m_height, m_width, CV_32FC(n_channels));
    fftwf_complex *in = reinterpret_cast<fftwf_complex*>(inputf.get_p_data());
    float *out = reinterpret_cast<float*>(real_result.data);

    if(n_channels != 1){
        if(!plan_if){
            int rank = 2;
            int n[] = {(int)m_height, (int)m_width};
            int howmany = n_channels;
            int idist = m_height*(m_width/2+1), odist = 1;
            int istride = 1, ostride = n_channels;
            int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

#ifdef ASYNC
            std::unique_lock<std::mutex> lock(fftw_mut);
            fftw_plan_with_nthreads(2);
#elif OPENMP
#pragma omp critical
            fftw_plan_with_nthreads(omp_get_max_threads());
#endif
#pragma omp critical
            plan_if = fftwf_plan_many_dft_c2r(rank, n, howmany,
                                                         in,  inembed, istride, idist,
                                                         out, onembed, ostride, odist,
                                                         FFTW_ESTIMATE);
            fftwf_execute(plan_if);
        }else{fftwf_execute_dft_c2r(plan_if,in,out);}
    }else{
        if(!plan_ir){
            int rank = 2;
            int n[] = {(int)m_height, (int)m_width};
            int howmany = n_channels;
            int idist = m_height*(m_width/2+1), odist = 1;
            int istride = 1, ostride = n_channels;
            int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

#ifdef ASYNC
            std::unique_lock<std::mutex> lock(fftw_mut);
            fftw_plan_with_nthreads(2);
#elif OPENMP
#pragma omp critical
            fftw_plan_with_nthreads(omp_get_max_threads());
#endif
#pragma omp critical
            plan_ir = fftwf_plan_many_dft_c2r(rank, n, howmany,
                                                         in,  inembed, istride, idist,
                                                         out, onembed, ostride, odist,
                                                         FFTW_ESTIMATE);
            fftwf_execute(plan_ir);
    }else{fftwf_execute_dft_c2r(plan_ir,in,out);}
  }

    return real_result/(m_width*m_height);
}

Fftw::~Fftw()
{
  fftwf_destroy_plan(plan_f);
  fftwf_destroy_plan(plan_fw);
  fftwf_destroy_plan(plan_fwh);
  fftwf_destroy_plan(plan_if);
  fftwf_destroy_plan(plan_ir);
}
