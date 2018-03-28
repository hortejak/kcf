#include "fft_cufft.h"

cuFFT::cuFFT(): m_num_of_streams(4)
{}

void cuFFT::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales)
{
    m_width = width;
    m_height = height;
    m_num_of_feats = num_of_feats;
    m_num_of_scales = num_of_scales;

    std::cout << "FFT: cuFFT" << std::endl;

    cudaSetDeviceFlags(cudaDeviceMapHost);

    for (unsigned i = 0; i < m_num_of_streams; i++) cudaStreamCreate(&streams[i]);

    //FFT forward one scale
    {
    cufftPlan2d(&plan_f, m_height, m_width, CUFFT_C2R);
    }
    //FFT forward all scales
    if(m_num_of_scales > 1)
    {
    int rank = 2;
    int n[] = {(int)m_height, (int)m_width};
    int howmany = m_num_of_scales;
    int idist = m_height*m_width, odist = m_height*(m_width/2+1);
    int istride = 1, ostride = 1;
    int *inembed = NULL, *onembed = NULL;

    cufftPlanMany(&plan_f_all_scales, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_R2C, howmany);
    }
    //FFT forward window one scale
    {
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_feats;
        int idist = m_height*m_width, odist = m_height*(m_width/2+1);
        int istride = 1, ostride = 1;
        int *inembed = NULL, *onembed = NULL;

        cufftPlanMany(&plan_fw, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_R2C, howmany);
    }
    //FFT forward window all scales all feats
    if(m_num_of_scales > 1)
    {
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_scales*m_num_of_feats;
        int idist = m_height*m_width, odist = m_height*(m_width/2+1);
        int istride = 1, ostride = 1;
        int *inembed = NULL, *onembed = NULL;

        cufftPlanMany(&plan_fw_all_scales, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_R2C, howmany);
    }
    //FFT inverse one scale
    {
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_feats;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = m_num_of_feats;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        cufftPlanMany(&plan_i_features, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_C2R, howmany);
    }
    //FFT inverse all scales
    if(m_num_of_scales > 1)
    {
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_feats*m_num_of_scales;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = m_num_of_feats*m_num_of_scales;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        cufftPlanMany(&plan_i_features_all_scales, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_C2R, howmany);
    }
    //FFT inver one channel one scale
    {
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = 1;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = 1;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        cufftPlanMany(&plan_i_1ch, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_C2R, howmany);
    }
    //FFT inver one channel all scales
    if(m_num_of_scales > 1)
    {
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_scales;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = m_num_of_scales;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        cufftPlanMany(&plan_i_1ch_all_scales, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_C2R, howmany);
    }
}

void cuFFT::set_window(const cv::Mat &window)
{
     m_window = window;
}

ComplexMat cuFFT::forward(const cv::Mat &input)
{
    CUDA::GpuMat input_d(input);
    ComplexMat complex_result;
    if(input.rows == (int)(m_height*m_num_of_scales)){
        complex_result.create(m_height, m_width / 2 + 1, m_num_of_scales);
        cufftExecR2C(plan_f_all_scales, reinterpret_cast<float*>(input.data),
                                reinterpret_cast<cufftComplex*>(complex_result.get_p_data()));
    } else {
        complex_result.create(m_height, m_width / 2 + 1, 1);
        cufftExecR2C(plan_f, reinterpret_cast<float*>(input.data),
                                reinterpret_cast<cufftComplex*>(complex_result.get_p_data()));
    }
    return complex_result;
}

ComplexMat cuFFT::forward_window(const std::vector<cv::Mat> &input)
{
    int n_channels = input.size();
    cv::Mat in_all(m_height * n_channels, m_width, CV_32F);
    for (int i = 0; i < n_channels; ++i) {
        cv::Mat in_roi(in_all, cv::Rect(0, i*m_height, m_width, m_height));
        in_roi = input[i].mul(m_window);
    }
    CUDA::GpuMat in_all_d(in_all);
    ComplexMat result;
    if(n_channels > (int) m_num_of_feats)
        result.create(m_height, m_width/2 + 1, n_channels,m_num_of_scales);
    else
        result.create(m_height, m_width/2 + 1, n_channels);

    float *in = reinterpret_cast<float*>(in_all.data);
    cufftComplex *out = reinterpret_cast<cufftComplex*>(result.get_p_data());

    if (n_channels <= (int) m_num_of_feats)
        cufftExecR2C(plan_fw, in, out);
    else
       cufftExecR2C(plan_fw_all_scales, in, out);

    return result;
}

cv::Mat cuFFT::inverse(const ComplexMat &inputf)
{
    int n_channels = inputf.n_channels;
    cv::Mat real_result(m_height, m_width, CV_32FC(n_channels));
    cufftComplex *in = reinterpret_cast<cufftComplex*>(inputf.get_p_data());
    float *out = reinterpret_cast<float*>(real_result.data);

    if(n_channels == 1)
        cufftExecC2R(plan_i_1ch, in, out);
    else if(n_channels == (int) m_num_of_scales)
        cufftExecC2R(plan_i_1ch_all_scales, in, out);
    else if(n_channels == (int) m_num_of_feats * (int) m_num_of_scales)
        cufftExecC2R(plan_i_features_all_scales, in, out);
    else
        cufftExecC2R(plan_i_features, in, out);

    return real_result/(m_width*m_height);
}

cuFFT::~cuFFT()
{

  for(unsigned i = 0; i < m_num_of_streams; i++) cudaStreamDestroy(streams[i]);

  cudaDeviceReset();
}
