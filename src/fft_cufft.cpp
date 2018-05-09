#include "fft_cufft.h"

void cuFFT::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales, bool big_batch_mode)
{
    m_width = width;
    m_height = height;
    m_num_of_feats = num_of_feats;
    m_num_of_scales = num_of_scales;
    m_big_batch_mode = big_batch_mode;

    std::cout << "FFT: cuFFT" << std::endl;

    if(m_height*(m_width/2+1) > 1024){
        std::cerr << "Image dimension after forward FFT are too big for CUDA kernels." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    //FFT forward one scale
    {
        CudaSafeCall(cudaMalloc(&data_f, m_height*m_width*sizeof(cufftReal)));

       CufftErrorCheck(cufftPlan2d(&plan_f, m_height, m_width, CUFFT_R2C));


    }
    //FFT forward all scales
    if(m_num_of_scales > 1 && m_big_batch_mode)
    {
        CudaSafeCall(cudaMalloc(&data_f_all_scales, m_height*m_num_of_scales*m_width*sizeof(cufftReal)));

	int rank = 2;
	int n[] = {(int)m_height, (int)m_width};
	int howmany = m_num_of_scales;
	int idist = m_height*m_width, odist = m_height*(m_width/2+1);
	int istride = 1, ostride = 1;
	int *inembed = n, onembed[] = {(int)m_height, (int)m_width/2+1};

	CufftErrorCheck(cufftPlanMany(&plan_f_all_scales, rank, n,
		      inembed, istride, idist,
		      onembed, ostride, odist,
		      CUFFT_R2C, howmany));
    }
    //FFT forward window one scale
    {
        CudaSafeCall(cudaHostAlloc(&data_fw, m_height*m_num_of_feats*m_width*sizeof(cufftReal), cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer(&data_fw_d, data_fw, 0));

        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_feats;
        int idist = m_height*m_width, odist = m_height*(m_width/2+1);
        int istride = 1, ostride = 1;
        int *inembed = n, onembed[] = {(int)m_height, (int)m_width/2+1};

        CufftErrorCheck(cufftPlanMany(&plan_fw, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_R2C, howmany));
    }
    //FFT forward window all scales all feats
    if(m_num_of_scales > 1 && m_big_batch_mode)
    {
        CudaSafeCall(cudaHostAlloc(&data_fw_all_scales, m_height*m_num_of_feats*m_num_of_scales*m_width*sizeof(cufftReal), cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer(&data_fw_all_scales_d, data_fw_all_scales, 0));

        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_scales*m_num_of_feats;
        int idist = m_height*m_width, odist = m_height*(m_width/2+1);
        int istride = 1, ostride = 1;
        int *inembed = n, onembed[] = {(int)m_height, (int)m_width/2+1};

        CufftErrorCheck(cufftPlanMany(&plan_fw_all_scales, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_R2C, howmany));


    }
    //FFT inverse one scale
    {
        CudaSafeCall(cudaHostAlloc(&data_i_features, m_height*m_num_of_feats*m_width*sizeof(cufftReal), cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer(&data_i_features_d, data_i_features, 0));

        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_feats;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = m_num_of_feats;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        CufftErrorCheck(cufftPlanMany(&plan_i_features, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_C2R, howmany));
    }
    //FFT inverse all scales
    if(m_num_of_scales > 1)
    {
        CudaSafeCall(cudaHostAlloc(&data_i_features_all_scales, m_height*m_num_of_feats*m_num_of_scales*m_width*sizeof(cufftReal), cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer(&data_i_features_all_scales_d, data_i_features_all_scales, 0));

        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_feats*m_num_of_scales;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = m_num_of_feats*m_num_of_scales;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        CufftErrorCheck(cufftPlanMany(&plan_i_features_all_scales, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_C2R, howmany));
    }
    //FFT inverse one channel one scale
    {
        CudaSafeCall(cudaHostAlloc(&data_i_1ch, m_height*m_width*sizeof(cufftReal), cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer(&data_i_1ch_d, data_i_1ch, 0));

        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = 1;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = 1;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        CufftErrorCheck(cufftPlanMany(&plan_i_1ch, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_C2R, howmany));
    }
    //FFT inverse one channel all scales
    if(m_num_of_scales > 1 && m_big_batch_mode)
    {
        CudaSafeCall(cudaHostAlloc(&data_i_1ch_all_scales, m_height*m_num_of_scales*m_width*sizeof(cufftReal), cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer(&data_i_1ch_all_scales_d, data_i_1ch_all_scales, 0));

        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_scales;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = m_num_of_scales;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        CufftErrorCheck(cufftPlanMany(&plan_i_1ch_all_scales, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_C2R, howmany));
    }
}

void cuFFT::set_window(const cv::Mat &window)
{
     m_window = window;
}

ComplexMat cuFFT::forward(const cv::Mat &input)
{
    ComplexMat complex_result;
    if(m_big_batch_mode && input.rows == (int)(m_height*m_num_of_scales)){
        CudaSafeCall(cudaMemcpy(data_f_all_scales, input.ptr<cufftReal>(), m_height*m_num_of_scales*m_width*sizeof(cufftReal), cudaMemcpyHostToDevice));
        complex_result.create(m_height, m_width / 2 + 1, m_num_of_scales);
        CufftErrorCheck(cufftExecR2C(plan_f_all_scales, reinterpret_cast<cufftReal*>(data_f_all_scales),
                                complex_result.get_p_data()));
    } else {
        CudaSafeCall(cudaMemcpy(data_f, input.ptr<cufftReal>(), m_height*m_width*sizeof(cufftReal), cudaMemcpyHostToDevice));
        complex_result.create(m_height, m_width/ 2 + 1, 1);
        CufftErrorCheck(cufftExecR2C(plan_f, reinterpret_cast<cufftReal*>(data_f),
                                complex_result.get_p_data()));
    }

    return complex_result;
}

ComplexMat cuFFT::forward_raw(float *input, bool all_scales)
{
    ComplexMat complex_result;
    if (all_scales){
        complex_result.create(m_height, m_width / 2 + 1, m_num_of_scales);
        CufftErrorCheck(cufftExecR2C(plan_f_all_scales, reinterpret_cast<cufftReal*>(input),
                                complex_result.get_p_data()));
    } else {
        complex_result.create(m_height, m_width/ 2 + 1, 1);
        CufftErrorCheck(cufftExecR2C(plan_f, reinterpret_cast<cufftReal*>(input),
                                complex_result.get_p_data()));
    }
    return complex_result;
}

ComplexMat cuFFT::forward_window(const std::vector<cv::Mat> &input)
{
    int n_channels = input.size();
    ComplexMat result;
    if(n_channels > (int) m_num_of_feats){
        cv::Mat in_all(m_height * n_channels, m_width, CV_32F, data_fw_all_scales);
        for (int i = 0; i < n_channels; ++i) {
            cv::Mat in_roi(in_all, cv::Rect(0, i*m_height, m_width, m_height));
            in_roi = input[i].mul(m_window);
        }

        result.create(m_height, m_width/2 + 1, n_channels,m_num_of_scales);

        CufftErrorCheck(cufftExecR2C(plan_fw_all_scales, reinterpret_cast<cufftReal*>(data_fw_all_scales_d), result.get_p_data()));
    } else {
        cv::Mat in_all(m_height * n_channels, m_width, CV_32F, data_fw);
        for (int i = 0; i < n_channels; ++i) {
            cv::Mat in_roi(in_all, cv::Rect(0, i*m_height, m_width, m_height));
            in_roi = input[i].mul(m_window);
        }

        result.create(m_height, m_width/2 + 1, n_channels);

        CufftErrorCheck(cufftExecR2C(plan_fw, reinterpret_cast<cufftReal*>(data_fw_d), result.get_p_data()));
    }
    return result;
}

cv::Mat cuFFT::inverse(const ComplexMat &input)
{
    int n_channels = input.n_channels;
    cufftComplex *in = reinterpret_cast<cufftComplex*>(input.get_p_data());

    if(n_channels == 1){
        cv::Mat real_result(m_height, m_width, CV_32FC1, data_i_1ch);

        CufftErrorCheck(cufftExecC2R(plan_i_1ch, in, reinterpret_cast<cufftReal*>(data_i_1ch_d)));
        cudaDeviceSynchronize();

        return real_result/(m_width*m_height);
    } else if(n_channels == (int) m_num_of_scales){
        cv::Mat real_result(m_height, m_width, CV_32FC(n_channels), data_i_1ch_all_scales);

        CufftErrorCheck(cufftExecC2R(plan_i_1ch_all_scales, in, reinterpret_cast<cufftReal*>(data_i_1ch_all_scales_d)));
        cudaDeviceSynchronize();

        return real_result/(m_width*m_height);
    } else if(n_channels == (int) m_num_of_feats * (int) m_num_of_scales){
        cv::Mat real_result(m_height, m_width, CV_32FC(n_channels), data_i_features_all_scales);

        CufftErrorCheck(cufftExecC2R(plan_i_features_all_scales, in, reinterpret_cast<cufftReal*>(data_i_features_all_scales_d)));
        cudaDeviceSynchronize();

        return real_result/(m_width*m_height);
    }

    cv::Mat real_result(m_height, m_width, CV_32FC(n_channels), data_i_features);

    CufftErrorCheck(cufftExecC2R(plan_i_features, in, reinterpret_cast<cufftReal*>(data_i_features_d)));
    cudaDeviceSynchronize();

    return real_result/(m_width*m_height);
}

float* cuFFT::inverse_raw(const ComplexMat &input)
{
    int n_channels = input.n_channels;
    cufftComplex *in = reinterpret_cast<cufftComplex*>(input.get_p_data());

    if(n_channels == 1){
        CufftErrorCheck(cufftExecC2R(plan_i_1ch, in, reinterpret_cast<cufftReal*>(data_i_1ch_d)));

        return data_i_1ch_d;
    } else if(n_channels == (int) m_num_of_scales){
        CufftErrorCheck(cufftExecC2R(plan_i_1ch_all_scales, in, reinterpret_cast<cufftReal*>(data_i_1ch_all_scales_d)));

        return data_i_1ch_all_scales_d;
    } else if(n_channels == (int) m_num_of_feats * (int) m_num_of_scales){
        CufftErrorCheck(cufftExecC2R(plan_i_features_all_scales, in, reinterpret_cast<cufftReal*>(data_i_features_all_scales_d)));

        return data_i_features_all_scales_d;
    }

    CufftErrorCheck(cufftExecC2R(plan_i_features, in, reinterpret_cast<cufftReal*>(data_i_features_d)));

    return data_i_features_d;
}

cuFFT::~cuFFT()
{
  CufftErrorCheck(cufftDestroy(plan_f));
  CufftErrorCheck(cufftDestroy(plan_fw));
  CufftErrorCheck(cufftDestroy(plan_i_1ch));
  CufftErrorCheck(cufftDestroy(plan_i_features));

  CudaSafeCall(cudaFree(data_f));
  CudaSafeCall(cudaFreeHost(data_fw));
  CudaSafeCall(cudaFreeHost(data_i_1ch));
  CudaSafeCall(cudaFreeHost(data_i_features));
  
  if (m_big_batch_mode) {
      CufftErrorCheck(cufftDestroy(plan_f_all_scales));
      CufftErrorCheck(cufftDestroy(plan_fw_all_scales));
      CufftErrorCheck(cufftDestroy(plan_i_1ch_all_scales));
      CufftErrorCheck(cufftDestroy(plan_i_features_all_scales));
      
      CudaSafeCall(cudaFree(data_f_all_scales));
      CudaSafeCall(cudaFreeHost(data_fw_all_scales));
      CudaSafeCall(cudaFreeHost(data_i_1ch_all_scales));
      CudaSafeCall(cudaFreeHost(data_i_features_all_scales));
  }
}
