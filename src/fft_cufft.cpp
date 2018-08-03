#include "fft_cufft.h"

void cuFFT::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales, bool big_batch_mode)
{
    m_width = width;
    m_height = height;
    m_num_of_feats = num_of_feats;
    m_num_of_scales = num_of_scales;
    m_big_batch_mode = big_batch_mode;

    std::cout << "FFT: cuFFT" << std::endl;

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
#ifdef BIG_BATCH
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
#endif
    //FFT inverse one channel one scale
    {
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
#ifdef BIG_BATCH
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
#endif
}

void cuFFT::set_window(const cv::Mat & window)
{
     m_window = window;
}

ComplexMat cuFFT::forward(const cv::Mat & input)
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

void cuFFT::forward(Scale_vars & vars)
{
    ComplexMat *complex_result = vars.flag & Tracker_flags::TRACKER_INIT ? vars.p_yf_ptr :
                                                  vars.flag & Tracker_flags::AUTO_CORRELATION ? & vars.kf : & vars.kzf;
    cv::Mat *input = vars.flag & Tracker_flags::TRACKER_INIT ? & vars.rot_labels : & vars.in_all;

    if(m_big_batch_mode && vars.in_all.rows == (int)(m_height*m_num_of_scales)){
        CudaSafeCall(cudaMemcpy(data_f_all_scales, input->ptr<cufftReal>(), m_height*m_num_of_scales*m_width*sizeof(cufftReal), cudaMemcpyHostToDevice));
        CufftErrorCheck(cufftExecR2C(plan_f_all_scales, reinterpret_cast<cufftReal*>(data_f_all_scales),
                                complex_result->get_p_data()));
    } else {
        CudaSafeCall(cudaMemcpy(data_f, input->ptr<cufftReal>(), m_height*m_width*sizeof(cufftReal), cudaMemcpyHostToDevice));
        CufftErrorCheck(cufftExecR2C(plan_f, reinterpret_cast<cufftReal*>(data_f),
                                complex_result->get_p_data()));
    }
    return;
}

void cuFFT::forward_raw(Scale_vars & vars, bool all_scales)
{
    ComplexMat *result = vars.flag & Tracker_flags::AUTO_CORRELATION ? & vars.kf : & vars.kzf;
    if (all_scales){
        CufftErrorCheck(cufftExecR2C(plan_f_all_scales, reinterpret_cast<cufftReal*>(vars.gauss_corr_res),
                                result->get_p_data()));
    } else {
        CufftErrorCheck(cufftExecR2C(plan_f, reinterpret_cast<cufftReal*>(vars.gauss_corr_res),
                                result->get_p_data()));
    }
    return;
}

ComplexMat cuFFT::forward_window(const std::vector<cv::Mat> & input)
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

void cuFFT::forward_window(Scale_vars & vars)
{
    int n_channels = vars.patch_feats.size();

    ComplexMat *result = vars.flag & Tracker_flags::TRACKER_INIT ? vars.p_model_xf_ptr :
                                                  vars.flag & Tracker_flags::TRACKER_UPDATE ? & vars.xf : & vars.zf;

    if(n_channels > (int) m_num_of_feats){
        cv::Mat in_all(m_height * n_channels, m_width, CV_32F, data_fw_all_scales);
        for (int i = 0; i < n_channels; ++i) {
            cv::Mat in_roi(in_all, cv::Rect(0, i*m_height, m_width, m_height));
            in_roi = vars.patch_feats[i].mul(m_window);
        }

        CufftErrorCheck(cufftExecR2C(plan_fw_all_scales, reinterpret_cast<cufftReal*>(data_fw_all_scales_d), result->get_p_data()));
    } else {
        cv::Mat in_all(m_height * n_channels, m_width, CV_32F, data_fw);
        for (int i = 0; i < n_channels; ++i) {
            cv::Mat in_roi(in_all, cv::Rect(0, i*m_height, m_width, m_height));
            in_roi = vars.patch_feats[i].mul(m_window);
        }

        CufftErrorCheck(cufftExecR2C(plan_fw, reinterpret_cast<cufftReal*>(data_fw_d), result->get_p_data()));
    }
    return;
}

void cuFFT::inverse(Scale_vars & vars)
{
    ComplexMat *input = vars.flag & Tracker_flags::RESPONSE ? & vars.kzf : &  vars.xyf;
    cv::Mat *real_result = vars.flag & Tracker_flags::RESPONSE ? & vars.response : & vars.ifft2_res;

    int n_channels = input->n_channels;
    cufftComplex *in = reinterpret_cast<cufftComplex*>(input->get_p_data());

    if(n_channels == 1){

        CufftErrorCheck(cufftExecC2R(plan_i_1ch, in, reinterpret_cast<cufftReal*>(vars.data_i_1ch_d)));
        cudaDeviceSynchronize();
        *real_result = *real_result/(m_width*m_height);
        return;
    }
#ifdef BIG_BATCH
    else if(n_channels == (int) m_num_of_scales){
        cv::Mat real_result(m_height, m_width, CV_32FC(n_channels), vars.data_i_1ch_all_scales);

        CufftErrorCheck(cufftExecC2R(plan_i_1ch_all_scales, in, reinterpret_cast<cufftReal*>(vars.data_i_1ch_all_scales_d)));
        cudaDeviceSynchronize();

        return real_result/(m_width*m_height);
    } else if(n_channels == (int) m_num_of_feats * (int) m_num_of_scales){
        cv::Mat real_result(m_height, m_width, CV_32FC(n_channels), data_i_features_all_scales);

        CufftErrorCheck(cufftExecC2R(plan_i_features_all_scales, in, reinterpret_cast<cufftReal*>(vars.data_i_features_all_scales_d)));
        cudaDeviceSynchronize();

        return real_result/(m_width*m_height);
    }
#endif

    CufftErrorCheck(cufftExecC2R(plan_i_features, in, reinterpret_cast<cufftReal*>(vars.data_i_features_d)));

    if (vars.cuda_gauss)
        return;
    else {
        cudaDeviceSynchronize();
        *real_result = *real_result/(m_width*m_height);
    }
    return;
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
