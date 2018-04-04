#include "fft_cufft.h"

#ifdef _CUFFT_H_
// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";

	case CUFFT_INVALID_DEVICE:
	    return "CUFFT_INVALID_DEVICE";

	case CUFFT_PARSE_ERROR:
	    return "CUFFT_PARSE_ERROR";

	case CUFFT_NO_WORKSPACE:
	    return "CUFFT_NO_WORKSPACE";

	case CUFFT_NOT_IMPLEMENTED:
	    return "CUFFT_NOT_IMPLEMENTED";

	case CUFFT_LICENSE_ERROR:
	    return "CUFFT_LICENSE_ERROR";

	case CUFFT_NOT_SUPPORTED:
	    return "CUFFT_NOT_SUPPORTED";

	case CUFFT_INCOMPLETE_PARAMETER_LIST:
	    return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    }

    return "<unknown>";
}
#endif

#define CHECK_CUFFT_ERRORS(call) { \
    cufftResult_t err; \
    if ((err = (call)) != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %d:%s at %s:%d\n", err, _cudaGetErrorEnum(err), \
                __FILE__, __LINE__); \
        exit(1); \
    } \
}

cuFFT::cuFFT(): m_num_of_streams(4)
{
    cudaSetDeviceFlags(cudaDeviceMapHost);
}

void cuFFT::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales, bool big_batch_mode)
{
    m_width = width;
    m_height = height;
    m_num_of_feats = num_of_feats;
    m_num_of_scales = num_of_scales;
    m_big_batch_mode = big_batch_mode;

    std::cout << "FFT: cuFFT" << std::endl;

//     cudaSetDeviceFlags(cudaDeviceMapHost);
    

    for (unsigned i = 0; i < m_num_of_streams; i++) cudaStreamCreate(&streams[i]);
    
    //FFT forward one scale
    {
        cudaMalloc(&data_f, m_height*m_width*sizeof(cufftReal));
        
        cufftPlan2d(&plan_f, m_height, m_width, CUFFT_R2C);
    }
    //FFT forward all scales
    if(m_num_of_scales > 1 && m_big_batch_mode)
    {
        cudaMalloc(&data_f_all_scales, m_height*m_num_of_scales*m_width*sizeof(cufftReal));
        
	int rank = 2;
	int n[] = {(int)m_height, (int)m_width};
	int howmany = m_num_of_scales;
	int idist = m_height*m_width, odist = m_height*(m_width/2+1);
	int istride = 1, ostride = 1;
	int *inembed = NULL, *onembed = NULL;

	CHECK_CUFFT_ERRORS(cufftPlanMany(&plan_f_all_scales, rank, n,
		      inembed, istride, idist,
		      onembed, ostride, odist,
		      CUFFT_R2C, howmany));
    }
    //FFT forward window one scale
    {
        cudaMalloc(&data_fw, m_height*m_num_of_feats*m_width*sizeof(cufftReal));
        
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_feats;
        int idist = m_height*m_width, odist = m_height*(m_width/2+1);
        int istride = 1, ostride = 1;
        int *inembed = NULL, *onembed = NULL;

        CHECK_CUFFT_ERRORS(cufftPlanMany(&plan_fw, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_R2C, howmany));
    }
    //FFT forward window all scales all feats
    if(m_num_of_scales > 1 && m_big_batch_mode)
    {
        cudaMalloc(&data_fw_all_scales, m_height*m_num_of_feats*m_num_of_scales*m_width*sizeof(cufftReal));
        
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_scales*m_num_of_feats;
        int idist = m_height*m_width, odist = m_height*(m_width/2+1);
        int istride = 1, ostride = 1;
        int *inembed = NULL, *onembed = NULL;

        CHECK_CUFFT_ERRORS(cufftPlanMany(&plan_fw_all_scales, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_R2C, howmany));
    }
    //FFT inverse one scale
    {
        cudaMalloc(&data_i_features, m_height*m_num_of_feats*m_width*sizeof(cufftReal));
        
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_feats;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = m_num_of_feats;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        CHECK_CUFFT_ERRORS(cufftPlanMany(&plan_i_features, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_C2R, howmany));
    }
    //FFT inverse all scales
    if(m_num_of_scales > 1)
    {
        cudaMalloc(&data_i_features_all_scales, m_height*m_num_of_feats*m_num_of_scales*m_width*sizeof(cufftReal));
        
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_feats*m_num_of_scales;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = m_num_of_feats*m_num_of_scales;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        CHECK_CUFFT_ERRORS(cufftPlanMany(&plan_i_features_all_scales, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_C2R, howmany));
    }
    //FFT inverse one channel one scale
    {
        cudaMalloc(&data_i_1ch, m_height*m_width*sizeof(cufftReal));
        
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = 1;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = 1;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        CHECK_CUFFT_ERRORS(cufftPlanMany(&plan_i_1ch, rank, n,
		  inembed, istride, idist,
		  onembed, ostride, odist,
		  CUFFT_C2R, howmany));
    }
    //FFT inverse one channel all scales
    if(m_num_of_scales > 1 && m_big_batch_mode)
    {
        cudaMalloc(&data_i_1ch_all_scales, m_height*m_num_of_scales*m_width*sizeof(cufftReal));
        
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_scales;
        int idist = m_height*(m_width/2+1), odist = 1;
        int istride = 1, ostride = m_num_of_scales;
        int inembed[] = {(int)m_height, (int)m_width/2+1}, *onembed = n;

        CHECK_CUFFT_ERRORS(cufftPlanMany(&plan_i_1ch_all_scales, rank, n,
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
        cudaMemcpy(data_f_all_scales, input.ptr<cufftReal>(), m_height*m_num_of_scales*m_width*sizeof(cufftReal), cudaMemcpyHostToDevice);
        complex_result.create(m_height, m_width / 2 + 1, m_num_of_scales);
        CHECK_CUFFT_ERRORS(cufftExecR2C(plan_f_all_scales, reinterpret_cast<cufftReal*>(data_f_all_scales),
                                complex_result.get_p_data()));
    } else {
        cudaMemcpy(data_f, input.ptr<cufftReal>(), m_height*m_width*sizeof(cufftReal), cudaMemcpyHostToDevice);
        complex_result.create(m_height, m_width/ 2 + 1, 1);
        CHECK_CUFFT_ERRORS(cufftExecR2C(plan_f, reinterpret_cast<cufftReal*>(data_f),
                                complex_result.get_p_data()));
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
    ComplexMat result;
    if(n_channels > (int) m_num_of_feats){
        result.create(m_height, m_width/2 + 1, n_channels,m_num_of_scales);
    } else {
        result.create(m_height, m_width/2 + 1, n_channels);
    }

    if (n_channels <= (int) m_num_of_feats){
        cudaMemcpy(data_fw, in_all.ptr<cufftReal>(), m_height*m_num_of_feats*m_width*sizeof(cufftReal), cudaMemcpyHostToDevice);
        CHECK_CUFFT_ERRORS(cufftExecR2C(plan_fw, reinterpret_cast<cufftReal*>(data_fw), result.get_p_data()));
    } else {
       cudaMemcpy(data_fw_all_scales, in_all.ptr<cufftReal>(), m_height*n_channels*m_width*sizeof(cufftReal), cudaMemcpyHostToDevice);
       CHECK_CUFFT_ERRORS(cufftExecR2C(plan_fw_all_scales, reinterpret_cast<cufftReal*>(data_fw_all_scales), result.get_p_data()));
    }
    return result;
}

cv::Mat cuFFT::inverse(const ComplexMat &inputf)
{
    int n_channels = inputf.n_channels;
    cv::Mat real_result(m_height, m_width, CV_32FC(n_channels));
    cufftComplex *in = reinterpret_cast<cufftComplex*>(inputf.get_p_data());

    if(n_channels == 1){
        CHECK_CUFFT_ERRORS(cufftExecC2R(plan_i_1ch, in, reinterpret_cast<cufftReal*>(data_i_1ch)));
        cudaMemcpy(real_result.ptr<cufftReal>(), data_i_1ch, m_height*m_width*sizeof(cufftReal), cudaMemcpyDeviceToHost);
    } else if(n_channels == (int) m_num_of_scales){
        CHECK_CUFFT_ERRORS(cufftExecC2R(plan_i_1ch_all_scales, in, reinterpret_cast<cufftReal*>(data_i_1ch_all_scales)));
        cudaMemcpy(real_result.ptr<cufftReal>(), data_i_1ch_all_scales, m_height*n_channels*m_width*sizeof(cufftReal), cudaMemcpyDeviceToHost);
    } else if(n_channels == (int) m_num_of_feats * (int) m_num_of_scales){
        CHECK_CUFFT_ERRORS(cufftExecC2R(plan_i_features_all_scales, in, reinterpret_cast<cufftReal*>(data_i_features_all_scales)));
        cudaMemcpy(real_result.ptr<cufftReal>(), data_i_features_all_scales, m_height*n_channels*m_width*sizeof(cufftReal), cudaMemcpyDeviceToHost);
    } else {
        CHECK_CUFFT_ERRORS(cufftExecC2R(plan_i_features, in, reinterpret_cast<cufftReal*>(data_i_features)));
        cudaMemcpy(real_result.ptr<cufftReal>(), data_i_features, m_height*n_channels*m_width*sizeof(cufftReal), cudaMemcpyDeviceToHost);
    }
    return real_result/(m_width*m_height);
}

cuFFT::~cuFFT()
{

  for(unsigned i = 0; i < m_num_of_streams; i++) cudaStreamDestroy(streams[i]);
  
  cufftDestroy(plan_f);
  cufftDestroy(plan_f_all_scales);
  cufftDestroy(plan_fw);
  cufftDestroy(plan_fw_all_scales);
  cufftDestroy(plan_i_1ch);
  cufftDestroy(plan_i_1ch_all_scales);
  cufftDestroy(plan_i_features);
  cufftDestroy(plan_i_features_all_scales);
  
  cudaFree(data_f);
  cudaFree(data_f_all_scales);
  cudaFree(data_fw);
  cudaFree(data_fw_all_scales);
  cudaFree(data_i_1ch);
  cudaFree(data_i_1ch_all_scales);
  cudaFree(data_i_features);
  cudaFree(data_i_features_all_scales);
  
  cudaDeviceReset();
}
