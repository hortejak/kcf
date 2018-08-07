#ifndef SCALE_VARS_HPP
#define SCALE_VARS_HPP

#ifdef CUFFT
  #include "complexmat.cuh"
#else
  #include "complexmat.hpp"
#endif

struct Scale_vars
{
public:
    Scale_vars();
    Scale_vars(int windows_size[2], int cell_size, int num_of_feats, int num_of_scales = 1,ComplexMat *model_xf = nullptr, ComplexMat *yf = nullptr,bool zero_index = false)
    {
        double alloc_size;

#ifdef CUFFT
        if (zero_index)
            cudaSetDeviceFlags(cudaDeviceMapHost);

        alloc_size = windows_size[0]/cell_size*windows_size[1]/cell_size*num_of_scales*sizeof(cufftReal);
        CudaSafeCall(cudaHostAlloc((void**)&this->data_i_1ch, alloc_size, cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer((void**)&this->data_i_1ch_d, (void*)this->data_i_1ch, 0));

        alloc_size = windows_size[0]/cell_size*windows_size[1]/cell_size*num_of_feats*sizeof(cufftReal);
        CudaSafeCall(cudaHostAlloc((void**)&this->data_i_features, alloc_size, cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer((void**)&this->data_i_features_d, (void*)this->data_i_features, 0));

        this->ifft2_res = cv::Mat(windows_size[1]/cell_size, windows_size[0]/cell_size, CV_32FC(num_of_feats), this->data_i_features);
        this->response = cv::Mat(windows_size[1]/cell_size, windows_size[0]/cell_size, CV_32FC(num_of_scales), this->data_i_1ch);

        this->zf = ComplexMat(windows_size[1]/cell_size, (windows_size[0]/cell_size)/2+1, num_of_feats, num_of_scales);
        this->kzf = ComplexMat(windows_size[1]/cell_size, (windows_size[0]/cell_size)/2+1, num_of_scales);
        this->kf = ComplexMat(windows_size[1]/cell_size, (windows_size[0]/cell_size)/2+1, num_of_scales);

        alloc_size = num_of_scales;

        CudaSafeCall(cudaHostAlloc((void**)&this->xf_sqr_norm, alloc_size*sizeof(float), cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer((void**)&this->xf_sqr_norm_d, (void*)this->xf_sqr_norm, 0));

        CudaSafeCall(cudaHostAlloc((void**)&this->yf_sqr_norm, sizeof(float), cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer((void**)&this->yf_sqr_norm_d, (void*)this->yf_sqr_norm, 0));

        alloc_size =(windows_size[0]/cell_size)*(windows_size[1]/cell_size)*alloc_size*sizeof(float);
        CudaSafeCall(cudaHostAlloc((void**)&this->gauss_corr_res, alloc_size, cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer((void**)&this->gauss_corr_res_d, (void*)this->gauss_corr_res, 0));
        this->in_all = cv::Mat(windows_size[1]/cell_size*num_of_scales, windows_size[0]/cell_size, CV_32F, this->gauss_corr_res_d);

        if (zero_index) {
            alloc_size = (windows_size[0]/cell_size)*(windows_size[1]/cell_size)*sizeof(float);
            CudaSafeCall(cudaHostAlloc((void**)&this->rot_labels_data, alloc_size, cudaHostAllocMapped));
            CudaSafeCall(cudaHostGetDevicePointer((void**)&this->rot_labels_data_d, (void*)this->rot_labels_data, 0));
            this->rot_labels = cv::Mat(windows_size[1]/cell_size, windows_size[0]/cell_size, CV_32FC1, this->rot_labels_data);
        }

        alloc_size = (windows_size[0]/cell_size)*((windows_size[1]/cell_size)*num_of_feats)*sizeof(cufftReal);
        CudaSafeCall(cudaHostAlloc((void**)&this->data_features, alloc_size, cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer((void**)&this->data_features_d, (void*)this->data_features, 0));
        this->fw_all = cv::Mat((windows_size[1]/cell_size)*num_of_feats, windows_size[0]/cell_size, CV_32F, this->data_features);
#else
        alloc_size = num_of_scales;

        this->xf_sqr_norm = (float*) malloc(alloc_size*sizeof(float));
        this->yf_sqr_norm = (float*) malloc(sizeof(float));

        this->patch_feats.reserve(num_of_feats);

        int height = windows_size[1]/cell_size;
#ifdef FFTW
        int width = (windows_size[0]/cell_size)/2+1;
#else
        int width = windows_size[0]/cell_size;
#endif

        this->ifft2_res = cv::Mat(height, windows_size[0]/cell_size, CV_32FC(num_of_feats));
        this->response = cv::Mat(height, windows_size[0]/cell_size, CV_32FC(num_of_scales));

        this->zf = ComplexMat(height, width, num_of_feats, num_of_scales);
        this->kzf = ComplexMat(height, width, num_of_scales);
        this->kf = ComplexMat(height, width, num_of_scales);
#ifdef FFTW
        this->in_all = cv::Mat((windows_size[1]/cell_size)*num_of_scales, windows_size[0]/cell_size, CV_32F);
        this->fw_all = cv::Mat((windows_size[1]/cell_size)*num_of_feats, windows_size[0]/cell_size, CV_32F);
#else
        this->in_all = cv::Mat((windows_size[1]/cell_size), windows_size[0]/cell_size, CV_32F);
#endif
#endif
#if defined(FFTW) || defined(CUFFT)
        if (zero_index) {
            model_xf->create(windows_size[1]/cell_size, (windows_size[0]/cell_size)/2+1, num_of_feats);
            yf->create(windows_size[1]/cell_size, (windows_size[0]/cell_size)/2+1, 1);
            //We use scale_vars[0] for updating the tracker, so we only allocate memory for  its xf only.
            this->xf.create(windows_size[1]/cell_size, (windows_size[0]/cell_size)/2+1, num_of_feats);
        } else if (num_of_scales > 1) {
            this->max_responses.reserve(num_of_scales);
            this->max_locs.reserve(num_of_scales);
            this->response_maps.reserve(num_of_scales);
        }
#else
        if (zero_index) {
            model_xf->create(windows_size[1]/cell_size, windows_size[0]/cell_size, num_of_feats);
            yf->create(windows_size[1]/cell_size, windows_size[0]/cell_size, 1);
            this->xf.create(windows_size[1]/cell_size, windows_size[0]/cell_size, num_of_feats);
        }
#endif
    }

    float *xf_sqr_norm = nullptr, *yf_sqr_norm = nullptr;
    std::vector<cv::Mat> patch_feats;

    cv::Mat in_all, fw_all, ifft2_res, response;
    ComplexMat zf, kzf, kf, xyf, xf;

    //CuFFT variables
    cv::Mat rot_labels;
    float *xf_sqr_norm_d = nullptr, *yf_sqr_norm_d = nullptr, *gauss_corr_res = nullptr, *gauss_corr_res_d = nullptr, *rot_labels_data = nullptr,
              *rot_labels_data_d = nullptr, *data_features = nullptr, *data_features_d = nullptr;
    float *data_f = nullptr, *data_i_features = nullptr, *data_i_features_d = nullptr, *data_i_1ch = nullptr, *data_i_1ch_d = nullptr;

    //Big batch variables
    cv::Point2i max_loc;
    double max_val, max_response;

    std::vector<double> max_responses;
    std::vector<cv::Point2i> max_locs;
    std::vector<cv::Mat> response_maps;
};

#endif // SCALE_VARS_HPP
