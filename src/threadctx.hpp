#ifndef SCALE_VARS_HPP
#define SCALE_VARS_HPP

#include "dynmem.hpp"

#ifdef CUFFT
#include "complexmat.cuh"
#else
#include "complexmat.hpp"
#ifndef CUFFTW
// For compatibility reasons between CuFFT and FFTW, OpenCVfft versions.
typedef int *cudaStream_t;
#endif
#endif

struct ThreadCtx {
  public:
    ThreadCtx(cv::Size windows_size, uint cell_size, uint num_of_feats, uint num_of_scales = 1,
              ComplexMat *model_xf = nullptr, ComplexMat *yf = nullptr, bool zero_index = false)
    {
#ifdef CUFFT
        if (zero_index) {
            cudaSetDeviceFlags(cudaDeviceMapHost);
            this->zero_index = true;
        }

#if defined(ASYNC) || defined(OPENMP)
        CudaSafeCall(cudaStreamCreate(&this->stream));
#endif

        this->patch_feats.reserve(uint(num_of_feats));
        // Size of cufftReal == float
        uint cells_size =
            ((uint(windows_size.width) / cell_size) * (uint(windows_size.height) / cell_size)) * sizeof(float);

        this->data_i_1ch = DynMem(cells_size * num_of_scales);
        this->data_i_features = DynMem(cells_size * num_of_feats);

        this->ifft2_res = cv::Mat(windows_size.height / int(cell_size), windows_size.width / int(cell_size),
                                  CV_32FC(int(num_of_feats)), this->data_i_features.hostMem());
        this->response = cv::Mat(windows_size.height / int(cell_size), windows_size.width / int(cell_size),
                                 CV_32FC(int(num_of_scales)), this->data_i_1ch.hostMem());

        this->zf.create(uint(windows_size.height) / cell_size, (uint(windows_size.width) / cell_size) / 2 + 1,
                        num_of_feats, num_of_scales, this->stream);
        this->kzf.create(uint(windows_size.height) / cell_size, (uint(windows_size.width) / cell_size) / 2 + 1,
                         num_of_scales, this->stream);
        this->kf.create(uint(windows_size.height) / cell_size, (uint(windows_size.width) / cell_size) / 2 + 1,
                        num_of_scales, this->stream);

        this->xf_sqr_norm = DynMem(num_of_scales * sizeof(float));
        this->yf_sqr_norm = DynMem(sizeof(float));

        this->gauss_corr_res = DynMem(cells_size * num_of_scales);
        this->in_all = cv::Mat(windows_size.height / int(cell_size) * int(num_of_scales),
                               windows_size.width / int(cell_size), CV_32F, this->gauss_corr_res.hostMem());

        if (zero_index) {
            this->rot_labels_data = DynMem(cells_size);
            this->rot_labels = cv::Mat(windows_size.height / int(cell_size), windows_size.width / int(cell_size),
                                       CV_32FC1, this->rot_labels_data.hostMem());
        }

        this->data_features = DynMem(cells_size * num_of_feats);
        this->fw_all = cv::Mat((windows_size.height / int(cell_size)) * int(num_of_feats),
                               windows_size.width / int(cell_size), CV_32F, this->data_features.hostMem());
#else

        this->xf_sqr_norm = DynMem(num_of_scales * sizeof(float));
        this->yf_sqr_norm = DynMem(sizeof (float));

        this->patch_feats.reserve(num_of_feats);

        uint height = uint(windows_size.height) / cell_size;
#ifdef FFTW
        uint width = (uint(windows_size.width) / cell_size) / 2 + 1;
#else
        int width = windows_size.width / cell_size;
#endif

        this->ifft2_res = cv::Mat(int(height), windows_size.width / int(cell_size), CV_32FC(int(num_of_feats)));
        this->response = cv::Mat(int(height), windows_size.width / int(cell_size), CV_32FC(int(num_of_scales)));

        this->zf = ComplexMat(height, width, num_of_feats, num_of_scales);
        this->kzf = ComplexMat(height, width, num_of_scales);
        this->kf = ComplexMat(height, width, num_of_scales);
#ifdef FFTW
        this->in_all = cv::Mat((windows_size.height / int(cell_size)) * int(num_of_scales),
                               windows_size.width / int(cell_size), CV_32F);
        this->fw_all = cv::Mat((windows_size.height / int(cell_size)) * int(num_of_feats),
                               windows_size.width / int(cell_size), CV_32F);
#else
        this->in_all = cv::Mat((windows_size.height / int(cell_size)), windows_size.width / int(cell_size), CV_32F);
#endif
#endif
#if defined(FFTW) || defined(CUFFT)
        if (zero_index) {
            model_xf->create(uint(windows_size.height) / cell_size, (uint(windows_size.width) / cell_size) / 2 + 1,
                             num_of_feats);
            yf->create(uint(windows_size.height) / cell_size, (uint(windows_size.width) / cell_size) / 2 + 1, 1);
            // We use scale_vars[0] for updating the tracker, so we only allocate memory for  its xf only.
#ifdef CUFFT
            this->xf.create(uint(windows_size.height) / cell_size, (uint(windows_size.width) / cell_size) / 2 + 1,
                            num_of_feats, this->stream);
#else
            this->xf.create(uint(windows_size.height) / cell_size, (uint(windows_size.width) / cell_size) / 2 + 1,
                            num_of_feats);
#endif
        } else if (num_of_scales > 1) {
            this->max_responses.reserve(uint(num_of_scales));
            this->max_locs.reserve(uint(num_of_scales));
            this->response_maps.reserve(uint(num_of_scales));
        }
#else
        if (zero_index) {
            model_xf->create(windows_size.height / cell_size, windows_size.width / cell_size, num_of_feats);
            yf->create(windows_size.height / cell_size, windows_size.width / cell_size, 1);
            this->xf.create(windows_size.height / cell_size, windows_size.width / cell_size, num_of_feats);
        }
#endif
    }

    ~ThreadCtx()
    {
#if defined(CUFFT) && (defined(ASYNC) || defined(OPENMP))
        CudaSafeCall(cudaStreamDestroy(this->stream));
#endif
    }

    DynMem xf_sqr_norm, yf_sqr_norm;
    std::vector<cv::Mat> patch_feats;

    cv::Mat in_all, fw_all, ifft2_res, response;
    ComplexMat zf, kzf, kf, xyf, xf;

    // CuFFT variables
    cv::Mat rot_labels;
    DynMem gauss_corr_res, rot_labels_data, data_features, data_f, data_i_features, data_i_1ch;

    cudaStream_t stream = nullptr;
    ComplexMat model_alphaf, model_xf;

    // Big batch variables
    cv::Point2i max_loc;
    double max_val, max_response;

    std::vector<double> max_responses;
    std::vector<cv::Point2i> max_locs;
    std::vector<cv::Mat> response_maps;
    bool zero_index = false;
};

#endif // SCALE_VARS_HPP
