#ifndef SCALE_VARS_HPP
#define SCALE_VARS_HPP

#include <future>
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
    ThreadCtx(cv::Size roi, uint num_of_feats, double scale, int angle, uint num_of_scales = 1, uint num_of_angles = 1)
        : scale(scale), angle(angle)
    {
        this->xf_sqr_norm = DynMem(num_of_scales * num_of_angles * sizeof(float));
        this->yf_sqr_norm = DynMem(sizeof(float));

        uint cells_size = roi.width * roi.height * sizeof(float);

#if  !defined(BIG_BATCH) && defined(CUFFT) && (defined(ASYNC) || defined(OPENMP))
        CudaSafeCall(cudaStreamCreate(&this->stream));
#endif

#if defined(CUFFT) || defined(FFTW)
        this->gauss_corr_res = DynMem(cells_size * num_of_scales * num_of_angles);
        this->data_features = DynMem(cells_size * num_of_feats);

        uint width_freq = roi.width / 2 + 1;

        this->in_all = cv::Mat(roi.height * num_of_scales * num_of_angles, roi.width, CV_32F, this->gauss_corr_res.hostMem());
        this->fw_all = cv::Mat(roi.height * num_of_feats, roi.width, CV_32F, this->data_features.hostMem());
#else
        uint width_freq = roi.width;

        this->in_all = cv::Mat(roi, CV_32F);
#endif

        this->data_i_features = DynMem(cells_size * num_of_feats);
        this->data_i_1ch = DynMem(cells_size * num_of_scales * num_of_angles);

        this->ifft2_res = cv::Mat(roi, CV_32FC(num_of_feats), this->data_i_features.hostMem());
        this->response = cv::Mat(roi, CV_32FC(num_of_scales * num_of_angles), this->data_i_1ch.hostMem());

#ifdef CUFFT
        this->zf.create(roi.height, width_freq, num_of_feats, num_of_scales * num_of_angles, this->stream);
        this->kzf.create(roi.height, width_freq, num_of_scales * num_of_angles, this->stream);
        this->kf.create(roi.height, width_freq, num_of_scales * num_of_angles, this->stream);
#else
        this->zf.create(roi.height, width_freq, num_of_feats, num_of_scales * num_of_angles);
        this->kzf.create(roi.height, width_freq, num_of_scales * num_of_angles);
        this->kf.create(roi.height, width_freq, num_of_scales * num_of_angles);
#endif

#ifdef BIG_BATCH
        if (num_of_scales > 1) {
            this->max_responses.reserve(num_of_scales * num_of_angles);
            this->max_locs.reserve(num_of_scales * num_of_angles);
            this->response_maps.reserve(num_of_scales * num_of_angles);
        }
#endif
    }
    ThreadCtx(ThreadCtx &&) = default;
    ~ThreadCtx()
    {
#if  !defined(BIG_BATCH) && defined(CUFFT) && (defined(ASYNC) || defined(OPENMP))
        CudaSafeCall(cudaStreamDestroy(this->stream));
#endif
    }

    const double scale;
    const int angle;
#ifdef ASYNC
    std::future<void> async_res;
#endif

    DynMem xf_sqr_norm, yf_sqr_norm;

    cv::Mat in_all, fw_all, ifft2_res, response;
    ComplexMat zf, kzf, kf, xyf;

    DynMem data_i_features, data_i_1ch;
    // CuFFT and FFTW variables
    DynMem gauss_corr_res, data_features;

    // CuFFT variables
    cudaStream_t stream = nullptr;
    ComplexMat model_alphaf, model_xf;

    // Variables used during non big batch mode and in big batch mode with ThreadCtx in p_threadctxs in kcf  on zero index.
    cv::Point2i max_loc;
    double max_val, max_response;

#ifdef BIG_BATCH
    // Stores value of responses, location of maximal response and response maps for each scale
    std::vector<double> max_responses;
    std::vector<cv::Point2i> max_locs;
    std::vector<cv::Mat> response_maps;
#endif
};

#endif // SCALE_VARS_HPP
