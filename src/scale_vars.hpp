#ifndef SCALE_VARS_HPP
#define SCALE_VARS_HPP

#ifdef CUFFT
  #include "complexmat.cuh"
#else
  #include "complexmat.hpp"
#endif

enum Track_flags
{
    RESPONSE = 1 << 0, // binary 0001
    AUTO_CORRELATION = 1 << 1, // binary 0010
    CROSS_CORRELATION = 1 << 2, // binary 0100
    SCALE_RESPONSE = 1 << 3,// binary 1000
    TRACKER_UPDATE = 1 << 4,// binary 0001 0000
    TRACKER_INIT = 1 << 5, // binary 0010 0000
};

struct Scale_vars
{
    float *xf_sqr_norm = nullptr, *yf_sqr_norm = nullptr;
#ifdef CUFFT
    float *xf_sqr_norm_d = nullptr, *yf_sqr_norm_d = nullptr, *gauss_corr_res = nullptr;
    float *data_f = nullptr, *data_fw = nullptr, *data_fw_d = nullptr,  *data_i_features = nullptr,
              *data_i_features_d = nullptr, *data_i_1ch = nullptr, *data_i_1ch_d = nullptr;
#ifdef BIG_BATCH
    float *data_f_all_scales = nullptr, *data_fw_all_scales = nullptr, *data_fw_all_scales_d = nullptr, *data_i_features_all_scales = nullptr,
              *data_i_features_all_scales_d = nullptr, *data_i_1ch_all_scales = nullptr, *data_i_1ch_all_scales_d = nullptr;
#endif
    bool cuda_gauss = true;
#endif

    std::vector<cv::Mat> patch_feats;

    cv::Mat in_all, ifft2_res, response;
    ComplexMat zf, kzf, kf, xyf, xf;

    Track_flags flag;

    cv::Point2i max_loc;
    double max_val, max_response;
};

#endif // SCALE_VARS_HPP
