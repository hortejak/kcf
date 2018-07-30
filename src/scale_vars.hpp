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
};

struct Scale_vars
{
    float *xf_sqr_norm = nullptr, *yf_sqr_norm = nullptr;
#ifdef CUFFT
    float *xf_sqr_norm_d = nullptr, *yf_sqr_norm_d = nullptr, *gauss_corr_res = nullptr;
#endif

    std::vector<cv::Mat> patch_feats;

    cv::Mat in_all, ifft2_res, response;
    ComplexMat zf, kzf, kf, xyf;

    Track_flags flag;

    cv::Point2i max_loc;
    double max_val, max_response;
};

#endif // SCALE_VARS_HPP
