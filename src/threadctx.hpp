#ifndef SCALE_VARS_HPP
#define SCALE_VARS_HPP

#include <future>
#include "dynmem.hpp"
#include "kcf.h"

#ifdef CUFFT
#include "complexmat.cuh"
#else
#include "complexmat.hpp"
#endif

class KCF_Tracker;

struct ThreadCtx {
  public:
    ThreadCtx(cv::Size roi, uint num_features
#ifdef BIG_BATCH
              , uint num_scales
#else
              , double scale
#endif
             )
        : roi(roi)
        , num_features(num_features)
        , num_scales(IF_BIG_BATCH(num_scales, 1))
#ifndef BIG_BATCH
        , scale(scale)
#endif
    {}

    ThreadCtx(ThreadCtx &&) = default;

    void track(const KCF_Tracker &kcf, cv::Mat &input_rgb, cv::Mat &input_gray);
private:
    cv::Size roi;
    uint num_features;
    uint num_scales;
    cv::Size freq_size = Fft::freq_size(roi);

    MatScaleFeats patch_feats{num_scales, num_features, roi};
    MatScaleFeats temp{num_scales, num_features, roi};

    KCF_Tracker::GaussianCorrelation gaussian_correlation{num_scales, roi};

    MatScales ifft2_res{num_scales, roi};

    ComplexMat zf{uint(freq_size.height), uint(freq_size.width), num_features, num_scales};
    ComplexMat kzf{uint(freq_size.height), uint(freq_size.width), num_scales};

public:
#ifdef ASYNC
    std::future<void> async_res;
#endif

    MatScales response{num_scales, roi};

    struct Max {
        cv::Point2i loc;
        double response;
    };

#ifdef BIG_BATCH
    std::vector<Max> max = std::vector<Max>(num_scales);
#else
    Max max;
    const double scale;
#endif
};

#endif // SCALE_VARS_HPP
