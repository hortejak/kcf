#ifndef SCALE_VARS_HPP
#define SCALE_VARS_HPP

#include <future>
#include "dynmem.hpp"
#include "kcf.h"
#include "complexmat.hpp"
#include <vector>

class KCF_Tracker;

template <typename T>
class ScaleRotVector : public std::vector<T> {
public:
    ScaleRotVector(const std::vector<double> &scales, const std::vector<double> &angles)
        : scales(scales)
        , angles(angles)
    {}

    uint getIdx(uint scale_idx, uint angle_idx) const { return angles.size() * scale_idx + angle_idx; }
    uint getScaleIdx(uint idx) const { return idx / angles.size(); }
    uint getAngleIdx(uint idx) const { return idx % angles.size(); }
    T& operator()(uint scale_idx, uint angle_idx) { return std::vector<T>::at(getIdx(scale_idx, angle_idx)); }
    double scale(uint idx) const { return scales[getScaleIdx(idx)]; }
    double angle(uint idx) const { return angles[getAngleIdx(idx)]; }
private:
    const std::vector<double> scales, angles;
};

struct ThreadCtx {
  public:
    static constexpr uint n_feats = KCF_Tracker::p_num_of_feats;
    static constexpr uint n_scales = KCF_Tracker::p_num_scales;
    static constexpr uint n_angles = KCF_Tracker::p_num_angles;

    ThreadCtx(cv::Size roi, uint num_features
#ifdef BIG_BATCH
              , const std::vector<double> &scales
              , const std::vector<double> &angles
#else
              , double scale
              , double angle
#endif
             )
        : roi(roi)
        , num_features(num_features)
        , num_scales(IF_BIG_BATCH(scales.size(), 1))
        , num_angles(IF_BIG_BATCH(angles.size(), 1))
#ifdef BIG_BATCH
        , max(scales, angles)
        , dbg_patch(scales, angles)
        {
            max.resize(scales.size() * angles.size());
            dbg_patch.resize(scales.size() * angles.size());
        }
#else
        , scale(scale)
        , angle(angle)
        {
            assert(num_features==n_feats);
            assert(num_scales  ==n_scales);
            assert(num_angles  ==n_angles);
        }
#endif


    ThreadCtx(ThreadCtx &&) = default;
    void track(const KCF_Tracker &kcf, cv::Mat &input_rgb, cv::Mat &input_gray);
private:
    cv::Size roi;
    uint num_features;
    uint num_scales;
    uint num_angles;
    cv::Size freq_size = Fft::freq_size(roi);

    MatScaleFeats patch_feats{num_scales * num_angles, num_features, roi};
    MatScaleFeats temp{num_scales * num_angles, num_features, roi};

    KCF_Tracker::GaussianCorrelation<n_feats,n_scales*n_angles> gaussian_correlation{num_scales*num_angles, num_features, roi};

    MatScales ifft1_res{num_scales * num_angles, roi};

    ComplexMat<n_feats,n_scales*n_angles> zf{uint(freq_size.height), uint(freq_size.width), num_features, num_scales * num_angles};
    ComplexMat<1,n_scales*n_angles> kzf{uint(freq_size.height), uint(freq_size.width), 1, num_scales * num_angles};
public:
#ifdef ASYNC
    std::future<void> async_res;
#endif

    MatScales response{num_scales * num_angles, roi};

    struct Max {
        cv::Point2i loc;
        double response;
    };

#ifdef BIG_BATCH
    ScaleRotVector<Max> max;
    ScaleRotVector<cv::Mat> dbg_patch; // images for visual debugging
#else
    Max max;
    const double scale, angle;
    cv::Mat dbg_patch; // image for visual debugging
#endif
};

#endif // SCALE_VARS_HPP
