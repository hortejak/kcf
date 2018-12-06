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
    static constexpr uint num_features = KCF_Tracker::p_num_of_feats;
    static constexpr uint num_scales = KCF_Tracker::p_num_scales;
    static constexpr uint num_angles = KCF_Tracker::p_num_angles;

    ThreadCtx(cv::Size roi
#ifdef BIG_BATCH
              , const std::vector<double> &scales
              , const std::vector<double> &angles
#else
              , double scale
              , double angle
#endif
             )
        : roi(roi)
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
        {}
#endif


    ThreadCtx(ThreadCtx &&) = default;
    void track(const KCF_Tracker &kcf, cv::Mat &input_rgb, cv::Mat &input_gray);
private:
    cv::Size roi;
    cv::Size freq_size = Fft::freq_size(roi);

    MatScaleFeats patch_feats{num_scales * num_angles, num_features, roi};
    MatScaleFeats temp{num_scales * num_angles, num_features, roi};

    KCF_Tracker::GaussianCorrelation<num_features,num_scales*num_angles> gaussian_correlation{roi};

    MatScales ifft1_res{num_scales * num_angles, roi};

    ComplexMat<num_features,num_scales*num_angles> zf;
    ComplexMat<1,num_scales*num_angles> kzf;
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
