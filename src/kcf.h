#ifndef KCF_HEADER_6565467831231
#define KCF_HEADER_6565467831231

#include <opencv2/opencv.hpp>
#include <vector>
#include "fhog.hpp"

#ifdef CUFFT
  #include "complexmat.cuh"
  #include "cuda_functions.cuh"
  #include "cuda/cuda_error_check.cuh"
  #include <cuda_runtime.h>
#else
  #include "complexmat.hpp"
#endif

#include "cnfeat.hpp"
#include "fft.h"

struct BBox_c
{
    double cx, cy, w, h, a;

    inline void scale(double factor)
    {
        cx *= factor;
        cy *= factor;
        w  *= factor;
        h  *= factor;
    }

    inline void scale_x(double factor)
    {
        cx *= factor;
        w  *= factor;
    }

    inline void scale_y(double factor)
    {
        cy *= factor;
        h  *= factor;
    }

    inline cv::Rect get_rect()
    {
        return cv::Rect(cx-w/2., cy-h/2., w, h);
    }

};

class KCF_Tracker
{
public:
    bool m_debug {false};
    bool m_use_scale {true};
    bool m_use_angle {true};
    bool m_use_color {true};
#ifdef ASYNC
    bool m_use_multithreading {true};
#else
    bool m_use_multithreading {false};
#endif //ASYNC
    bool m_use_subpixel_localization {true};
    bool m_use_subgrid_scale {true};
    bool m_use_cnfeat {true};
    bool m_use_linearkernel {false};
#ifdef BIG_BATCH
    bool m_use_big_batch {true};
#else
    bool m_use_big_batch {false};
#endif
#ifdef BIG_BATCH
    bool m_use_cuda {true};
#else
    bool m_use_cuda {false};
#endif

    /*
    padding             ... extra area surrounding the target           (1.5)
    kernel_sigma        ... gaussian kernel bandwidth                   (0.5)
    lambda              ... regularization                              (1e-4)
    interp_factor       ... linear interpolation factor for adaptation  (0.02)
    output_sigma_factor ... spatial bandwidth (proportional to target)  (0.1)
    cell_size           ... hog cell size                               (4)
    */
    KCF_Tracker(double padding, double kernel_sigma, double lambda, double interp_factor, double output_sigma_factor, int cell_size);
    KCF_Tracker();
    ~KCF_Tracker();

    // Init/re-init methods
    void init(cv::Mat & img, const cv::Rect & bbox, int fit_size_x, int fit_size_y);
    void setTrackerPose(BBox_c & bbox, cv::Mat & img, int fit_size_x, int fit_size_y);
    void updateTrackerPosition(BBox_c & bbox);

    // frame-to-frame object tracking
    void track(cv::Mat & img);
    BBox_c getBBox();

private:
    Fft &fft;

    BBox_c p_pose;
    bool p_resize_image = false;
    bool p_fit_to_pw2 = false;

    bool first = true;

    const double p_downscale_factor = 0.5;
    double p_scale_factor_x = 1;
    double p_scale_factor_y = 1;

    double p_padding = 1.5;
    double p_output_sigma_factor = 0.1;
    double p_output_sigma;
    double p_kernel_sigma = 0.5;    //def = 0.5
    double p_lambda = 1e-4;         //regularization in learning step
    double p_interp_factor = 0.02;  //def = 0.02, linear interpolation factor for adaptation
    int p_cell_size = 4;            //4 for hog (= bin_size)
    int p_windows_size[2];
    int p_num_scales {5};
    double p_scale_step = 1.02;
    double p_current_scale = 1.;
    double p_min_max_scale[2];
    std::vector<double> p_scales;
    int p_num_angles {5};
    int p_current_angle = 0;
    int p_angle_step = 10;
    std::vector<double> p_angles;

    //for big batch
    int p_num_of_feats;
    int p_roi_height, p_roi_width;
    float *xf_sqr_norm = nullptr, *yf_sqr_norm = nullptr;
#ifdef CUFFT
    float *xf_sqr_norm_d = nullptr, *yf_sqr_norm_d = nullptr, *gauss_corr_res = nullptr;
#endif

    //model
    ComplexMat p_yf;
    ComplexMat p_model_alphaf;
    ComplexMat p_model_alphaf_num;
    ComplexMat p_model_alphaf_den;
    ComplexMat p_model_xf;
    //helping functions
    cv::Mat get_subwindow(const cv::Mat & input, int cx, int cy, int size_x, int size_y, int angle);
    cv::Mat gaussian_shaped_labels(double sigma, int dim1, int dim2);
    ComplexMat gaussian_correlation(const ComplexMat & xf, const ComplexMat & yf, double sigma, bool auto_correlation = false);
    cv::Mat circshift(const cv::Mat & patch, int x_rot, int y_rot);
    cv::Mat cosine_window_function(int dim1, int dim2);
    std::vector<cv::Mat> get_features(cv::Mat & input_rgb, cv::Mat & input_gray, int cx, int cy, int size_x, int size_y, double scale = 1., int angle = 0);
    cv::Point2f sub_pixel_peak(cv::Point & max_loc, cv::Mat & response);
    double sub_grid_scale(std::vector<double> & responses, int index = -1);

};

#endif //KCF_HEADER_6565467831231
