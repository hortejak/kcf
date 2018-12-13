#ifndef COMPLEX_MAT_HPP_213123048309482094
#define COMPLEX_MAT_HPP_213123048309482094

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <functional>
#include "dynmem.hpp"
#include "pragmas.h"
#include "prem.hpp"

#ifdef CUFFT
#include <cufft.h>
#endif

template <int C, int R, int CH, int S=1>
class ComplexMat_ {
  public:
    typedef float T;

    static constexpr uint cols = C;
    static constexpr uint rows = R;
    static constexpr uint n_channels = CH;
    static constexpr uint n_scales = S;

    template<int, int, int, int> friend class ComplexMat_;

    ComplexMat_(uint _rows, uint _cols, uint _n_channels, uint _n_scales = 1)
        : //cols(_cols), rows(_rows), n_channels(_n_channels * _n_scales), n_scales(_n_scales),
          p_data(n_channels * cols * rows) {}
    ComplexMat_(cv::Size size, uint _n_channels, uint _n_scales = 1)
        : //cols(size.width), rows(size.height), n_channels(_n_channels * _n_scales), n_scales(_n_scales),
          p_data(n_channels * cols * rows) {}

    // assuming that mat has 2 channels (real, img)
    ComplexMat_(const cv::Mat &mat) : //cols(uint(mat.cols)), rows(uint(mat.rows)), n_channels(1), n_scales(1),
                                      p_data(n_channels * cols * rows)
    {
        cudaSync();
        memcpy(p_data.hostMem(), mat.ptr<std::complex<T>>(), mat.total() * mat.elemSize());
    }

    static ComplexMat_ same_size(const ComplexMat_ &o)
    {
        return ComplexMat_(o.rows, o.cols, o.n_channels / o.n_scales, o.n_scales);
    }

    // cv::Mat API compatibility
    cv::Size size() const { return cv::Size(cols, rows); }
    uint channels() const { return n_channels; }

    // assuming that mat has 2 channels (real, imag)
    void set_channel(uint idx, const cv::Mat &mat)
    {
        assert(idx < n_channels);
        cudaSync();
        for (uint i = 0; i < rows; ++i) {
            const std::complex<T> *row = mat.ptr<std::complex<T>>(i);
            for (uint j = 0; j < cols; ++j)
                p_data.hostMem()[idx * rows * cols + i * cols + j] = row[j];
        }
    }

    T sqr_norm() const
    {
        assert(n_scales == 1);

        int n_channels_per_scale = n_channels / n_scales;
        T sum_sqr_norm = 0;
        for (int i = 0; i < n_channels_per_scale; ++i) {
            for (auto lhs = p_data.hostMem() + i * rows * cols; lhs != p_data.hostMem() + (i + 1) * rows * cols; ++lhs)
                sum_sqr_norm += lhs->real() * lhs->real() + lhs->imag() * lhs->imag();
        }
        sum_sqr_norm = sum_sqr_norm / static_cast<T>(cols * rows);
        return sum_sqr_norm;
    }

    void sqr_norm(DynMem_<T> &result) const
    {
        int n_channels_per_scale = n_channels / n_scales;
        int scale_offset = n_channels_per_scale * rows * cols;
        for (uint scale = 0; scale < n_scales; ++scale) {
            T sum_sqr_norm = 0;
            for (int i = 0; i < n_channels_per_scale; ++i)
                for (auto lhs = p_data.hostMem() + i * rows * cols + scale * scale_offset;
                     lhs != p_data.hostMem() + (i + 1) * rows * cols + scale * scale_offset; ++lhs)
                    sum_sqr_norm += lhs->real() * lhs->real() + lhs->imag() * lhs->imag();
            result.hostMem()[scale] = sum_sqr_norm / static_cast<T>(cols * rows);
        }
        return;
    }

    ComplexMat_ sqr_mag() const
    {
        return mat_const_operator([](std::complex<T> &c) { c = c.real() * c.real() + c.imag() * c.imag(); });
    }

    ComplexMat_ conj() const
    {
        return mat_const_operator([](std::complex<T> &c) { c = std::complex<T>(c.real(), -c.imag()); });
    }

    ComplexMat_<cols,rows,1,n_scales> sum_over_channels() const
    {
        assert(p_data.num_elem == n_channels * rows * cols);

        uint n_channels_per_scale = n_channels / n_scales;
        uint scale_offset = n_channels_per_scale * rows * cols;

        ComplexMat_<cols,rows,1,n_scales> result(this->rows, this->cols, 1, n_scales);
        for (uint scale = 0; scale < n_scales; ++scale) {
            for (uint i = 0; i < rows * cols; ++i) {
                std::complex<T> acc = 0;
                for (uint ch = 0; ch < n_channels_per_scale; ++ch)
                    acc +=  p_data[scale * scale_offset + i + ch * rows * cols];
                result.p_data.hostMem()[scale * rows * cols + i] = acc;
            }
        }
        return result;
    }

    // return 2 channels (real, imag) for first complex channel
    cv::Mat to_cv_mat() const
    {
        assert(p_data.num_elem >= 1);
        return channel_to_cv_mat(0);
    }
    // return a vector of 2 channels (real, imag) per one complex channel
    std::vector<cv::Mat> to_cv_mat_vector() const
    {
        std::vector<cv::Mat> result;
        result.reserve(n_channels);

        for (uint i = 0; i < n_channels; ++i)
            result.push_back(channel_to_cv_mat(i));

        return result;
    }

    std::complex<T> *get_p_data() {
        cudaSync();
        return p_data.hostMem();
    }
    const std::complex<T> *get_p_data() const {
        cudaSync();
        return p_data.hostMem();
    }

#ifdef CUFFT
    cufftComplex *get_dev_data() { return (cufftComplex*)p_data.deviceMem(); }
    const cufftComplex *get_dev_data() const { return (cufftComplex*)p_data.deviceMem(); }
#endif

    // element-wise per channel multiplication, division and addition
    ComplexMat_ operator*(const ComplexMat_ &rhs) const
    { return mat_mat_operator([](std::complex<T> &c_lhs, const std::complex<T> &c_rhs) { c_lhs *= c_rhs; }, rhs); }
    ComplexMat_ operator/(const ComplexMat_ &rhs) const
    { return mat_mat_operator([](std::complex<T> &c_lhs, const std::complex<T> &c_rhs) { c_lhs /= c_rhs; }, rhs); }
    ComplexMat_ operator+(const ComplexMat_ &rhs) const
    { return mat_mat_operator([](std::complex<T> &c_lhs, const std::complex<T> &c_rhs) { c_lhs += c_rhs; }, rhs); }

    // multiplying or adding constant
    ComplexMat_ operator*(const T &rhs) const
    { return mat_const_operator([&rhs](std::complex<T> &c) { c *= rhs; }); }
    ComplexMat_ operator+(const T &rhs) const
    { return mat_const_operator([&rhs](std::complex<T> &c) { c += rhs; }); }

    // multiplying element-wise multichannel by one channel mats (rhs mat is with one channel)
    ComplexMat_ mul(const ComplexMat_<complexmat_w, complexmat_h, 1, n_scales> &rhs) const
    {
        return matn_mat1_operator([](std::complex<T> &c_lhs, const std::complex<T> &c_rhs) { c_lhs *= c_rhs; }, rhs);
    }

    // multiplying element-wise multichannel mats - same as operator*(ComplexMat), but without allocating memory for the result
    ComplexMat_ muln(const ComplexMat_ &rhs) const
    {
        return mat_mat_operator([](std::complex<T> &c_lhs, const std::complex<T> &c_rhs) { c_lhs *= c_rhs; }, rhs);
    }

    // text output
    friend std::ostream &operator<<(std::ostream &os, const ComplexMat_ &mat)
    {
        // for (int i = 0; i < mat.n_channels; ++i){
        for (int i = 0; i < 1; ++i) {
            os << "Channel " << i << std::endl;
            for (uint j = 0; j < mat.rows; ++j) {
                for (uint k = 0; k < mat.cols - 1; ++k)
                    os << mat.p_data[j * mat.cols + k] << ", ";
                os << mat.p_data[j * mat.cols + mat.cols - 1] << std::endl;
            }
        }
        return os;
    }

  private:
    DynMem_<std::complex<T>> p_data;

    // convert 2 channel mat (real, imag) to vector row-by-row
    std::vector<std::complex<T>> convert(const cv::Mat &mat)
    {
        std::vector<std::complex<T>> result;
        result.reserve(mat.cols * mat.rows);
        for (int y = 0; y < mat.rows; ++y) {
            const T *row_ptr = mat.ptr<T>(y);
            for (int x = 0; x < 2 * mat.cols; x += 2) {
                result.push_back(std::complex<T>(row_ptr[x], row_ptr[x + 1]));
            }
        }
        return result;
    }

    ComplexMat_ mat_mat_operator(void (*op)(std::complex<T> &c_lhs, const std::complex<T> &c_rhs),
                                 const ComplexMat_ &mat_rhs) const
    {
        assert(mat_rhs.n_channels == n_channels/n_scales && mat_rhs.cols == cols && mat_rhs.rows == rows);

        ComplexMat_ result = *this;
        for (uint s = 0; s < n_scales; ++s) {
            auto lhs = result.p_data.hostMem() + (s * n_channels/n_scales * rows * cols);
            auto rhs = mat_rhs.p_data.hostMem();
            for (uint i = 0; i < n_channels/n_scales * rows * cols; ++i)
                op(*(lhs + i), *(rhs + i));
        }

        return result;
    }
    ComplexMat_ matn_mat1_operator(void (*op)(std::complex<T> &c_lhs, const std::complex<T> &c_rhs),
                                   const ComplexMat_ &mat_rhs) const
    {
        assert(mat_rhs.n_channels == 1 && mat_rhs.cols == cols && mat_rhs.rows == rows);

        ComplexMat_ result = *this;
        for (uint i = 0; i < n_channels; ++i) {
            auto lhs = result.p_data.hostMem() + i * rows * cols;
            auto rhs = mat_rhs.p_data.hostMem();
            for (; lhs != result.p_data.hostMem() + (i + 1) * rows * cols; ++lhs, ++rhs)
                op(*lhs, *rhs);
        }

        return result;
    }
    ComplexMat_ matn_mat2_operator(void (*op)(std::complex<T> &c_lhs, const std::complex<T> &c_rhs),
                                   const ComplexMat_ &mat_rhs) const
    {
        assert(mat_rhs.n_channels == n_channels / n_scales && mat_rhs.cols == cols && mat_rhs.rows == rows);

        int n_channels_per_scale = n_channels / n_scales;
        int scale_offset = n_channels_per_scale * rows * cols;
        ComplexMat_ result = *this;
        for (uint i = 0; i < n_scales; ++i) {
            for (int j = 0; j < n_channels_per_scale; ++j) {
                auto lhs = result.p_data.hostMem() + (j * rows * cols) + (i * scale_offset);
                auto rhs = mat_rhs.p_data.hostMem() + (j * rows * cols);
                for (; lhs != result.p_data.hostMem() + ((j + 1) * rows * cols) + (i * scale_offset); ++lhs, ++rhs)
                    op(*lhs, *rhs);
            }
        }

        return result;
    }
    ComplexMat_ mat_const_operator(const std::function<void(std::complex<T> &c_rhs)> &op) const
    {
        ComplexMat_ result = *this;
        for (uint i = 0; i < n_channels; ++i) {
            for (auto lhs = result.p_data.hostMem() + i * rows * cols;
                 lhs != result.p_data.hostMem() + (i + 1) * rows * cols; ++lhs)
                op(*lhs);
        }
        return result;
    }

    cv::Mat channel_to_cv_mat(int channel_id) const
    {
        cv::Mat result(rows, cols, CV_32FC2);
        for (uint y = 0; y < rows; ++y) {
            std::complex<T> *row_ptr = result.ptr<std::complex<T>>(y);
            for (uint x = 0; x < cols; ++x) {
                row_ptr[x] = p_data[channel_id * rows * cols + y * cols + x];
            }
        }
        return result;
    }

#ifdef CUFFT
    void cudaSync() const;
#else
    void cudaSync() const {}
#endif
};

template <int CH, int S=1>
using ComplexMat = ComplexMat_<complexmat_w, complexmat_h, CH, S>;

#endif // COMPLEX_MAT_HPP_213123048309482094
