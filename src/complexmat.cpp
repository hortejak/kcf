#include "complexmat.hpp"

ComplexMat::ComplexMat() : cols(0), rows(0), n_channels(0) {}
ComplexMat::ComplexMat(int _rows, int _cols, int _n_channels) : cols(_cols), rows(_rows), n_channels(_n_channels)
{
    p_data.resize(n_channels*cols*rows);
}


//assuming that mat has 2 channels (real, img)
ComplexMat::ComplexMat(const cv::Mat & mat) : cols(mat.cols), rows(mat.rows), n_channels(1)
{
    p_data = convert(mat);
}

void ComplexMat::create(int _rows, int _cols, int _n_channels)
{
    rows = _rows;
    cols = _cols;
    n_channels = _n_channels;
    p_data.resize(n_channels*cols*rows);
}

void ComplexMat::create(int _rows, int _cols, int _n_channels, int _n_scales)
{
    rows = _rows;
    cols = _cols;
    n_channels = _n_channels;
    n_scales = _n_scales;
    p_data.resize(n_channels*cols*rows);
}
// cv::Mat API compatibility
cv::Size ComplexMat::size() { return cv::Size(cols, rows); }
int ComplexMat::channels() { return n_channels; }
int ComplexMat::channels() const { return n_channels; }

//assuming that mat has 2 channels (real, imag)
void ComplexMat::set_channel(int idx, const cv::Mat & mat)
{
    assert(idx >= 0 && idx < n_channels);
    for (int i = 0; i < rows; ++i){
        const std::complex<float> *row = mat.ptr<std::complex<float>>(i);
        for (int j = 0; j < cols; ++j)
            p_data[idx*rows*cols+i*cols+j]=row[j];
    }
}


void ComplexMat::sqr_norm(float *sums_sqr_norms) const
{
    int n_channels_per_scale = n_channels/n_scales;
    int scale_offset = n_channels_per_scale*rows*cols;
    float sum_sqr_norm;
    for (int scale = 0; scale < n_scales; ++scale) {
        sum_sqr_norm = 0;
        for (int i = 0; i < n_channels_per_scale; ++i)
            for (auto lhs = p_data.begin()+i*rows*cols+scale*scale_offset; lhs != p_data.begin()+(i+1)*rows*cols+scale*scale_offset; ++lhs)
                sum_sqr_norm += lhs->real()*lhs->real() + lhs->imag()*lhs->imag();
        sums_sqr_norms[scale] = sum_sqr_norm/static_cast<float>(cols*rows);
    }
    return;
}

ComplexMat ComplexMat::sqr_mag() const
{
    return mat_const_operator( [](std::complex<float> & c) { c = c.real()*c.real() + c.imag()*c.imag(); } );
}

ComplexMat ComplexMat::conj() const
{
    return mat_const_operator( [](std::complex<float> & c) { c = std::complex<float>(c.real(), -c.imag()); } );
}

ComplexMat ComplexMat::sum_over_channels() const
{
    assert(p_data.size() > 1);

    int n_channels_per_scale = n_channels/n_scales;
    int scale_offset = n_channels_per_scale*rows*cols;

    ComplexMat result(this->rows, this->cols, n_scales);
    for (int scale = 0; scale < n_scales; ++scale) {
        std::copy(p_data.begin()+scale*scale_offset,p_data.begin()+rows*cols+scale*scale_offset, result.p_data.begin()+scale*rows*cols);
        for (int i = 1; i < n_channels_per_scale; ++i) {
            std::transform(result.p_data.begin()+scale*rows*cols, result.p_data.begin()+(scale+1)*rows*cols, p_data.begin()+i*rows*cols+scale*scale_offset,
                           result.p_data.begin()+scale*rows*cols, std::plus<std::complex<float>>());
        }
    }
    return result;
}

//return 2 channels (real, imag) for first complex channel
cv::Mat ComplexMat::to_cv_mat() const
{
    assert(p_data.size() >= 1);
    return channel_to_cv_mat(0);
}
// return a vector of 2 channels (real, imag) per one complex channel
std::vector<cv::Mat> ComplexMat::to_cv_mat_vector() const
{
    std::vector<cv::Mat> result;
    result.reserve(n_channels);

    for (int i = 0; i < n_channels; ++i)
        result.push_back(channel_to_cv_mat(i));

    return result;
}

std::complex<float>* ComplexMat::get_p_data() const
{
    return p_data.data();
}

//element-wise per channel multiplication, division and addition
ComplexMat ComplexMat::operator*(const ComplexMat & rhs) const
{
    return ComplexMat::mat_mat_operator( [](std::complex<float> & c_lhs, const std::complex<float> & c_rhs) { c_lhs *= c_rhs; }, rhs);
}
ComplexMat ComplexMat::operator/(const ComplexMat & rhs) const
{
    return ComplexMat::mat_mat_operator( [](std::complex<float> & c_lhs, const std::complex<float> & c_rhs) { c_lhs /= c_rhs; }, rhs);
}
ComplexMat ComplexMat::operator+(const ComplexMat & rhs) const
{
    return ComplexMat::mat_mat_operator( [](std::complex<float> & c_lhs, const std::complex<float> & c_rhs)  { c_lhs += c_rhs; }, rhs);
}

//multiplying or adding constant
ComplexMat ComplexMat::operator*(const float & rhs) const
{
    return ComplexMat::mat_const_operator( [&rhs](std::complex<float> & c) { c *= rhs; });
}
ComplexMat ComplexMat::operator+(const float & rhs) const
{
    return ComplexMat::mat_const_operator( [&rhs](std::complex<float> & c) { c += rhs; });
}

//multiplying element-wise multichannel by one channel mats (rhs mat is with one channel)
ComplexMat ComplexMat::mul(const ComplexMat & rhs) const
{
    return ComplexMat::matn_mat1_operator( [](std::complex<float> & c_lhs, const std::complex<float> & c_rhs) { c_lhs *= c_rhs; }, rhs);
}

//multiplying element-wise multichannel by one channel mats (rhs mat is with multiple channel)
ComplexMat ComplexMat::mul2(const ComplexMat & rhs) const
{
    return ComplexMat::matn_mat2_operator( [](std::complex<float> & c_lhs, const std::complex<float> & c_rhs) { c_lhs *= c_rhs; }, rhs);
}


//convert 2 channel mat (real, imag) to vector row-by-row
std::vector<std::complex<float>> ComplexMat::convert(const cv::Mat & mat)
{
    std::vector<std::complex<float>> result;
    result.reserve(mat.cols*mat.rows);
    for (int y = 0; y < mat.rows; ++y) {
        const float * row_ptr = mat.ptr<float>(y);
        for (int x = 0; x < 2*mat.cols; x += 2){
            result.push_back(std::complex<float>(row_ptr[x], row_ptr[x+1]));
        }
    }
    return result;
}

ComplexMat ComplexMat::mat_mat_operator(void (*op)(std::complex<float> & c_lhs, const std::complex<float> & c_rhs), const ComplexMat & mat_rhs) const
{
    assert(mat_rhs.n_channels == n_channels && mat_rhs.cols == cols && mat_rhs.rows == rows);

    ComplexMat result = *this;
    for (int i = 0; i < n_channels; ++i) {
        auto lhs = result.p_data.begin()+i*rows*cols;
        auto rhs = mat_rhs.p_data.begin()+i*rows*cols;
        for ( ; lhs != result.p_data.begin()+(i+1)*rows*cols; ++lhs, ++rhs)
            op(*lhs, *rhs);
    }

    return result;
}
ComplexMat ComplexMat::matn_mat1_operator(void (*op)(std::complex<float> & c_lhs, const std::complex<float> & c_rhs), const ComplexMat & mat_rhs) const
{
    assert(mat_rhs.n_channels == 1 && mat_rhs.cols == cols && mat_rhs.rows == rows);

    ComplexMat result = *this;
    for (int i = 0; i < n_channels; ++i) {
        auto lhs = result.p_data.begin()+i*rows*cols;
        auto rhs = mat_rhs.p_data.begin();
        for ( ; lhs != result.p_data.begin()+(i+1)*rows*cols; ++lhs, ++rhs)
            op(*lhs, *rhs);
    }

    return result;
}
ComplexMat ComplexMat::matn_mat2_operator(void (*op)(std::complex<float> & c_lhs, const std::complex<float> & c_rhs), const ComplexMat & mat_rhs) const
{
    assert(mat_rhs.n_channels == n_channels/n_scales && mat_rhs.cols == cols && mat_rhs.rows == rows);

    int n_channels_per_scale = n_channels/n_scales;
    int scale_offset = n_channels_per_scale*rows*cols;
    ComplexMat result = *this;
    for (int i = 0; i < n_scales; ++i) {
        for (int j = 0; j < n_channels_per_scale; ++j) {
            auto lhs = result.p_data.begin()+(j*rows*cols)+(i*scale_offset);
            auto rhs = mat_rhs.p_data.begin()+(j*rows*cols);
            for ( ; lhs != result.p_data.begin()+((j+1)*rows*cols)+(i*scale_offset); ++lhs, ++rhs)
                op(*lhs, *rhs);
        }
    }

    return result;
}
ComplexMat ComplexMat::mat_const_operator(const std::function<void(std::complex<float> & c_rhs)> & op) const
{
    ComplexMat result = *this;
    for (int i = 0; i < n_channels; ++i)
        for (auto lhs = result.p_data.begin()+i*rows*cols; lhs != result.p_data.begin()+(i+1)*rows*cols; ++lhs)
            op(*lhs);
    return result;
}

cv::Mat ComplexMat::channel_to_cv_mat(int channel_id) const
{
    cv::Mat result(rows, cols, CV_32FC2);
    for (int y = 0; y < rows; ++y) {
        std::complex<float> * row_ptr = result.ptr<std::complex<float>>(y);
        for (int x = 0; x < cols; ++x){
            row_ptr[x] = p_data[channel_id*rows*cols+y*cols+x];
        }
    }
    return result;
}
