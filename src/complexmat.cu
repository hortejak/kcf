#include "complexmat.hpp"

__global__ void sqr_norm_kernel(int n, float *out, const float *data, float rows, float cols)
{
    extern __shared__ float sdata[];
    int i = blockDim.x * threadIdx.y + threadIdx.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    sdata[i] = 0;
    sdata[i] = data[threadId] * data[threadId] + data[threadId + 1] * data[threadId + 1];
    __syncthreads();

    for (unsigned int s = (blockDim.x * blockDim.y + 1) / 2, old_s = blockDim.x * blockDim.y; s > 0; s >>= 1) {

        if (old_s & 1) s += 1;

        if (i < s && i + s < old_s) {
            sdata[i] += sdata[i + s];
        }
        old_s = s;
        __syncthreads();
    }

    if (i == 0) {
        atomicAdd(&out[blockId / n], sdata[0] / (rows * cols));
    }
}

void ComplexMat::sqr_norm(DynMem &result) const
{
    CudaSafeCall(cudaMemsetAsync(result.deviceMem(), 0, n_scales * sizeof(float)));

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);

    sqr_norm_kernel<<<numBlocks, threadsPerBlock, rows * cols * sizeof(float)>>>(
        n_channels / n_scales, result.deviceMem(), (float*)this->p_data.deviceMem(), rows, cols);
    CudaCheckError();

    return;
}

__global__ void sqr_mag_kernel(const float *data, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = data[threadId] * data[threadId] + data[threadId + 1] * data[threadId + 1];
    result[threadId + 1] = 0;
}

ComplexMat ComplexMat::sqr_mag() const
{
    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    sqr_mag_kernel<<<numBlocks, threadsPerBlock, 0>>>((float*)this->p_data.deviceMem(), (float*)result.p_data.deviceMem());
    CudaCheckError();

    return result;
}

__global__ void conj_kernel(const float *data, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = data[threadId];
    result[threadId + 1] = -data[threadId + 1];
}

ComplexMat ComplexMat::conj() const
{
    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    conj_kernel<<<numBlocks, threadsPerBlock, 0>>>((float*)this->p_data.deviceMem(), (float*)result.p_data.deviceMem());
    CudaCheckError();

    return result;
}

__global__ static void sum_channels(float *dest, const float *src, uint channels, uint num_channel_elem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_channel_elem)
        return;

    float acc = 0;
    for (uint i = 0; i < channels; ++i)
        acc += src[idx + i * num_channel_elem];
    dest[idx] = acc;
}

ComplexMat ComplexMat::sum_over_channels() const
{
    assert(p_data.num_elem == n_channels * rows * cols);

    uint n_channels_per_scale = n_channels / n_scales;
    uint scale_offset = n_channels_per_scale * rows * cols;

    ComplexMat_ result(this->rows, this->cols, 1, n_scales);

    const uint total = rows * cols * 2;
    const dim3 threads(256);
    const dim3 blocks((total + threads.x - 1) / threads.x);

    for (uint scale = 0; scale < n_scales; ++scale) {
        sum_channels<<<blocks, threads>>>(reinterpret_cast<float*>(result.p_data.deviceMem() + scale * scale_offset),
                                          reinterpret_cast<const float*>(p_data.deviceMem() + scale * scale_offset),
                                          n_channels_per_scale, total);
    }
    CudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
    return result;
}

__global__ void same_num_channels_mul_kernel(const float *data_l, const float *data_r, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = data_l[threadId] * data_r[threadId] - data_l[threadId + 1] * data_r[threadId + 1];
    result[threadId + 1] = data_l[threadId] * data_r[threadId + 1] + data_l[threadId + 1] * data_r[threadId];
}

// element-wise per channel multiplication, division and addition
ComplexMat ComplexMat::operator*(const ComplexMat &rhs) const
{
    assert(rhs.n_channels == n_channels && rhs.cols == cols && rhs.rows == rows);

    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    same_num_channels_mul_kernel<<<numBlocks, threadsPerBlock, 0>>>((float*)this->p_data.deviceMem(),
                                                                    (float*)rhs.p_data.deviceMem(),
                                                                    (float*)result.p_data.deviceMem());
    CudaCheckError();

    return result;
}

__global__ void same_num_channels_div_kernel(const float *data_l, const float *data_r, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = (data_l[threadId] * data_r[threadId] + data_l[threadId + 1] * data_r[threadId + 1]) /
                       (data_r[threadId] * data_r[threadId] + data_r[threadId + 1] * data_r[threadId + 1]);
    result[threadId + 1] = (data_l[threadId + 1] * data_r[threadId] - data_l[threadId] * data_r[threadId + 1]) /
                           (data_r[threadId] * data_r[threadId] + data_r[threadId + 1] * data_r[threadId + 1]);
}

ComplexMat ComplexMat::operator/(const ComplexMat &rhs) const
{
    assert(rhs.n_channels == n_channels && rhs.cols == cols && rhs.rows == rows);

    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    same_num_channels_div_kernel<<<numBlocks, threadsPerBlock, 0>>>((float*)this->p_data.deviceMem(),
                                                                    (float*)rhs.p_data.deviceMem(),
                                                                    (float*)result.p_data.deviceMem());
    CudaCheckError();

    return result;
}

__global__ void same_num_channels_add_kernel(const float *data_l, const float *data_r, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = data_l[threadId] + data_r[threadId];
    result[threadId + 1] = data_l[threadId + 1] + data_r[threadId + 1];
}

ComplexMat ComplexMat::operator+(const ComplexMat &rhs) const
{
    assert(rhs.n_channels == n_channels && rhs.cols == cols && rhs.rows == rows);

    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    same_num_channels_add_kernel<<<numBlocks, threadsPerBlock, 0>>>((float*)this->p_data.deviceMem(),
                                                                    (float*)rhs.p_data.deviceMem(),
                                                                    (float*)result.p_data.deviceMem());
    CudaCheckError();

    return result;
}

__global__ void constant_mul_kernel(const float *data_l, float constant, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = data_l[threadId] * constant;
    result[threadId + 1] = data_l[threadId + 1] * constant;
}

ComplexMat ComplexMat::operator*(const float &rhs) const
{
    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    constant_mul_kernel<<<numBlocks, threadsPerBlock, 0>>>((float*)this->p_data.deviceMem(),
                                                           rhs,
                                                           (float*)result.p_data.deviceMem());
    CudaCheckError();

    return result;
}

__global__ void constant_add_kernel(const float *data_l, float constant, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = data_l[threadId] + constant;
    result[threadId + 1] = data_l[threadId + 1];
}

ComplexMat ComplexMat::operator+(const float &rhs) const
{
    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    constant_add_kernel<<<numBlocks, threadsPerBlock, 0>>>((float*)this->p_data.deviceMem(),
                                                           rhs,
                                                           (float*)result.p_data.deviceMem());
    CudaCheckError();

    return result;
}

__global__ void one_channel_mul_kernel(const float *data_l, const float *data_r, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);
    int one_ch_index = 2 * ((threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = data_l[threadId] * data_r[one_ch_index] - data_l[threadId + 1] * data_r[one_ch_index + 1];
    result[threadId + 1] = data_l[threadId] * data_r[one_ch_index + 1] + data_l[threadId + 1] * data_r[one_ch_index];
}

// multiplying element-wise multichannel by one channel mats (rhs mat is with one channel)
ComplexMat ComplexMat::mul(const ComplexMat &rhs) const
{
    assert(rhs.n_channels == 1 && rhs.cols == cols && rhs.rows == rows);

    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    one_channel_mul_kernel<<<numBlocks, threadsPerBlock, 0>>>((float*)this->p_data.deviceMem(),
                                                              (float*)rhs.p_data.deviceMem(),
                                                              (float*)result.p_data.deviceMem());
    CudaCheckError();

    return result;
}

__global__ void scales_channel_mul_kernel(float *data_l, float *data_r, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);
    int one_ch_index = 2 * ((threadIdx.y * blockDim.x) + threadIdx.x + blockIdx.x * blockDim.x * blockDim.y);

    result[threadId] = data_l[threadId] * data_r[one_ch_index] - data_l[threadId + 1] * data_r[one_ch_index + 1];
    result[threadId + 1] = data_l[threadId] * data_r[one_ch_index + 1] + data_l[threadId + 1] * data_r[one_ch_index];
}

// multiplying element-wise multichannel by one channel mats (rhs mat is with multiple channel)
// ComplexMat ComplexMat::mul2(const ComplexMat &rhs) const
// {
//     assert(rhs.n_channels == n_channels / n_scales && rhs.cols == cols && rhs.rows == rows);

//     ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

//     dim3 threadsPerBlock(rows, cols);
//     dim3 numBlocks(n_channels / n_scales, n_scales);
//     scales_channel_mul_kernel<<<numBlocks, threadsPerBlock, 0>>>(this->p_data, rhs.p_data, result.p_data);
//     CudaCheckError();

//     return result;
// }

// void ComplexMat::operator=(ComplexMat &&rhs)
// {
//     cols = rhs.cols;
//     rows = rhs.rows;
//     n_channels = rhs.n_channels;
//     n_scales = rhs.n_scales;

//     p_data = rhs.p_data;

//     rhs.p_data = nullptr;
// }
