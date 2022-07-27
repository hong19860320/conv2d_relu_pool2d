// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <limits>
#include <random>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

const int MAX_OMP_THEADS_NUM = 4;

int64_t shape_production(const std::vector<int64_t> &shape) {
  int64_t product = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    product *= shape[i];
  }
  return product;
}

template <typename T>
inline void fill_data_with_random(T *dio, T vstart, T vend, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 1.f);
  for (size_t i = 0; i < size; ++i) {
    dio[i] = static_cast<T>(vstart + (vend - vstart) * dis(gen));
  }
}

inline int64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

class Tensor {
public: // NOLINT
  Tensor() {}
  explicit Tensor(const std::vector<int64_t> &shape) : shape_(shape) {}
  ~Tensor() {}

  void Resize(const std::vector<int64_t> &shape) { shape_ = shape; }

  float *data() {
    auto size = shape_production(shape_);
    if (size > buffer_.size()) {
      buffer_.resize(size);
    }
    return buffer_.data();
  }

  int64_t rank() const { return shape_.size(); }

  int64_t size() const { return shape_production(shape_); }

  std::vector<int64_t> shape() const { return shape_; }

  std::vector<int64_t> strides() const {
    int64_t r = rank();
    std::vector<int64_t> ss(r);
    ss[r - 1] = 1;
    for (int64_t i = r - 2; i >= 0; i--) {
      ss[i] = ss[i + 1] * shape_[i + 1];
    }
    return ss;
  }

  bool operator==(Tensor &tensor) { // NOLINT
    int64_t r = rank();
    if (tensor.rank() != r)
      return false;
    for (int64_t i = 0; i < r; i++) {
      if (shape_[i] != tensor.shape()[i])
        return false;
    }
    int64_t s = size();
    if (tensor.size() != s)
      return false;
    for (int64_t i = 0; i < s; i++) {
      double a = buffer_[i];
      double b = tensor.data()[i];
      const double threshold = 1e-5f;
      if (fabs(a - b) >
          std::max(threshold * std::max(fabs(a), fabs(b)), threshold)) {
        auto ss = strides();
        printf("[");
        int64_t remain = i;
        for (int64_t j = 0; j < r; j++) {
          auto coord = remain / ss[j];
          remain %= ss[j];
          printf("%ld,", coord);
        }
        printf("] %f, %f\n", a, b);
        return false;
      }
    }
    return true;
  }

private: // NOLINT
  std::vector<float> buffer_;
  std::vector<int64_t> shape_;
};

typedef void (*Conv2DRELUPool2DFunction)(
    Tensor *input_tensor, Tensor *weight_tensor, Tensor *bias_tensor,
    Tensor *output_tensor, int64_t *calc_start_time, int64_t *calc_end_time);
void eval_conv2d_relu_pool2d(const char *label,
                             Conv2DRELUPool2DFunction conv2d_relu_pool2d_func,
                             Tensor *input_tensor, Tensor *weight_tensor,
                             Tensor *bias_tensor, Tensor *output_tensor) {
  double max = 0.0f;
  double min = std::numeric_limits<float>::max();
  double total = 0.0f;
  const int repeats = 5;
  // printf("%s start evaluation ...\n", label);
  for (int i = 0; i < repeats; i++) {
    int64_t start, end;
    conv2d_relu_pool2d_func(input_tensor, weight_tensor, bias_tensor,
                            output_tensor, &start, &end);
    double elapse = (end - start) / 1000.0f;
    // printf("iter %d cost: %f ms\n", i, elapse);
    if (elapse > max) {
      max = elapse;
    }
    if (elapse < min) {
      min = elapse;
    }
    total += elapse;
  }
  printf("%s average: %f ms, max: %f ms, min: %f ms\n", label, total / repeats,
         max, min);
}

#define EXTRACT_INPUTS_OUTPUTS_PARAMS                                          \
  std::vector<int64_t> input_shape = input_tensor->shape();                    \
  std::vector<int64_t> weight_shape = weight_tensor->shape();                  \
  std::vector<int64_t> bias_shape = bias_tensor->shape();                      \
  int batch_size = input_shape[0];                                             \
  int input_channel_size = input_shape[1];                                     \
  int input_height = input_shape[2];                                           \
  int input_width = input_shape[3];                                            \
  int output_channel_size = weight_shape[0];                                   \
  int output_height = input_height;                                            \
  int output_width = input_width;                                              \
  std::vector<int64_t> output_shape = {batch_size, output_channel_size,        \
                                       output_height, output_width};           \
  output_tensor->Resize(output_shape);                                         \
  auto input_data = input_tensor->data();                                      \
  auto weight_data = weight_tensor->data();                                    \
  auto bias_data = bias_tensor->data();                                        \
  auto output_data = output_tensor->data();                                    \
  auto output_size = output_tensor->size();                                    \
  auto input_strides = input_tensor->strides();                                \
  auto weight_strides = weight_tensor->strides();                              \
  auto output_strides = output_tensor->strides();

void conv2d_3x3s1p1d1_relu_pool2d_avg_3x3s1p1d1_ref(
    Tensor *input_tensor, Tensor *weight_tensor, Tensor *bias_tensor,
    Tensor *output_tensor, int64_t *calc_start_time, int64_t *calc_end_time) {
  EXTRACT_INPUTS_OUTPUTS_PARAMS

  Tensor conv2d_output_tensor(output_shape);
  Tensor relu_output_tensor(output_shape);
  auto conv2d_output_data = conv2d_output_tensor.data();
  auto relu_output_data = relu_output_tensor.data();
  *calc_start_time = get_current_us();
  // Conv2D
  //   FLOPs = 2 * batch_size * output_channel_size * output_height *
  //   output_width * input_channel_size * 3 * 3
  //   Load Insts = 2 * batch_size * output_channel_size * output_height *
  //   output_width * input_channel_size * 3 * 3
  //   Write Insts = batch_size * output_channel_size * output_height *
  //   output_width
  for (int bs = 0; bs < batch_size; bs++) {
    for (int oc = 0; oc < output_channel_size; oc++) {
      for (int oh = 0; oh < output_height; oh++) {
        for (int ow = 0; ow < output_width; ow++) {
          float result = bias_data[oc];
          for (int ic = 0; ic < input_channel_size; ic++) {
            for (int kh = 0; kh < 3; kh++) {
              for (int kw = 0; kw < 3; kw++) {
                int iw = ow + kw - 1;
                int ih = oh + kh - 1;
                auto iidx = bs * input_strides[0] + ic * input_strides[1] +
                            ih * input_strides[2] + iw * input_strides[3];
                auto widx = oc * weight_strides[0] + ic * weight_strides[1] +
                            kh * weight_strides[2] + kw * weight_strides[3];
                result +=
                    iw < 0 || iw >= input_width || ih < 0 || ih >= input_height
                        ? 0.0f
                        : input_data[iidx] * weight_data[widx];
              }
            }
          }
          auto idx = bs * output_strides[0] + oc * output_strides[1] +
                     oh * output_strides[2] + ow * output_strides[3];
          conv2d_output_data[idx] = result;
        }
      }
    }
  }
  // RELU
  //   FLOPs = batch_size * output_channel_size * output_height * output_width
  //   Load Insts = batch_size * output_channel_size * output_height *
  //   output_width
  //   Write Insts = batch_size * output_channel_size * output_height *
  //   output_width
  for (int i = 0; i < output_size; i++) {
    auto result = conv2d_output_data[i];
    relu_output_data[i] = result > 0.0f ? result : 0.0f;
  }
  // Pool2D
  //   FLOPs = batch_size * output_channel_size * output_height * output_width *
  //   3 * 3 + 1
  //   Load Insts = batch_size * output_channel_size * output_height *
  //   output_width * 3 * 3
  //   Write Insts = batch_size * output_channel_size * output_height *
  //   output_width
  for (int bs = 0; bs < batch_size; bs++) {
    for (int oc = 0; oc < output_channel_size; oc++) {
      for (int oh = 0; oh < output_height; oh++) {
        for (int ow = 0; ow < output_width; ow++) {
          float result = 0.0f;
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              int iw = ow + kw - 1;
              int ih = oh + kh - 1;
              auto idx = bs * output_strides[0] + oc * output_strides[1] +
                         ih * output_strides[2] + iw * output_strides[3];
              result +=
                  iw < 0 || iw >= output_width || ih < 0 || ih >= output_height
                      ? 0.0f
                      : relu_output_data[idx];
            }
          }
          auto idx = bs * output_strides[0] + oc * output_strides[1] +
                     oh * output_strides[2] + ow * output_strides[3];
          output_data[idx] = result / 9.0f;
        }
      }
    }
  }
  *calc_end_time = get_current_us();
}

void conv2d_3x3s1p1d1_relu_pool2d_avg_3x3s1p1d1_conv2d_relu_pool2d_fused(
    Tensor *input_tensor, Tensor *weight_tensor, Tensor *bias_tensor,
    Tensor *output_tensor, int64_t *calc_start_time, int64_t *calc_end_time) {
  EXTRACT_INPUTS_OUTPUTS_PARAMS

  *calc_start_time = get_current_us();
  // Conv2D+RELU fused
  //   FLOPs = 2 * batch_size * output_channel_size * output_height *
  //   output_width * input_channel_size * 3 * 3
  //           + batch_size * output_channel_size * output_height * output_width
  //   Load Insts = 2 * batch_size * output_channel_size * output_height *
  //   output_width * input_channel_size * 3 * 3
  //   Write Insts = batch_size * output_channel_size * output_height *
  //   output_width
  // Pool2D (cache friendly)
  //   FLOPs = batch_size * output_channel_size * output_height * output_width *
  //   3 * 3
  //   Load Insts = batch_size * output_channel_size * output_height *
  //   output_width * 3 * 3
  //   Write Insts = batch_size * output_channel_size * output_height *
  //   output_width
  for (int bs = 0; bs < batch_size; bs++) {
    for (int oc = 0; oc < output_channel_size; oc++) {
      float block[3][output_width + 2];
      memset(&block[0][0], 0, sizeof(float) * 3 * (output_width + 2));
      for (int oh = 0; oh < output_height + 1; oh++) {
        if (oh == output_height) {
          memset(&block[2][1], 0, sizeof(float) * output_width);
        } else {
          for (int ow = 0; ow < output_width; ow++) {
            float result = bias_data[oc];
            for (int ic = 0; ic < input_channel_size; ic++) {
              for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                  int iw = ow + kw - 1;
                  int ih = oh + kh - 1;
                  auto iidx = bs * input_strides[0] + ic * input_strides[1] +
                              ih * input_strides[2] + iw * input_strides[3];
                  auto widx = oc * weight_strides[0] + ic * weight_strides[1] +
                              kh * weight_strides[2] + kw * weight_strides[3];
                  result += iw < 0 || iw >= input_width || ih < 0 ||
                                    ih >= input_height
                                ? 0.0f
                                : input_data[iidx] * weight_data[widx];
                }
              }
            }
            block[2][ow + 1] = result > 0.0f ? result : 0.0f;
          }
        }
        if (oh > 0) {
          for (int ow = 0; ow < output_width; ow++) {
            float result = block[0][ow] + block[0][ow + 1] + block[0][ow + 2];
            result += block[1][ow] + block[1][ow + 1] + block[1][ow + 2];
            result += block[2][ow] + block[2][ow + 1] + block[2][ow + 2];
            auto idx = bs * output_strides[0] + oc * output_strides[1] +
                       (oh - 1) * output_strides[2] + ow * output_strides[3];
            output_data[idx] = result / 9.0f;
          }
        }
        memcpy(&block[0][1], &block[1][1], output_width * sizeof(float));
        memcpy(&block[1][1], &block[2][1], output_width * sizeof(float));
      }
    }
  }
  *calc_end_time = get_current_us();
}

void conv2d_3x3s1p1d1_relu_pool2d_avg_3x3s1p1d1_conv2d_relu_pool2d_fused_omp(
    Tensor *input_tensor, Tensor *weight_tensor, Tensor *bias_tensor,
    Tensor *output_tensor, int64_t *calc_start_time, int64_t *calc_end_time) {
  EXTRACT_INPUTS_OUTPUTS_PARAMS

  *calc_start_time = get_current_us();
  // Using multi-threaded parallel computing in the channel dimension
  // Conv2D+RELU fused
  //   FLOPs = 2 * batch_size * output_channel_size * output_height *
  //   output_width * input_channel_size * 3 * 3
  //           + batch_size * output_channel_size * output_height * output_width
  //   Load Insts = 2 * batch_size * output_channel_size * output_height *
  //   output_width * input_channel_size * 3 * 3
  //   Write Insts = batch_size * output_channel_size * output_height *
  //   output_width
  // Pool2D (cache friendly)
  //   FLOPs = batch_size * output_channel_size * output_height * output_width *
  //   3 * 3
  //   Load Insts = batch_size * output_channel_size * output_height *
  //   output_width * 3 * 3
  //   Write Insts = batch_size * output_channel_size * output_height *
  //   output_width
  for (int bs = 0; bs < batch_size; bs++) {
#pragma omp parallel for num_threads(MAX_OMP_THEADS_NUM)
    for (int oc = 0; oc < output_channel_size; oc++) {
      float block[3][output_width + 2];
      memset(&block[0][0], 0, sizeof(float) * 3 * (output_width + 2));
      for (int oh = 0; oh < output_height + 1; oh++) {
        if (oh == output_height) {
          memset(&block[2][1], 0, sizeof(float) * output_width);
        } else {
          for (int ow = 0; ow < output_width; ow++) {
            float result = bias_data[oc];
            for (int ic = 0; ic < input_channel_size; ic++) {
              for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                  int iw = ow + kw - 1;
                  int ih = oh + kh - 1;
                  auto iidx = bs * input_strides[0] + ic * input_strides[1] +
                              ih * input_strides[2] + iw * input_strides[3];
                  auto widx = oc * weight_strides[0] + ic * weight_strides[1] +
                              kh * weight_strides[2] + kw * weight_strides[3];
                  result += iw < 0 || iw >= input_width || ih < 0 ||
                                    ih >= input_height
                                ? 0.0f
                                : input_data[iidx] * weight_data[widx];
                }
              }
            }
            block[2][ow + 1] = result > 0.0f ? result : 0.0f;
          }
        }
        if (oh > 0) {
          for (int ow = 0; ow < output_width; ow++) {
            float result = block[0][ow] + block[0][ow + 1] + block[0][ow + 2];
            result += block[1][ow] + block[1][ow + 1] + block[1][ow + 2];
            result += block[2][ow] + block[2][ow + 1] + block[2][ow + 2];
            auto idx = bs * output_strides[0] + oc * output_strides[1] +
                       (oh - 1) * output_strides[2] + ow * output_strides[3];
            output_data[idx] = result / 9.0f;
          }
        }
        memcpy(&block[0][1], &block[1][1], output_width * sizeof(float));
        memcpy(&block[1][1], &block[2][1], output_width * sizeof(float));
      }
    }
  }
  *calc_end_time = get_current_us();
}

int main(int argc, char **argv) {
  // Conv2D + RELU + Pool2D
  // Conv2D: stride=[1,1], pad=[1,1], dilation=[1,1], group=1
  // Pool2D: method=avg, kernel=[3,3], stride=[1,1], pad=[1,1], exclusive=false
  // Initialize input, weight tensor and fill their data with the random value.
  Tensor input_tensor({1, 16, 224, 224});
  fill_data_with_random(input_tensor.data(), -1.f, 1.f, input_tensor.size());
  Tensor weight_tensor({32, 16, 3, 3});
  fill_data_with_random(weight_tensor.data(), -1.f, 1.f, weight_tensor.size());
  Tensor bias_tensor({32});
  fill_data_with_random(bias_tensor.data(), -1.f, 1.f, bias_tensor.size());

  // Reference version
  Tensor ref_output_tensor;
  eval_conv2d_relu_pool2d("conv2d_3x3s1p1d1_relu_pool2d_avg_3x3s1p1d1_ref",
                          conv2d_3x3s1p1d1_relu_pool2d_avg_3x3s1p1d1_ref,
                          &input_tensor, &weight_tensor, &bias_tensor,
                          &ref_output_tensor);

  // Optimized with Conv2D+RELU+Pool2D fused
  Tensor conv2d_relu_pool2d_fused_output_tensor;
  eval_conv2d_relu_pool2d(
      "conv2d_3x3s1p1d1_relu_pool2d_avg_3x3s1p1d1_conv2d_relu_pool2d_fused",
      conv2d_3x3s1p1d1_relu_pool2d_avg_3x3s1p1d1_conv2d_relu_pool2d_fused,
      &input_tensor, &weight_tensor, &bias_tensor,
      &conv2d_relu_pool2d_fused_output_tensor);

  // Optimized with Conv2D+RELU+Pool2D fused + OpenMP
  Tensor conv2d_relu_pool2d_fused_omp_output_tensor;
  eval_conv2d_relu_pool2d(
      "conv2d_3x3s1p1d1_relu_pool2d_avg_3x3s1p1d1_conv2d_relu_pool2d_fused_omp",
      conv2d_3x3s1p1d1_relu_pool2d_avg_3x3s1p1d1_conv2d_relu_pool2d_fused_omp,
      &input_tensor, &weight_tensor, &bias_tensor,
      &conv2d_relu_pool2d_fused_omp_output_tensor);
  return 0;
}
