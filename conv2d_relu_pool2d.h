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

#if __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

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
        //return false;
      }
    }
    return true;
  }

private: // NOLINT
  std::vector<float> buffer_;
  std::vector<int64_t> shape_;
};

 // Use the command `lscpu` to query the L3 cache size of CPU, Xeon Silver 4210 CPU's is 14080K.
const int64_t L3_CACHE_SIZE = 14080*1024;

class Workspace {
 public:
  static Workspace* Instance() {
    static Workspace* instance = new Workspace();
    return instance;
  }
  Workspace() {
    Resize(L3_CACHE_SIZE);
  }
  ~Workspace() {}

  void Resize(const size_t size) {
    if (size > buffer_.size()) {
      buffer_.resize(size);
    }
  }

  void *data() {
    return buffer_.data();
  }

  size_t size() const { return buffer_.size(); }

private:
  std::vector<int8_t> buffer_;
};
