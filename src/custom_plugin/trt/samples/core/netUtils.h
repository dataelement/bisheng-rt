#ifndef TRT_NET_UTILS_H
#define TRT_NET_UTILS_H

#include <fstream>
#include <iostream>
#include "fileops.hpp"
#include "common.h"

class Timer {
 public:
  Timer() {
    _tstart = std::chrono::system_clock::now();
  }
  ~Timer() {}

  void tic() {
    _tstart = std::chrono::system_clock::now();
  }

  float toc() {
    auto tstop = std::chrono::system_clock::now();
    float elapse = std::chrono::duration<float, std::milli>(tstop - _tstart).count();
    tic();
    return elapse;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> _tstart;
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

template <typename T>
TrtUniquePtr<T> makeUnique(T* t) {
  if (!t) {
    throw std::runtime_error{"Failed to create TensorRT object"};
  }
  return TrtUniquePtr<T> {t};
}

template <typename T>
using TrtSharedPtr = std::shared_ptr<T>;

template <typename T>
std::shared_ptr<T> makeShared(T* t) {
  if (!t) {
    throw std::runtime_error{"Failed to create TensorRT object"};
  }
  return std::shared_ptr<T>(t, samplesCommon::InferDeleter());
}

template <typename T>
void show(const T* outputs, nvinfer1::Dims outputs_dims, std::string name) {
  std::cout<<name<<std::endl;
  std::cout<<"dims:";
  int n = 1;
  for(int i=0; i<outputs_dims.nbDims; i++) {
    n = n * outputs_dims.d[i];
    std::cout<<outputs_dims.d[i]<<" ";
  }
  std::cout<<std::endl;
  std::cout << "0-100: ";
  auto *ptr = outputs;
  int num = 100;
  for (int i = 0; i < num; i++) {
    std::cout << *ptr << " ";
    ptr++;
  }
  std::cout << "\n100-0: ";
  auto *ptr2 = outputs + n - num;
  for (int i = n - num; i < n; i++) {
    std::cout << *ptr2 << " ";
    ptr2++;
  }
  std::cout<<std::endl;
}

template <typename T>
void write_bin(const T* data, nvinfer1::Dims dims, std::string dir, std::string name) {
  if (!fileops::dir_exists(dir)) {
    fileops::create_dir(dir, 0755);
  }
  if (!fileops::dir_exists(dir+"/bin/")) {
    fileops::create_dir(dir+"/bin/", 0755);
  }
  if (!fileops::dir_exists(dir+"/shape/")) {
    fileops::create_dir(dir+"/shape/", 0755);
  }
  std::ofstream f_bin(dir+"/bin/"+name, std::ios::out | std::ios::binary);
  std::ofstream f_shape(dir+"/shape/"+name, std::ios::out | std::ios::binary);
  int size = 1;
  int shape_size = dims.nbDims;
  std::vector<int> shape(shape_size);
  for(int i=0; i<dims.nbDims; i++) {
    size *= dims.d[i];
    shape[i] = int(dims.d[i]);
  }
  f_shape.write((char*)shape.data(), shape_size*sizeof(int));
  f_bin.write((char*)data, size*sizeof(T));
}

inline int DimsCount(nvinfer1::Dims& d) {
  int count = 1;
  for (int i = 0; i < d.nbDims; i++) {
    count *= d.d[i];
  }
  return count;
}

#endif
