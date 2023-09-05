#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "acl/acl.h"

void
CHW2HWC(cv::Mat m_in, cv::Mat& m_out)
{
  int h = m_in.size[2];
  int w = m_in.size[3];
  cv::Mat m0 = cv::Mat(3, h * w, CV_32FC1, m_in.data);
  cv::Mat m1 = cv::Mat(h * w, 3, CV_32FC1, m_out.data);
  cv::transpose(m0, m1);
}

void
HWC2CHW(cv::Mat m_in, cv::Mat& m_out)
{
  int h = m_in.size[0];
  int w = m_in.size[1];
  cv::Mat m0 = cv::Mat(h * w, 3, CV_32FC1, m_in.data);
  cv::Mat m1 = cv::Mat(3, h * w, CV_32FC1, m_out.data);
  cv::transpose(m0, m1);
}

bool
image_pad(cv::Mat img, cv::Mat& graph_in, int imgH, int imgW)
{
  int h = img.size[2];
  int w = img.size[3];
  if (h == imgH && w == imgW) {
    graph_in = img;
    return true;
  }

  cv::Mat img0 = cv::Mat(h, w, CV_32FC3, cv::Scalar(0.0f));
  CHW2HWC(img, img0);
  cv::Mat m0 = cv::Mat(imgH, imgW, CV_32FC3, cv::Scalar(0.0f));
  img0.copyTo(m0(cv::Rect(0, 0, w, h)));
  HWC2CHW(m0, graph_in);
  return true;
}

int
main(int argc, char** argv)
{
  std::string model_path = "/home/public/models/det_r34_vd_db.om";
  std::string acl_config_path = "acl.json";
  int32_t device_id = 0;
  aclrtContext context;
  aclrtStream stream;
  aclrtRunMode run_mode;
  aclError ret = aclInit(acl_config_path.c_str());
  if (ret != ACL_SUCCESS) {
    std::cout << "aclInit failed!" << std::endl;
    return 0;
  }
  ret = aclrtSetDevice(device_id);
  if (ret != ACL_SUCCESS) {
    std::cout << "aclrtSetDevice failed!" << std::endl;
    return 0;
  }
  ret = aclrtCreateContext(&context, device_id);
  if (ret != ACL_SUCCESS) {
    std::cout << "aclrtCreateContext failed!" << std::endl;
    return 0;
  }
  ret = aclrtCreateStream(&stream);
  if (ret != ACL_SUCCESS) {
    std::cout << "aclrtCreateStream failed!" << std::endl;
    return 0;
  }
  ret = aclrtGetRunMode(&run_mode);
  if (ret != ACL_SUCCESS) {
    std::cout << "aclrtGetRunMode failed!" << std::endl;
    return 0;
  }
  bool is_device = (run_mode == ACL_DEVICE);

  uint32_t model_id;
  size_t model_worksize = 0;
  size_t model_weightsize = 0;
  ret = aclmdlQuerySize(model_path.c_str(), &model_worksize, &model_weightsize);
  if (ret != ACL_SUCCESS) {
    std::cout << "aclmdlQuerySize failed!" << std::endl;
    return 0;
  }
  void* model_workptr;
  void* model_weightptr;
  ret = aclrtMalloc(&model_workptr, model_worksize, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    std::cout << "aclrtMalloc model_workptr failed!" << std::endl;
    return 0;
  }
  ret = aclrtMalloc(
      &model_weightptr, model_weightsize, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    std::cout << "aclrtMalloc model_weightptr failed!" << std::endl;
    return 0;
  }
  ret = aclmdlLoadFromFileWithMem(
      model_path.c_str(), &model_id, model_workptr, model_worksize,
      model_weightptr, model_weightsize);
  if (ret != ACL_SUCCESS) {
    std::cout << "aclmdlLoadFromFileWithMem failed!" << std::endl;
    return 0;
  }

  aclmdlDesc* model_desc = aclmdlCreateDesc();
  ret = aclmdlGetDesc(model_desc, model_id);
  if (ret != ACL_SUCCESS) {
    std::cout << "aclmdlGetDesc failed!" << std::endl;
    return 0;
  }

  uint32_t num_inputs = aclmdlGetNumInputs(model_desc);
  std::cout << "num_inputs:" << num_inputs << std::endl;
  aclmdlDataset* device_inputs = aclmdlCreateDataset();
  std::vector<void*> host_inputs(num_inputs);
  for (size_t i = 0; i < num_inputs; i++) {
    size_t data_len = aclmdlGetInputSizeByIndex(model_desc, i);
    void* data = nullptr;
    ret = aclrtMalloc(&data, data_len, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclDataBuffer* data_buf = aclCreateDataBuffer(data, data_len);
    ret = aclmdlAddDatasetBuffer(device_inputs, data_buf);
    aclrtMallocHost(&host_inputs[i], data_len);
  }

  uint32_t num_outputs = aclmdlGetNumOutputs(model_desc);
  std::cout << "num_outputs:" << num_outputs << std::endl;
  aclmdlDataset* device_outputs = aclmdlCreateDataset();
  std::vector<void*> host_outputs(num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    size_t data_len = aclmdlGetOutputSizeByIndex(model_desc, i);
    void* data = nullptr;
    ret = aclrtMalloc(&data, data_len, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclDataBuffer* data_buf = aclCreateDataBuffer(data, data_len);
    ret = aclmdlAddDatasetBuffer(device_outputs, data_buf);
    std::cout << "i:" << i << " host_outputs:" << data_len << std::endl;
    aclrtMallocHost(&host_outputs[i], data_len);
  }

  aclmdlIODims current_dims;
  current_dims.dimCount = 4;
  current_dims.dims[0] = 1;
  current_dims.dims[1] = 3;
  current_dims.dims[2] = 960;
  current_dims.dims[3] = 960;
  ret = aclmdlSetInputDynamicDims(model_id, device_inputs, 0, &current_dims);

  std::string data_dir = "/home/public/data/";
  std::string read_name = "ppdet_prep.cvfs";
  cv::FileStorage fs(data_dir + read_name, cv::FileStorage::READ);

  int num = 0;
  fs["num"] >> num;
  num = 1;
  std::cout << "num:" << num << std::endl;
  for (int k = 0; k < num; k++) {
    cv::Mat prepout;
    fs["prepout" + std::to_string(k)] >> prepout;
    std::vector<int> input_dims = {1, 3, 960, 960};
    cv::Mat graph_in = cv::Mat(input_dims, CV_32FC1, cv::Scalar(0.0f));
    image_pad(prepout, graph_in, 960, 960);

    int batch_size = graph_in.size[0];
    int channels = graph_in.size[1];
    int height = graph_in.size[2];
    int width = graph_in.size[3];
    std::cout << "batch_size:" << batch_size << " channels:" << channels
              << " height:" << height << " width:" << width << std::endl;

    float* indata = (float*)graph_in.data;
    float sin = 0.0f;
    std::cout << "indata:";
    for (size_t i = 0; i < batch_size * channels * height * width; i++) {
      if (i < 16) {
        std::cout << *(indata + i) << ",";
      }

      sin += *(indata + i);
    }

    std::cout << "sum:" << sin << std::endl;

    int inputsize = batch_size * channels * height * width * 4;
    memcpy(host_inputs[0], graph_in.data, inputsize);

    for (size_t i = 0; i < num_inputs; i++) {
      aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(device_inputs, i);
      void* data = aclGetDataBufferAddr(data_buffer);
      uint32_t data_len = aclGetDataBufferSizeV2(data_buffer);
      ret = aclrtMemcpy(
          data, data_len, host_inputs[i], data_len, ACL_MEMCPY_HOST_TO_DEVICE);
      if (ret != ACL_SUCCESS) {
        std::cout << "aclrtMemcpy host2device failed!" << std::endl;
      }
    }

    // ret = aclmdlExecute(model_id, device_inputs, device_outputs);
    ret = aclmdlExecuteAsync(model_id, device_inputs, device_outputs, stream);
    aclrtSynchronizeStream(stream);

    uint32_t out_data_len = 0;
    for (size_t i = 0; i < num_outputs; i++) {
      aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(device_outputs, i);
      void* data = aclGetDataBufferAddr(data_buffer);
      uint32_t data_len = aclGetDataBufferSizeV2(data_buffer);
      std::cout << "i:" << i << " before copy host_outputs:" << data_len
                << std::endl;
      ret = aclrtMemcpy(
          host_outputs[i], data_len, data, data_len, ACL_MEMCPY_DEVICE_TO_HOST);
      if (ret != ACL_SUCCESS) {
        std::cout << "aclrtMemcpy device2host failed!" << std::endl;
      }
      out_data_len = data_len;
    }

    std::cout << "out_data_len:" << out_data_len << std::endl;
    uint32_t out_data_num = out_data_len / 4;
    float* outdata = reinterpret_cast<float*>(host_outputs[0]);
    float sout = 0.0f;
    std::cout << "outdata:";
    for (size_t i = 0; i < out_data_num; i++) {
      if (i < 16) {
        std::cout << *(outdata + i) << ",";
      }

      sout += *(outdata + i);
    }

    std::cout << "sum:" << sout << std::endl;
  }

  for (size_t i = 0; i < num_inputs; ++i) {
    (void)aclrtFreeHost(host_inputs[i]);
    aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(device_inputs, i);
    void* data = aclGetDataBufferAddr(data_buffer);
    (void)aclrtFree(data);
    (void)aclDestroyDataBuffer(data_buffer);
  }
  (void)aclmdlDestroyDataset(device_inputs);
  device_inputs = nullptr;

  for (size_t i = 0; i < num_outputs; ++i) {
    (void)aclrtFreeHost(host_outputs[i]);
    aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(device_outputs, i);
    void* data = aclGetDataBufferAddr(data_buffer);
    (void)aclrtFree(data);
    (void)aclDestroyDataBuffer(data_buffer);
  }
  (void)aclmdlDestroyDataset(device_outputs);
  device_outputs = nullptr;

  if (model_desc != nullptr) {
    (void)aclmdlDestroyDesc(model_desc);
    model_desc = nullptr;
  }

  if (model_workptr != nullptr) {
    (void)aclrtFree(model_workptr);
    model_workptr = nullptr;
    model_worksize = 0;
  }

  if (model_weightptr != nullptr) {
    (void)aclrtFree(model_weightptr);
    model_weightptr = nullptr;
    model_weightsize = 0;
  }
  model_id = 0;

  if (stream != nullptr) {
    aclrtDestroyStream(stream);
    stream = nullptr;
  }

  if (context != nullptr) {
    aclrtDestroyContext(context);
    context = nullptr;
  }

  aclrtResetDevice(device_id);
  aclFinalize();

  fs.release();

  return 0;
}