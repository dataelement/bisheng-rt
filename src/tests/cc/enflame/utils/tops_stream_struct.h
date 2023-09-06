#pragma once
#include <iostream>
#include <vector>

#include "../utils/tops_utils.h"
#include "TopsInference/TopsInferRuntime.h"

class TopsStreamStruct {
 public:
  TopsStreamStruct(
      std::vector<ShapeInfo>& _inputs_info,
      std::vector<ShapeInfo>& _outputs_info)
  {
    inputs_info = _inputs_info;
    outputs_info = _outputs_info;
    int input_num = inputs_info.size();
    int output_num = outputs_info.size();
    device_inputs = new void*[input_num];
    device_outputs = new void*[output_num];
    // allocate buffer on device
    for (int i = 0; i < input_num; i++) {
      std::cout << "[INFO] mem_alloc input, size: " << inputs_info[i].mem_size
                << std::endl;
      TopsInference::mem_alloc(&device_inputs[i], inputs_info[i].mem_size);
    }
    for (int i = 0; i < output_num; i++) {
      std::cout << "[INFO] mem_alloc output, size: " << outputs_info[i].mem_size
                << std::endl;
      TopsInference::mem_alloc(&device_outputs[i], outputs_info[i].mem_size);
    }
    TopsInference::create_stream(&stream);
  }

  ~TopsStreamStruct()
  {
    int input_num = inputs_info.size();
    int output_num = outputs_info.size();
    // free memory on device
    for (int i = 0; i < input_num; i++) {
      TopsInference::mem_free(device_inputs[i]);
    }
    for (int i = 0; i < output_num; i++) {
      TopsInference::mem_free(device_outputs[i]);
    }

    TopsInference::destroy_stream(stream);
    delete[] device_inputs;
    delete[] device_outputs;
  }

 public:
  void** device_inputs;
  void** device_outputs;
  std::vector<ShapeInfo> inputs_info;
  std::vector<ShapeInfo> outputs_info;
  // create stream to bond with async actions
  TopsInference::topsInferStream_t stream;
};