#include "run_utils.h"

#include <assert.h>

#include <iostream>

int
engine_run_async_d2d(
    TopsInference::IEngine* engine, TopsStreamStruct* stream_struct,
    std::vector<void*>& inputs, std::vector<void*>& outputs, bool use_api2)
{
  assert(inputs.size() == stream_struct->inputs_info.size());
  assert(outputs.size() == stream_struct->outputs_info.size());
  // copy data from host to above allocated buffer on device
  for (unsigned int i = 0; i < stream_struct->inputs_info.size(); i++) {
    TopsInference::mem_copy_async(
        inputs[i], stream_struct->device_inputs[i],
        stream_struct->inputs_info[i].mem_size,
        TopsInference::MemcpyKind::TIF_MEMCPY_HOST_TO_DEVICE,
        stream_struct->stream);
  }

  // call async run() or run_with_batch()
  bool r;
  if (!use_api2) {
    r = engine->run(
        stream_struct->device_inputs, stream_struct->device_outputs,
        TopsInference::BufferType::TIF_ENGINE_RSC_IN_DEVICE_OUT_DEVICE,
        stream_struct->stream);
  } else {
    r = engine->run_with_batch(
        1, stream_struct->device_inputs, stream_struct->device_outputs,
        TopsInference::BufferType::TIF_ENGINE_RSC_IN_DEVICE_OUT_DEVICE,
        stream_struct->stream);
  }

  if (!r) {
    std::cout << "[ERROR] engine run failed!" << std::endl;
    return r;
  }

  // copy output data from device to host
  for (unsigned int i = 0; i < stream_struct->outputs_info.size(); i++) {
    TopsInference::mem_copy_async(
        stream_struct->device_outputs[i], outputs[i],
        stream_struct->outputs_info[i].mem_size,
        TopsInference::MemcpyKind::TIF_MEMCPY_DEVICE_TO_HOST,
        stream_struct->stream);
  }

  return true;
}