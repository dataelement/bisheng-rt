#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

#include <cnrt.h>
#include "mm_builder.h"  // NOLINT
#include "mm_network.h"  // NOLINT
#include "mm_runtime.h"  // NOLINT

int main(int argc, char** argv){
  //std::string model_path = "/root/workspace/model_repo/model_repository_cambricon/det_db_graph/1/model.mm";
  std::string model_path = "/root/workspace/projects/mm_cdemo/dbnet_fp16.mm";
  magicmind::IModel *model = magicmind::CreateIModel();
  model->DeserializeFromFile(model_path.c_str());

  std::string data_dir = "/root/workspace/data/";
  std::string read_name = "ppdet_prep.cvfs";

  cnrtSetDevice(0);
  cnrtQueue_t queue;
  cnrtQueueCreate(&queue);

  magicmind::IEngine *engine = model->CreateIEngine();
  magicmind::IContext *context = engine->CreateIContext();

  std::vector<magicmind::IRTTensor *> inputs, outputs;
  context->CreateInputTensors(&inputs);
  context->CreateOutputTensors(&outputs);

  cv::FileStorage fs(data_dir + read_name, cv::FileStorage::READ);

  int num = 0;
  fs["num"] >> num;
  num = 1;
  std::cout << "num:" << num << std::endl;
  for (int k = 0; k < num; k++) {
    cv::Mat prepout;
    fs["prepout" + std::to_string(k)] >> prepout;
    int batch_size = prepout.size[0];
    int channels = prepout.size[1];
    int height = prepout.size[2];
    int width = prepout.size[3];
    std::cout<<"batch_size:"<<batch_size<<" channels:"<<channels<<" height:"<<height<<" width:"<<width<<std::endl;
    inputs[0] -> SetDimensions(magicmind::Dims({batch_size, channels, height, width}));

    for(auto tensor: inputs){
      void *mlu_addr_ptr;
      CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, tensor->GetSize()));
      tensor->SetData(mlu_addr_ptr);
    }

    magicmind::Dims input_shape0 = inputs[0] -> GetDimensions();
    std::cout<< "input_shape0:"<< input_shape0 << std::endl;

    context->InferOutputShape(inputs, outputs);

    magicmind::Dims output_shape0 = outputs[0] -> GetDimensions();
    std::cout<< "output_shape0:"<< output_shape0 << std::endl;

    for(auto tensor: outputs){
      void *mlu_addr_ptr;
      CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, tensor->GetSize()));
      tensor->SetData(mlu_addr_ptr);
    }

    float* indata = (float*)prepout.data;
    float sin = 0.0f;
    std::cout<<"indata:";
    for(size_t i=0; i<batch_size*channels*height*width; i++){
      if(i < 16){
        std::cout<<*(indata+i)<<",";
      }

      sin += *(indata+i);
    }

    std::cout<<"sum:"<<sin<<std::endl;

    CNRT_CHECK(cnrtMemcpy(inputs[0]->GetMutableData(), prepout.data, inputs[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));

    context->Enqueue(inputs, outputs, queue);
    CNRT_CHECK(cnrtQueueSync(queue));

    auto out_size = outputs[0]->GetSize();
    void *output_data = outputs[0]->GetMutableData();
    std::vector<float> output_cpu(out_size/sizeof(float));

    CNRT_CHECK(cnrtMemcpy(output_cpu.data(), output_data, out_size, CNRT_MEM_TRANS_DIR_DEV2HOST));

    magicmind::Dims output_shape;
    output_shape = outputs[0] -> GetDimensions();
    std::cout<< output_shape << std::endl;
    std::cout<< output_shape[0] << std::endl;

    float sout = 0.0f;
    std::cout<<"outdata:";
    for(size_t i=0; i<output_cpu.size(); i++){
      if(i < 16){
        std::cout<<output_cpu[i]<<",";
      }

      sout += output_cpu[i];
    }

    std::cout<<"sum:"<<sout<<std::endl;
  }

  for (auto tensor : inputs) {
    cnrtFree(tensor->GetMutableData());
    tensor->Destroy();
  }
  for (auto tensor : outputs) {
    cnrtFree(tensor->GetMutableData());
    tensor->Destroy();
  }
  context->Destroy();
  engine->Destroy();
  model->Destroy();

  fs.release();
  
  return 0;
}