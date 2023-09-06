#include <unistd.h>

#include <numeric>

#include "TopsInference/TopsInferRuntime.h"
#include "dtu/util/switch_logging.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/tops_utils.h"
template <typename F>
cv::Scalar
_tensor_sum(const cv::Mat& m)
{
  // for channel size great 4, only calc sum for first 4 channels
  cv::Scalar r = {0, 0, 0, 0};
  std::vector<cv::Mat> channels;
  cv::split(m, channels);
  int n =
      std::accumulate(m.size.p, m.size.p + m.dims, 1, std::multiplies<int>());
  for (int i = 0; i < std::min(m.channels(), 4); i++) {
    auto* ptr = reinterpret_cast<F*>(channels[i].data);
    double s = 0;
    for (int j = 0; j < n; j++) {
      s += double(*(ptr + j));
    }
    r[i] = (double)s;
  }
  return r;
}

template <typename F>
void
print_mat(
    const cv::Mat& m, const std::string& name = "", bool with_content = false)
{
  // std::cout << std::setprecision(10);
  std::cout << ">>>>>>>>>>>>>>>>>>>>" << std::endl;
  std::cout << "mat.name:" << name << std::endl;
  ;
  std::cout << "mat::type=" << m.type() << std::endl;
  std::cout << "mat::sum=" << _tensor_sum<F>(m) << std::endl;

  std::cout << "mat::dims=[";
  int i = 0;
  for (; i < m.dims - 1; i++) {
    std::cout << m.size[i] << ",";
  }
  std::cout << m.size[i];
  if (m.dims == 2 && m.channels() > 1) {
    std::cout << "," << m.channels();
  }
  std::cout << "]" << std::endl;

  std::cout << "mat.isContinuous=" << m.isContinuous() << std::endl;
  std::cout << "mat.rows(h)=" << m.rows << std::endl;
  std::cout << "mat.cols(w)=" << m.cols << std::endl;
  std::cout << "mat.channels(c)=" << m.channels() << std::endl;

  auto func1 = [](F f) { std::cout << (int)f << ","; };
  auto func2 = [](F f) { std::cout << f << ","; };
  auto func = m.depth() <= 1 ? func1 : func2;

  if (m.empty())
    return;

  if (with_content) {
    // print normal mat
    if (m.dims == 2) {
      std::cout << "mat::data:" << std::endl;
      for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
          auto* ptr = m.ptr<F>(i, j);
          for (int k = 0; k < m.channels(); k++) {
            func(*(ptr + k));
          }
        }
        std::cout << std::endl;
      }
    } else {
      std::cout << "not supported for high dimention mat" << std::endl;
    }
  } else {
    int c = m.channels();
    int size = (c >= 1 ? c : 1);
    for (int i = 0; i < m.dims; i++) {
      size *= m.size[i];
    }
    int n = std::min(16, size);
    auto* ptr = reinterpret_cast<F*>(m.data);
    std::cout << "mat::data[0-15]=[";
    for (int i = 0; i < n; i++) {
      func(*(ptr + i));
    }
    std::cout << "]" << std::endl;
    ;
  }
}

void
CHW2HWC(cv::Mat m_in, cv::Mat& m_out)
{
  int b = m_in.size[0];
  int h = m_in.size[2];
  int w = m_in.size[3];
  cv::Mat m0 = cv::Mat(3, b * h * w, CV_32FC1, m_in.data);
  cv::Mat m1 = cv::Mat(b * h * w, 3, CV_32FC1, m_out.data);
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
image_pad(cv::Mat imgs, cv::Mat& graph_in, int imgH, int imgW)
{
  int b = imgs.size[0];
  int h = imgs.size[2];
  int w = imgs.size[3];
  if (h == imgH && w == imgW) {
    graph_in = imgs;
    return true;
  }

  cv::Mat imgs0 = cv::Mat(b * h, w, CV_32FC3, cv::Scalar(0.0f));
  CHW2HWC(imgs, imgs0);
  cv::Mat m0 = cv::Mat(b * imgH, imgW, CV_32FC3, cv::Scalar(0.0f));
  imgs0.copyTo(m0(cv::Rect(0, 0, w, b * h)));
  HWC2CHW(m0, graph_in);
  return true;
}

int
main()
{
  TopsInference::topsInference_init();
  int card_id = 0;
  uint32_t clusterIds[] = {0};

  void* tops_handler_ = TopsInference::set_device(card_id, clusterIds);
  // std::string exec_path = "../engines/pprec_mv1_svtr_new-mix-bs32.exec";
  std::string exec_path =
      "../engines/rec_res34_bilstm_new0_opset16-mix-bs32.exec";
  // std::string exec_path =
  // "../engines/rec_res34_bilstm_new0_opset16-default-bs32.exec"; std::string
  // exec_path = "../engines/rec_res34_bilstm_new0_opset16-fp16-bs32.exec";

  TopsInference::IEngine* engine = TopsInference::create_engine();
  engine->loadExecutable(exec_path.c_str());

  std::vector<ShapeInfo> inputs_shape_info = get_inputs_shape(engine);
  std::vector<ShapeInfo> outputs_shape_info = get_outputs_shape(engine);

  std::vector<int> input_dims = inputs_shape_info[0].dims;
  std::vector<int> output_dims0 = outputs_shape_info[0].dims;
  std::vector<int> output_dims1 = outputs_shape_info[1].dims;
  int inputH = input_dims[2];
  int inputW = input_dims[3];
  int input_men_size = inputs_shape_info[0].mem_size;
  int output_men_size0 = outputs_shape_info[0].mem_size;
  int output_men_size1 = outputs_shape_info[1].mem_size;
  std::cout << "input:";
  for (size_t i = 0; i < input_dims.size(); i++) {
    std::cout << input_dims[i] << " ";
  }
  std::cout << input_men_size << std::endl;
  std::cout << "output0:";
  for (size_t i = 0; i < output_dims0.size(); i++) {
    std::cout << output_dims0[i] << " ";
  }
  std::cout << output_men_size0 << std::endl;
  std::cout << "output1:";
  for (size_t i = 0; i < output_dims1.size(); i++) {
    std::cout << output_dims1[i] << " ";
  }
  std::cout << output_men_size1 << std::endl;

  void* storage_on_device = nullptr;
  TopsInference::mem_alloc(&storage_on_device, input_men_size);
  void* output_on_device0 = nullptr;
  TopsInference::mem_alloc(&output_on_device0, output_men_size0);
  void* output_on_device1 = nullptr;
  TopsInference::mem_alloc(&output_on_device1, output_men_size1);
  TopsInference::topsInferStream_t stream;
  TopsInference::create_stream(&stream);

  std::string data_dir = "/home/workspace/dataelem/data/";
  // std::string read_name = "pprec_prep_mv1_svtr.cvfs";
  // std::string write_name = "pprec_graph_mv1_svtr_enflame.cvfs";
  std::string read_name = "rec_prep_res34_bilstm.cvfs";
  std::string write_name = "rec_graph_res34_bilstm_enflame.cvfs";

  cv::FileStorage fs_rd(data_dir + read_name, cv::FileStorage::READ);
  cv::FileStorage fs_wr(data_dir + write_name, cv::FileStorage::WRITE);

  cv::Mat graph_out0 = cv::Mat(output_dims0, CV_32SC1, cv::Scalar(0));
  cv::Mat graph_out1 = cv::Mat(output_dims1, CV_32FC1, cv::Scalar(0.0f));
  // cv::Mat graph_ind = cv::Mat(output_dims0, CV_32SC1, cv::Scalar(0));
  int num = 0;
  fs_rd["num"] >> num;
  fs_wr << "num" << num;
  std::cout << "num:" << num << std::endl;
  int elapse = 0;
  int repeats = 1;
  bool save_result = true;
  for (int k0 = 0; k0 < repeats; k0++) {
    for (int k = 0; k < num; k++) {
      cv::Mat batch_inputs;
      fs_rd["batch_inputs" + std::to_string(k)] >> batch_inputs;
      int image_num = 0;
      fs_rd["image_num" + std::to_string(k)] >> image_num;
      for (int i = 0; i < image_num; i++) {
        std::string name, gt_text;
        ;
        fs_rd["image_name" + std::to_string(k) + "_" + std::to_string(i)] >>
            name;
        fs_rd["gt_text" + std::to_string(k) + "_" + std::to_string(i)] >>
            gt_text;
        if (save_result) {
          fs_wr << "image_name" + std::to_string(k) + "_" + std::to_string(i)
                << name;
          fs_wr << "gt_text" + std::to_string(k) + "_" + std::to_string(i)
                << gt_text;
        }
      }

      if (save_result) {
        fs_wr << "image_num" + std::to_string(k) << image_num;
        fs_wr << "input_width" + std::to_string(k) << batch_inputs.size[3];
      }

      cv::Mat graph_in = cv::Mat(input_dims, CV_32FC1, cv::Scalar(0.0f));
      print_mat<float>(batch_inputs, "batch_inputs");

      image_pad(batch_inputs, graph_in, inputH, inputW);
      print_mat<float>(graph_in, "graph_in");
      auto start = std::chrono::system_clock::now();
      TopsInference::mem_copy_async(
          (void*)(graph_in.data), storage_on_device, input_men_size,
          TopsInference::MemcpyKind::TIF_MEMCPY_HOST_TO_DEVICE, stream);

      void* inputs[] = {storage_on_device};
      void* outputs[] = {output_on_device0, output_on_device1};

      // call async run
      bool rtn = engine->run(
          (void**)inputs, (void**)outputs,
          TopsInference::BufferType::TIF_ENGINE_RSC_IN_DEVICE_OUT_DEVICE,
          stream);

      if (!rtn) {
        printf("engine run failed!");
      }

      TopsInference::mem_copy_async(
          output_on_device0, (void*)(graph_out0.data), output_men_size0,
          TopsInference::MemcpyKind::TIF_MEMCPY_DEVICE_TO_HOST, stream);
      TopsInference::mem_copy_async(
          output_on_device1, (void*)(graph_out1.data), output_men_size1,
          TopsInference::MemcpyKind::TIF_MEMCPY_DEVICE_TO_HOST, stream);

      TopsInference::synchronize_stream(stream);

      auto end = std::chrono::system_clock::now();
      elapse += (end - start).count() / 1000000;

      print_mat<int>(graph_out0, "graph_out0");
      print_mat<float>(graph_out1, "graph_out1");
      /*int64_t *pdata0 = (int64_t*)graph_out0.data;
      int *pdata1 = (int*)graph_ind.data;
      for(int i=0; i<output_dims0[0]*output_dims0[1]; i++){
        *(pdata1+i) = *(pdata0+i);
      }
      print_mat<int>(graph_ind, "graph_ind");
      */
      if (save_result) {
        fs_wr << "inds" + std::to_string(k) << graph_out0;
        fs_wr << "probs" + std::to_string(k) << graph_out1;
      }
    }
  }

  std::cout << "Latency: " << elapse / (num * repeats) << "(ms)" << std::endl;

  fs_rd.release();
  fs_wr.release();
  // make all streams synchronized
  TopsInference::synchronize_stream(stream);

  // destroy created stream and release resource
  TopsInference::destroy_stream(stream);

  // free memory on device
  TopsInference::mem_free(storage_on_device);
  TopsInference::mem_free(output_on_device0);
  TopsInference::mem_free(output_on_device1);

  TopsInference::release_engine(engine);
  // TopsInference::release_optimizer(optimizer_);
  // TopsInference::release_parser(parser_);
  TopsInference::release_device(tops_handler_);
  TopsInference::topsInference_finish();

  return 0;
}
