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
  // print_mat<float>(img, "img");
  // print_mat<float>(img0, "img0");
  cv::Mat m0 = cv::Mat(imgH, imgW, CV_32FC3, cv::Scalar(0.0f));
  img0.copyTo(m0(cv::Rect(0, 0, w, h)));
  // print_mat<float>(m0, "m0");
  HWC2CHW(m0, graph_in);
  // print_mat<float>(graph_in, "graph_in");
  return true;
}

int
main()
{
  TopsInference::topsInference_init();
  int card_id = 0;
  uint32_t clusterIds[] = {0};

  void* tops_handler_;
  TopsInference::IParser* parser_;
  TopsInference::IOptimizer* optimizer_;

  tops_handler_ = TopsInference::set_device(card_id, clusterIds);
  parser_ = TopsInference::create_parser(TopsInference::TIF_ONNX);
  optimizer_ = TopsInference::create_optimizer();
  // std::string exec_path = "../engines/dbnet-mix-bs1.exec";
  // std::string exec_path = "../engines/det_r50_db_opset16-default-bs1.exec";
  // std::string exec_path = "../engines/det_r50_db_opset16-fp16-bs1.exec";
  // std::string exec_path = "../engines/det_r50_db_opset16-mix-bs1.exec";
  // std::string exec_path = "../engines/det_r50_db_350_opset16-mix-bs1.exec";
  // std::string exec_path =
  // "../engines/ch_ppocr_server_v2.0_det_opset16-mix-bs1.exec";
  std::string exec_path = "../engines/det_r34_db_0_opset16-mix-bs1.exec";
  // std::string fpath = "../models/dbnet.onnx";
  std::string fpath = "";

  TopsInference::IEngine* engine = nullptr;
  if (access(exec_path.c_str(), 0) == -1) {
    parser_->setInputShapes("1,3,960,960");
    TopsInference::INetwork* network = parser_->readModel(fpath.c_str());
    engine = optimizer_->build(network);
    engine->saveExecutable(exec_path.c_str());
    TopsInference::release_network(network);
  } else {
    engine = TopsInference::create_engine();
    engine->loadExecutable(exec_path.c_str());
  }


  std::vector<ShapeInfo> inputs_shape_info = get_inputs_shape(engine);
  std::vector<ShapeInfo> outputs_shape_info = get_outputs_shape(engine);

  std::vector<int> input_dims = inputs_shape_info[0].dims;
  std::vector<int> output_dims = outputs_shape_info[0].dims;
  int inputH = input_dims[2];
  int inputW = input_dims[3];
  int input_men_size = inputs_shape_info[0].mem_size;
  int output_men_size = outputs_shape_info[0].mem_size;
  std::cout << "input:";
  for (size_t i = 0; i < input_dims.size(); i++) {
    std::cout << input_dims[i] << " ";
  }
  std::cout << inputs_shape_info[0].mem_size << std::endl;
  std::cout << "output:";
  for (size_t i = 0; i < output_dims.size(); i++) {
    std::cout << output_dims[i] << " ";
  }
  std::cout << outputs_shape_info[0].mem_size << std::endl;

  void* storage_on_device = nullptr;
  TopsInference::mem_alloc(&storage_on_device, input_men_size);
  void* output_on_device = nullptr;
  TopsInference::mem_alloc(&output_on_device, output_men_size);
  TopsInference::topsInferStream_t stream;
  TopsInference::create_stream(&stream);

  // std::string data_dir = "/home/workspace/dataelem/data/";
  std::string data_dir = "../data/";
  std::string read_name = "ppdet_prep.cvfs";
  std::string write_name = "det_graph_dbnet_r50_enflame.cvfs";

  cv::FileStorage fs_rd(data_dir + read_name, cv::FileStorage::READ);
  cv::FileStorage fs_wr(data_dir + write_name, cv::FileStorage::WRITE);

  cv::Mat graph_out = cv::Mat(output_dims, CV_32FC1, cv::Scalar(0.0f));
  int num = 0;
  fs_rd["num"] >> num;
  // num = 1;
  fs_wr << "num" << num;
  std::cout << "num:" << num << std::endl;
  int elapse = 0;
  int repeats = 1;
  bool save_result = true;
  for (int k0 = 0; k0 < repeats; k0++) {
    for (int k = 0; k < num; k++) {
      cv::Mat prepout, shape_list, orig_shape, gt;
      fs_rd["prepout" + std::to_string(k)] >> prepout;
      fs_rd["shape_list" + std::to_string(k)] >> shape_list;
      fs_rd["orig_shape" + std::to_string(k)] >> orig_shape;
      fs_rd["gt" + std::to_string(k)] >> gt;
      std::string name;
      fs_rd["image_name" + std::to_string(k)] >> name;
      if (save_result) {
        fs_wr << "image_name" + std::to_string(k) << name;
        fs_wr << "shape_list" + std::to_string(k) << shape_list;
        fs_wr << "orig_shape" + std::to_string(k) << orig_shape;
        fs_wr << "gt" + std::to_string(k) << gt;
      }

      cv::Mat graph_in = cv::Mat(input_dims, CV_32FC1, cv::Scalar(0.0f));
      print_mat<float>(prepout, "prepout");
      image_pad(prepout, graph_in, inputH, inputW);
      print_mat<float>(graph_in, "graph_in");
      auto start = std::chrono::system_clock::now();
      TopsInference::mem_copy_async(
          (void*)(graph_in.data), storage_on_device, input_men_size,
          TopsInference::MemcpyKind::TIF_MEMCPY_HOST_TO_DEVICE, stream);

      void* inputs[] = {storage_on_device};
      void* outputs[] = {output_on_device};

      // call async run
      bool rtn = engine->run(
          (void**)inputs, (void**)outputs,
          TopsInference::BufferType::TIF_ENGINE_RSC_IN_DEVICE_OUT_DEVICE,
          stream);

      if (!rtn) {
        printf("engine run failed!");
      }

      TopsInference::mem_copy_async(
          output_on_device, (void*)(graph_out.data), output_men_size,
          TopsInference::MemcpyKind::TIF_MEMCPY_DEVICE_TO_HOST, stream);

      TopsInference::synchronize_stream(stream);

      auto end = std::chrono::system_clock::now();
      elapse += (end - start).count() / 1000000;

      print_mat<float>(graph_out, "graph_out");
      if (save_result) {
        fs_wr << "featmap" + std::to_string(k) << graph_out;
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
  TopsInference::mem_free(output_on_device);

  TopsInference::release_engine(engine);
  TopsInference::release_optimizer(optimizer_);
  TopsInference::release_parser(parser_);
  TopsInference::release_device(tops_handler_);
  TopsInference::topsInference_finish();

  return 0;
}
