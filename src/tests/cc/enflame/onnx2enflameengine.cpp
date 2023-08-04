#include <iostream>
#include <string>

#include "utils/tops_utils.h"

int
main(int argc, char** argv)
{
  std::string onnx_path = argv[1];
  std::string engine_dir = argv[2];
  std::string input_names = argv[3];
  std::string input_shapes = argv[4];
  int precision_type = atoi(argv[5]);
  std::cout << "onnx_path:" << onnx_path << std::endl;
  std::cout << "engine_dir:" << engine_dir << std::endl;
  std::cout << "input_names:" << input_names << std::endl;
  std::cout << "input_shapes:" << input_shapes << std::endl;
  std::cout << "precision_type:" << precision_type << std::endl;
  TopsInference::BuildFlag bf = TopsInference::BuildFlag::TIF_KTYPE_MIX_FP16;
  if (precision_type == 0) {
    bf = TopsInference::BuildFlag::TIF_KTYPE_DEFAULT;
  } else if (precision_type == 1) {
    bf = TopsInference::BuildFlag::TIF_KTYPE_FLOAT16;
  }
  std::string exec_path = engine_name_construct(
      onnx_path.c_str(), engine_dir.c_str(), atoi(input_shapes.c_str()),
      get_precision_str(precision_type));

  int card_id = 0;
  uint32_t cluster_ids[] = {0};
  uint32_t cluster_num = 1;  // count of cluster_ids
  TopsInference::topsInference_init();
  void* tops_handler =
      TopsInference::set_device(card_id, cluster_ids, cluster_num);
  // TopsInference::IParser *parser =
  // TopsInference::create_parser(TopsInference::TIF_ONNX);;
  // TopsInference::IOptimizer *optimizer = TopsInference::create_optimizer();;
  // parser->setInputShapes(input_shapes.c_str());
  // TopsInference::INetwork *network = parser->readModel(onnx_path.c_str());
  // TopsInference::IEngine *engine = optimizer->build(network);
  TopsInference::IEngine* engine = loadOrCreateEngine(
      exec_path.c_str(), onnx_path.c_str(),
      // TopsInference::BuildFlag::TIF_KTYPE_MIX_FP16,
      bf, input_names.c_str(), input_shapes.c_str());

  engine->saveExecutable(exec_path.c_str());
  // TopsInference::release_network(network);

  TopsInference::release_engine(engine);
  // TopsInference::release_optimizer(optimizer);
  // TopsInference::release_parser(parser);
  TopsInference::release_device(tops_handler);
  TopsInference::topsInference_finish();
  return 1;
}
