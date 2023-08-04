#include <gflags/gflags.h>
#include <fstream>
#include <iterator>
#include <streambuf>
#include <string>

#include <google/protobuf/text_format.h>

#include "model_config.pb.h"
#include "ops_def.pb.h"

#include "triton/common/fileops.h"

DEFINE_string(input_file, "", "input prototxt file");
DEFINE_string(output_file, "", "output proto file");
DEFINE_string(type, "", "protobuf message type");


struct MemBuffer : std::streambuf {
  MemBuffer(char* begin, char* end) { this->setg(begin, begin, end); }
};

bool
ReadTextFile(const std::string& path, std::string* contents)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    return false;
  }

  in.seekg(0, std::ios::end);
  contents->resize(in.tellg());
  in.seekg(0, std::ios::beg);
  in.read(&(*contents)[0], contents->size());
  in.close();
  return true;
}

int
main(int argc, char** argv)
{
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_input_file.length() == 0 || FLAGS_type.length() == 0) {
    printf(
        "run %s --input_file /path/to/in.pbtxt"
        " --type [op,pipeline]\n",
        argv[0]);
    return -1;
  }

  if (!dataelem::common::fileops::file_exists(FLAGS_input_file)) {
    printf("input file not existed: %s\n", FLAGS_input_file.c_str());
    return -1;
  }

  std::string output_file = FLAGS_input_file;
  output_file.replace(output_file.end() - 6, output_file.end(), ".proto");

  if (FLAGS_type.compare("op") == 0) {
    std::string content;
    bool ret = ReadTextFile(FLAGS_input_file, &content);
    inference::OpsDef opsdef;
    if (!google::protobuf::TextFormat::ParseFromString(content, &opsdef)) {
      return -1;
    }

    // std::string ser;
    // opsdef.SerializeToString(&ser);
    // std::vector<char> bytes(ser.begin(), ser.end());
    // inference::OpsDef opsdef2;
    // MemBuffer sbuf(bytes.data(), bytes.data() + bytes.size());
    // std::istream in(&sbuf);
    // opsdef2.ParseFromIstream(&in);

    std::fstream output(
        output_file, std::ios::out | std::ios::trunc | std::ios::binary);
    opsdef.SerializeToOstream(&output);
    output.close();
  } else if (FLAGS_type.compare("pipeline") == 0) {
    std::string content;
    ReadTextFile(FLAGS_input_file, &content);
    inference::PipelinesDef pipelines_def;
    if (!google::protobuf::TextFormat::ParseFromString(
            content, &pipelines_def)) {
      return -1;
    }

    std::fstream output(
        output_file, std::ios::out | std::ios::trunc | std::ios::binary);
    pipelines_def.SerializeToOstream(&output);
    output.close();
  }

  printf("succ\n");

  return 0;
}
