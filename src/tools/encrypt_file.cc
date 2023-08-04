#include <gflags/gflags.h>
#include <fstream>
#include <iterator>
#include <string>

#include "triton/common/cipher/aes.hpp"
#include "triton/common/fileops.h"

DEFINE_string(input_file, "", "input file");
DEFINE_string(output_file, "", "output file");

int
main(int argc, char** argv)
{
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_input_file.length() == 0) {
    printf(
        "run %s --input_file /path/to/file.in"
        " --output_file /path/to/file.out\n",
        argv[0]);
    return -1;
  }

  if (!dataelem::common::fileops::file_exists(FLAGS_input_file)) {
    printf("input file not existed: %s\n", FLAGS_input_file.c_str());
    return -1;
  }

  std::ifstream input(FLAGS_input_file, std::ios::binary);
  std::vector<char> bytes(
      (std::istreambuf_iterator<char>(input)),
      (std::istreambuf_iterator<char>()));
  input.close();

  std::string output_file = FLAGS_output_file;
  if (output_file.length() == 0) {
    output_file = FLAGS_input_file + ".pri";
  }

  // cipher::WriteAESBinary(output_file, bytes);

  cipher::WriteSimpleEncBinary(output_file, bytes);

  // std::vector<char> new_bytes;
  // cipher::ReadSimpleEncBinary(output_file, new_bytes);
  // std::ofstream fout;
  // auto o1 = FLAGS_input_file + ".ori";
  // fout.open(o1, std::ios::binary | std::ios::out);
  // fout.write(new_bytes.data(), new_bytes.size());
  // fout.close();

  return 0;
}
