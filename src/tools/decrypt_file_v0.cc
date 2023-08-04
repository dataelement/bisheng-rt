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

  std::vector<char> bytes;
  cipher::read_plain_binary(FLAGS_input_file, bytes);

  std::ofstream fout;
  fout.open(FLAGS_output_file, std::ios::binary | std::ios::out);
  fout.write(bytes.data(), bytes.size());
  fout.close();

  return 0;
}
