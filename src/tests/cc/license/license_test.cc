#include <gflags/gflags.h>
#include <time.h>

#include <fstream>
#include <sstream>
#undef LICENSE_STRATEGY
#define LICENSE_STRATEGY LICENSE_WEAK
#include "fileops.h"
#include "license_utils.h"

int
main(int argc, char** argv)
{
  const std::string DB_FILE = "./license.db";
  if (!dataelem::common::fileops::file_exists(DB_FILE)) {
    std::cout << DB_FILE << " NOT EXISTS\n";
    return 0;
  }

  std::stringstream buffer;
  std::ifstream fin(DB_FILE, std::ios::binary);
  buffer.str("");
  buffer << fin.rdbuf();
  std::string content(buffer.str());
  fin.close();
  cipher::KeyBytes hashKey;
  const std::string db_key = dataelem::common::LicenseInfo::get_db_key();
  picosha2::hash256_bytes(db_key, hashKey);
  std::vector<char> vec(content.begin(), content.end());
  int ret = cipher::DecryptAES(hashKey, vec.size(), vec);
  if (ret != 0) {
    std::cout << "DECRYPT DB FILE FAILED!" << std::endl;
    return 0;
  }

  std::string rc(vec.begin(), vec.end());
  std::cout << "db:" << rc << std::endl;
  if (rc.length() < 19) {
    std::cout << "PARSE DB FAILED, PLEASE ADD KEY FIRST!" << std::endl;
    return 0;
  }

  std::string license_head1 =
      dataelem::common::LicenseInfo::get_license_head_v1();
  std::string license_head0 = rc.substr(0, 17);
  if (license_head1 != license_head0) {
    std::cout << "PARSE DB FAILED, MAKE SURE DB FILE RIGHT!" << std::endl;
    return 0;
  }

  std::string key_base64 = rc.substr(17, 8);
  std::string pc_hash = dataelem::common::LicenseInfo::get_pc_code();
  int year = 0, month = 0, day = 0, feat_id = -1, custom_id = -1, gpu_num = -1;
  if (!dataelem::common::LicenseInfo::decode_key(
          pc_hash, key_base64, year, month, day, feat_id, custom_id, gpu_num)) {
    std::cout << "DECODE KEY FAILED!" << std::endl;
    return 0;
  }

  std::string finish_date0 =
      dataelem::common::LicenseInfo::get_date_from_dif(year, month, day);
  std::string cur_date = dataelem::common::LicenseInfo::get_current_date();
  std::string start_date = rc.substr(25, 8);
  std::string finish_date = rc.substr(33, 8);
  if (finish_date != finish_date0) {
    std::cout << "INVALID DB FILE" << std::endl;
  }
  if (cur_date < start_date) {
    std::cout << "INVALID SYSTIME, PLEASE SET TIME CORRECT!" << std::endl;
    return 0;
  }

  if (cur_date > finish_date) {
    std::cout << "授权文件到期，请联系DATAELEM" << std::endl;
    return 0;
  }

  std::cout << "有效期至:" << finish_date << " GPU_NUM:" << gpu_num
            << " FEAT_ID:" << feat_id << " CUSTOM_ID:" << custom_id
            << std::endl;
  return 0;
}
