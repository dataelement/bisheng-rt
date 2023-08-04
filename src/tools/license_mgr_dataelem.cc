#include <gflags/gflags.h>
#include <time.h>

#include <chrono>
#include <fstream>
#include <sstream>
#undef LICENSE_STRATEGY
#define LICENSE_STRATEGY LICENSE_WEAK
#include "fileops.h"
#include "license_utils.h"
#include "str_utils.h"

DEFINE_bool(show_info, false, "show pc code");
DEFINE_string(add_key, "", "add new key in license db");
DEFINE_string(update_key, "", "update key in license db");
DEFINE_string(db_file, "/opt/sae/license/license.db", "db file path");

int
password_check(bool is_simple_check = false)
{
  const std::string DECRYPT_KEY = "CV@4PARADIGM";
  const std::string COMMON_KEY = "20200214";
  std::string key = is_simple_check ? COMMON_KEY : DECRYPT_KEY;
  dataelem::common::LicenseInfo::set_terminal_buf(1);
  setbuf(stdin, NULL);
  std::string password =
      dataelem::common::LicenseInfo::get_password("EnterPassword:");
  dataelem::common::LicenseInfo::set_terminal_buf(0);
  if (password.compare(key) != 0) {
    std::cout << "ERROR PASSWORD\n";
    return -1;
  }
  return 0;
}

std::string
get_str_date(std::chrono::time_point<std::chrono::system_clock> date)
{
  std::time_t date_t = std::chrono::system_clock::to_time_t(date);
  std::tm* local_time = std::localtime(&date_t);
  int year = local_time->tm_year + 1900;
  int month = local_time->tm_mon + 1;
  int day = local_time->tm_mday;
  return std::to_string(year * 10000 + month * 100 + day);
}

int
main(int argc, char** argv)
{
  std::cout << "WELCOME TO USE DATAELEM AUTH PROCEDURE...\n";
  std::cout << "LICENSE MANAGER VERSION 2023.03.03\n";
  google::ParseCommandLineFlags(&argc, &argv, true);
  const std::string DB_FILE = FLAGS_db_file;

  if (FLAGS_show_info) {
    std::string pc_hash = dataelem::common::LicenseInfo::get_pc_code();
    std::cout << "PC_CODE:" << pc_hash << std::endl;
    return 0;
  }

  // add key / update key
  if (FLAGS_add_key.length() > 0 || FLAGS_update_key.length() > 0) {
    std::string msg = FLAGS_add_key.length() > 0 ? "START TO ADD KEY..."
                                                 : "START TO UPDATE KEY...";
    std::cout << msg << std::endl;
    ;
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
      std::cout << "DECRYPT DB FILE FAILED\n";
      return -1;
    }
    std::string rc(vec.begin(), vec.end());
    std::string license_head =
        FLAGS_add_key.length() > 0
            ? dataelem::common::LicenseInfo::get_license_head_v0()
            : dataelem::common::LicenseInfo::get_license_head_v1();
    if (rc.substr(0, license_head.length()) != license_head) {
      std::cout << "INVALID DB FILE!" << std::endl;
      std::string msg =
          FLAGS_add_key.length() ? "ADD KEY FAILED!" : "UPDATE KEY FAILED!";
      std::cout << msg << std::endl;
      return 0;
    }

    std::string key_base64 =
        FLAGS_add_key.length() > 0 ? FLAGS_add_key : FLAGS_update_key;
    std::string pc_hash = dataelem::common::LicenseInfo::get_pc_code();
    int year = 0, month = 0, day = 0, feat_id = -1, custom_id = -1,
        gpu_num = -1;
    if (!dataelem::common::LicenseInfo::decode_key(
            pc_hash, key_base64, year, month, day, feat_id, custom_id,
            gpu_num)) {
      std::cout << "DECODE KEY FAILED!" << std::endl;
      return 0;
    }

    std::string finish_date =
        dataelem::common::LicenseInfo::get_date_from_dif(year, month, day);
    std::string cur_date = dataelem::common::LicenseInfo::get_current_date();
    // std::cout<<"finish date:"<<finish_date<<std::endl;

    if (cur_date > finish_date) {
      std::cout << "INVALID SYSTIME, PLEASE SET TIME CORRECT!" << std::endl;
      return 0;
    }
    if (FLAGS_add_key.length() > 0) {
      license_head = dataelem::common::LicenseInfo::get_license_head_v1();
      license_head += FLAGS_add_key;
      license_head += cur_date;
      license_head += finish_date;
    }

    if (FLAGS_update_key.length() > 0) {
      license_head += FLAGS_update_key;
      license_head += cur_date;
      license_head += finish_date;
    }

    license_head += pc_hash;

    std::vector<char> vec2(license_head.begin(), license_head.end());
    ret = cipher::EncryptAES(hashKey, vec2);
    if (ret != 0) {
      std::cout << "ENCRYPT CONTENT FAILED\n";
      return -2;
    }

    std::ofstream dst(DB_FILE, std::ios::binary | std::ios::out);
    if (!dst.is_open()) {
      std::cout << "FAILED TO CREATE " << DB_FILE << "\n";
      return -3;
    }
    for (const auto& c : vec2) {
      dst << c;
    }
    dst.close();
    std::cout << "SUCC TO CREATE " << DB_FILE << "\n";
    std::cout << "有效期至:" << finish_date << " GPU_NUM:" << gpu_num
              << " FEAT_ID:" << feat_id << std::endl;
    return 0;
  }

  return 0;
}
