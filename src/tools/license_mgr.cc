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
DEFINE_bool(create_key, false, "option to create key");
DEFINE_string(pc_code, "", "pc code");
DEFINE_string(duration, "", "duration");
DEFINE_int32(custom_id, 0, "custom_id");
DEFINE_int32(feat_id, 0, "feat_id");
DEFINE_int32(gpu_num, 0, "gpu_num");
DEFINE_bool(create_license_db, false, "create license db");
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
    // std::cout<<"有效期至:"<<finish_date<<" GPU_NUM:"<<gpu_num<<"
    // FEAT_ID:"<<feat_id<<std::endl;

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

  // create license pub/key
  if (FLAGS_create_key) {
    std::cout << "START TO CREATE KEY...\n";
    if (password_check(true) != 0) {
      return 0;
    }
    if (FLAGS_pc_code.length() != 8) {
      std::cout << "PC CODE ERROR, LEN MUST BE 8:" << FLAGS_pc_code.length()
                << "\n";
      return 0;
    }

    const std::vector<int> invalid_inds =
        dataelem::common::LicenseInfo::get_invalid_inds_for_pc_code();
    std::bitset<48> valid_pc_code =
        dataelem::common::LicenseInfo::get_valid_pc_code(
            FLAGS_pc_code, invalid_inds);
    // std::cout<<"valid_pc_code:"<<valid_pc_code<<std::endl;

    int day = 0, month = 0, year = 0;
    if (FLAGS_duration.length() == 0) {
      std::cout << "duration is empty, defalt: 3d" << std::endl;
      day = 3;
    } else if (FLAGS_duration.length() == 1) {
      std::cout << "duration invalid:" << FLAGS_duration << ", must **y/**m/**d"
                << std::endl;
      return 0;
    } else {
      if (!dataelem::common::LicenseInfo::get_duration(
              FLAGS_duration, year, month, day)) {
        std::cout << "duration invalid, must **y/**m/**d" << std::endl;
        return 0;
      }
    }

    // std::cout<<"duration year:"<<year<<" month:"<<month<<"
    // day:"<<day<<std::endl;
    if (day == 0 && month == 0 && year == 0) {
      std::cout << "duration invalid, must **y/**m/**d" << std::endl;
      return 0;
    }

    auto cur_date = std::chrono::system_clock::now();
    std::chrono::duration<int, std::ratio<60 * 60 * 24>> days(
        int(std::round(year * 365.25 + month * 30.5 + day)));
    auto finish_date = cur_date + days;
    std::string str_finish_date = get_str_date(finish_date);
    std::cout << "finish_date:" << str_finish_date << std::endl;

    std::string str_start_time =
        dataelem::common::LicenseInfo::get_start_time();
    // std::cout<<"str_start_time:"<<str_start_time<<std::endl;

    int dif_year = 0;
    int dif_month = 0;
    int dif_day = 0;
    dataelem::common::LicenseInfo::get_dif_date(
        str_finish_date, dif_year, dif_month, dif_day);
    std::bitset<8> year_bit(dif_year);
    std::bitset<4> month_bit(dif_month);
    std::bitset<5> day_bit(dif_day);
    // std::cout<<"year_bit:"<<year_bit<<" month_bit:"<<month_bit<<"
    // day_bit:"<<day_bit<<std::endl;
    if (FLAGS_custom_id > 255 | FLAGS_custom_id < 0) {
      std::cout << "custom_id invalid, must greater than 0 and less than 256"
                << std::endl;
      return 0;
    }

    if (FLAGS_feat_id > 255 | FLAGS_feat_id < 0) {
      std::cout << "feat invalid, must greater than 0 and less than 256"
                << std::endl;
      return 0;
    }

    if (FLAGS_gpu_num > 255 | FLAGS_gpu_num < 0) {
      std::cout << "gpu_num invalid, must greater than 0 and less than 256"
                << std::endl;
      return 0;
    }

    std::bitset<48> valid_code;
    std::bitset<8> feat_id(FLAGS_feat_id);
    std::bitset<8> custom_id(FLAGS_custom_id);
    std::bitset<8> gpu_num(FLAGS_gpu_num);
    for (int i = 0; i < 8; i++) {
      valid_code[i] = valid_pc_code[i] ^ feat_id[i];
      valid_code[i + 8] = valid_pc_code[i + 8] ^ custom_id[i];
      valid_code[i + 16] = valid_pc_code[i + 16] ^ gpu_num[i];
      valid_code[i + 31] = valid_pc_code[i + 31] ^ year_bit[i];
    }
    for (int i = 0; i < 4; i++) {
      valid_code[i + 39] = valid_pc_code[i + 39] ^ month_bit[i];
    }
    for (int i = 0; i < 5; i++) {
      valid_code[i + 43] = valid_pc_code[i + 43] ^ day_bit[i];
    }

    // std::cout<<"valid_code:"<<valid_code<<std::endl;

    std::string code_base64 =
        dataelem::common::LicenseInfo::get_base64(valid_code);
    std::cout << "KEY:" << code_base64 << std::endl;
  }

  // create license db
  if (FLAGS_create_license_db) {
    std::cout << "START TO CREATE LICENSE DB...\n";
    // if (password_check() != 0) {
    //   return 0;
    // }
    cipher::KeyBytes hashKey;
    const std::string db_key = dataelem::common::LicenseInfo::get_db_key();
    picosha2::hash256_bytes(db_key, hashKey);
    std::string license_head =
        dataelem::common::LicenseInfo::get_license_head_v0();
    std::vector<char> vec(license_head.begin(), license_head.end());
    std::ofstream dst(DB_FILE, std::ios::binary | std::ios::out);

    std::string rc(vec.begin(), vec.end());
    // std::cout<<"db:"<<rc<<std::endl;
    int ret = cipher::EncryptAES(hashKey, vec);
    if (ret != 0) {
      std::cout << "ENCRYPT CONTENT FAILED\n";
      return -2;
    }
    if (!dst.is_open()) {
      std::cout << "FAILED TO CREATE " << DB_FILE << "\n";
      return -1;
    }
    for (const auto& c : vec) {
      dst << c;
    }
    dst.close();
    std::cout << "SUCC TO CREATE " << DB_FILE << "\n";
    return 0;
  }

  return 0;
}
