#pragma once

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

#include <ifaddrs.h>
#include <linux/if_link.h>
#include <memory.h>
#include <net/if.h>
#include <netdb.h>
#include <netpacket/packet.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>

#include <iostream>
#include <unordered_map>

#if defined __arm__ || defined __aarch64__
#include <asm/hwcap.h>
#include <sys/auxv.h>
#else
#include <cpuid.h>
#endif

#include <arpa/inet.h>
#include <rapidjson/error/en.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>
#include <termios.h>
#include <unistd.h>

#include <atomic>
#include <bitset>
#include <chrono>
#include <thread>
#include <vector>

#include "triton/common/fileops.h"
#include "triton/common/str_utils.h"
#include "triton/common/utils.h"

#include "hasp_api.h"
#include "license_config.h"
#include "rapidxml/rapidxml.hpp"
#include "triton/common/cipher/aes.hpp"

#include "absl/strings/escaping.h"
#include "client.h"

namespace dataelem { namespace common {


class LicenseInfo {
  // qps
 public:
  static void enable_qps_quota_limit()
  {
    if (qps_quota_renew_interval() != 0 && qps_quota_cnt_limit() != 0) {
      std::unique_lock<std::mutex> lock(qps_mutex());
      qps_cv().wait(lock, []() { return qps_quota_remain() > 0; });
      qps_quota_remain()--;
    }
  }

 public:
  static std::thread& monitor_thread()
  {
    static std::thread thrd;
    return thrd;
  }

  static bool& stop_monitor_thread()
  {
    static bool stop = false;
    return stop;
  }

  static int& monitor_interval_sec()
  {
    static int interval = 1;
    return interval;
  }

  // qps
 public:
  static std::mutex& qps_mutex()
  {
    static std::mutex mtx;
    return mtx;
  }

  static std::condition_variable& qps_cv()
  {
    static std::condition_variable cv;
    return cv;
  }

  static int& qps_quota_cnt_limit()
  {
    static int qps_quota_cnt_limit = 0;
    return qps_quota_cnt_limit;
  }

  static int& qps_quota_renew_interval()
  {
    static int qps_time = 0;
    return qps_time;
  }

  static int& qps_quota_remain()
  {
    static int renew_count;
    return renew_count;
  }

  // Add variable and method for control by req cnt
  static std::atomic<int>& req_cnt()
  {
    static std::atomic<int> req_cnt;
    return req_cnt;
  }

  static std::atomic<int>& req_cnt_per_day()
  {
    static std::atomic<int> req_cnt_per_day;
    return req_cnt_per_day;
  }

  static void increase_req_cnt() { req_cnt()++; }
  static void increase_req_cnt_per_day() { req_cnt_per_day()++; }

  static void reset_req_cnt() { req_cnt() = 0; }

  static void reset_req_cnt_per_day() { req_cnt_per_day() = 0; }

  static int& req_cnt_quota()
  {
    static int req_cnt_quota = 0;
    return req_cnt_quota;
  }

  static int& req_cnt_quota_per_day()
  {
    static int req_cnt_quota_per_day = 0;
    return req_cnt_quota_per_day;
  }

 public:
  static bool auth()
  {
    LOG_INFO << "HASP LICENSE MANAGER\n";
    LOG_INFO << "LICENSE_STRATEGY: "
              << "HASP";
    hasp_status_t status;
    LOG_INFO << "SHORT VENDOR_CODE: "
              << std::string(LICENSE_VENDOR_CODE).substr(0, 8);
    // login
    status = hasp_login(LICENSE_FEATURE_ID, LICENSE_VENDOR_CODE, &(hasp_hdl()));
    if (status != HASP_STATUS_OK) {
      LOG_INFO << "License login failed, ret code: " << status;
      return false;
    }

    const size_t VERSION_OFFSET = 0;
    const size_t VERSION_LENGTH = 2;
    char buffer[4096];
    // get version
    status = hasp_read(
        hasp_hdl(), HASP_FILEID_RO, VERSION_OFFSET, VERSION_LENGTH, buffer);
    buffer[VERSION_LENGTH] = '\0';
    if (status != HASP_STATUS_OK) {
      LOG_INFO << "Fetch version from License failed, ret code: " << status;
      return false;
    }

    // auth
    std::string ver(buffer);
    LOG_INFO << "Licnese layout version: " << ver;
    if (ver == "01") {
      return inner_auth_v01(VERSION_LENGTH + 1, 0);
    } else {
      LOG_INFO << "Invalid license version: " << ver;
      return false;
    }
  }

  static void deauth()
  {
    hasp_status_t status;
    status = hasp_logout(hasp_hdl());
    if (status != HASP_STATUS_OK) {
      LOG_INFO << "License logout failed, ret code: " << status;
    } else {
      LOG_INFO << "Logout HASP license";
    }
  }

  static bool auth_dataelem()
  {
    LOG_INFO << "DATAELEM LICENSE MANAGER";
    LOG_INFO << "LICENSE_STRATEGY: " << "ElemHASP";
    std::string cfg_file = "/root/.hasplm/licenselm.ini";

    if (!dataelem::common::fileops::file_exists(cfg_file)) {
      LOG_INFO << "License config file not exist!";
      return false;
    }

    std::ifstream fs(cfg_file);
    std::string url(
        (std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());

    get_dclient()->set_url(rtrim(url));

    // login
    if (!dataelem_login()) {
      LOG_INFO << "License login failed!";
      return false;
    }

    stop_monitor_thread() = false;
    monitor_thread() = std::thread(renew_lic_thread_runner_dataelem);
    return true;
  }

  static void deauth_dataelem() { dataelem_logout(); }

 private:
  static hasp_handle_t& hasp_hdl()
  {
    static hasp_handle_t hdl;
    return hdl;
  }

  static bool inner_auth_v01(size_t ro_offset, size_t rw_offset)
  {
    char buffer[4096];
    hasp_status_t status;
    // qps
    const size_t QPS_OFFSET = 11 + ro_offset;
    const size_t QPS_LENGTH = 3;
    const size_t QPS_TIME_OFFSET = 19 + ro_offset;
    const size_t QPS_TIME_LENGTH = 3;
    status =
        hasp_read(hasp_hdl(), HASP_FILEID_RO, QPS_OFFSET, QPS_LENGTH, buffer);
    if (status != HASP_STATUS_OK) {
      LOG_INFO << "Fetch version from License failed, ret code: " << status;
      return false;
    }
    int qps_limit = 0;
    if (strlen(buffer) > 0 && !safe_lexical_cast(buffer, qps_limit)) {
      LOG_INFO << "Invalid QPS quota in License, " << buffer;
      return false;
    }
    qps_quota_cnt_limit() = qps_limit;
    LOG_INFO << "QPS quota:" << qps_quota_cnt_limit();

    status = hasp_read(
        hasp_hdl(), HASP_FILEID_RO, QPS_TIME_OFFSET, QPS_TIME_LENGTH, buffer);
    if (status != HASP_STATUS_OK) {
      LOG_INFO << "Fetch qps from License failed, ret code: " << status;
      return false;
    }
    int qps_interval_time = 0;
    if (strlen(buffer) > 0 && !safe_lexical_cast(buffer, qps_interval_time)) {
      LOG_INFO << "Invalid QPS interval time in License, " << buffer;
      return false;
    }
    qps_quota_renew_interval() = qps_interval_time;
    LOG_INFO << "QPS interval time:" << qps_quota_renew_interval();

    // update req cnt quota
    int accu_req_cnt = 0;
    int ACCU_REQ_CNT_OFFSET = 0x06B8 + rw_offset;
    int ACCU_REQ_CNT_LENGTH = 4;
    int max_req_cnt = 0;
    int REQ_CNT_QUOTA_OFFSET = 0x0016 + ro_offset;
    int REQ_CNT_QUOTA_LENGTH = 10;

    status = hasp_read(
        hasp_hdl(), HASP_FILEID_RO, REQ_CNT_QUOTA_OFFSET, REQ_CNT_QUOTA_LENGTH,
        buffer);
    if (status != HASP_STATUS_OK) {
      LOG_INFO << "Fetch req cnt quota from License failed, ret code: "
                << status;
      return false;
    }

    if (strlen(buffer) > 0 && !safe_lexical_cast(buffer, max_req_cnt)) {
      LOG_INFO << "Invalid req cnt quota in License, " << buffer;
      return false;
    }
    // if max_req_cnt equal 0 then disable req cnt quota check
    req_cnt_quota() = max_req_cnt;
    LOG_INFO << "Request Count Quota:" << req_cnt_quota();

    status = hasp_read(
        hasp_hdl(), HASP_FILEID_RW, ACCU_REQ_CNT_OFFSET, ACCU_REQ_CNT_LENGTH,
        &accu_req_cnt);
    if (status != HASP_STATUS_OK) {
      LOG_INFO << "Fetch accu req cnt from License failed, ret code: "
                << status;
      return false;
    }

    if (max_req_cnt > 0 && accu_req_cnt >= max_req_cnt) {
      LOG_INFO << "Exceeded Request Count Quota, Please Get Auth Again.";
      return false;
    } else {
      LOG_INFO << "Succ to Check Request Count Quota";
    }

    // update req cnt quota per day
    int accu_req_cnt_per_day = 0;
    int ACCU_REQ_CNT_PER_DAY_OFFSET = 0x06C2 + rw_offset;
    int ACCU_REQ_CNT_PER_DAY_LENGTH = 4;
    int max_req_cnt_per_day = 0;
    int REQ_CNT_QUOTA_PER_DAY_OFFSET = 0x03AB + ro_offset;
    int REQ_CNT_QUOTA_PER_DAY_LENGTH = 10;

    status = hasp_read(
        hasp_hdl(), HASP_FILEID_RO, REQ_CNT_QUOTA_PER_DAY_OFFSET, REQ_CNT_QUOTA_PER_DAY_LENGTH,
        buffer);
    if (status != HASP_STATUS_OK) {
      LOG_INFO << "Fetch req cnt quota from License failed, ret code: "
                << status;
      return false;
    }

    if (strlen(buffer) > 0 && !safe_lexical_cast(buffer, max_req_cnt_per_day)) {
      LOG_INFO << "Invalid req cnt quota per day in License, " << buffer;
      return false;
    }
    // if max_req_cnt_per_day equal 0 then disable req cnt quota check
    req_cnt_quota_per_day() = max_req_cnt_per_day;
    LOG_INFO << "Request Count Quota Per Day:" << req_cnt_quota_per_day();

    status = hasp_read(
        hasp_hdl(), HASP_FILEID_RW, ACCU_REQ_CNT_PER_DAY_OFFSET, ACCU_REQ_CNT_PER_DAY_LENGTH,
        &accu_req_cnt_per_day);
    if (status != HASP_STATUS_OK) {
      LOG_INFO << "Fetch accu req cnt per day from License failed, ret code: "
                << status;
      return false;
    }

    if (max_req_cnt_per_day > 0 && accu_req_cnt_per_day >= max_req_cnt_per_day) {
      LOG_INFO << "Exceeded Request Count Quota Per Day, Please Get Auth Again.";
      return false;
    } else {
      LOG_INFO << "Succ to Check Request Count Quota Per Day";
    }

    // req_cnt_quota() = 100;
    reset_req_cnt();
    reset_req_cnt_per_day();

#ifdef TRITON_ENABLE_GPU
    const size_t MAX_ALLOWED_GPU_NUM_OFFSET = 0 + ro_offset;
    const size_t MAX_ALLOWED_GPU_NUM_LENGTH = 2;
    // const size_t GPUID_OFFSET = 0 + rw_offset;
    // const size_t SINGLE_GPUID_LENGTH = 40;
    // const int MAX_GPUID_NUM = 20;
    // const size_t TOTAL_GPUID_LENGTH =
    //     (SINGLE_GPUID_LENGTH + 1) * MAX_GPUID_NUM - 1;

    // check gpu quota
    int max_allowed_gpu_num = 0;
    status = hasp_read(
        hasp_hdl(), HASP_FILEID_RO, MAX_ALLOWED_GPU_NUM_OFFSET,
        MAX_ALLOWED_GPU_NUM_LENGTH, buffer);
    buffer[MAX_ALLOWED_GPU_NUM_LENGTH] = '\0';
    if (status != HASP_STATUS_OK) {
      LOG_INFO << "Fetch GPU quota from License failed, ret code: " << status;
      return false;
    }
    if (!safe_lexical_cast(buffer, max_allowed_gpu_num)) {
      LOG_INFO << "Invalid GPU quota in License, " << buffer;
      return false;
    }
    qps_quota_cnt_limit() = max_allowed_gpu_num * qps_quota_cnt_limit();

    int gpu_device_cnt = 0;
    GetGpuDeviceCnt(gpu_device_cnt);
    LOG_INFO << "GPU quota in License: " << max_allowed_gpu_num;
    LOG_INFO << "Actually GPU num on local machine: " << gpu_device_cnt;
    if (max_allowed_gpu_num < gpu_device_cnt) {
      LOG_INFO << "Exceeded GPU quota";
      return false;
    } else {
      LOG_INFO << "Check GPU quota succ";
    }
#endif
    stop_monitor_thread() = false;
    monitor_thread() = std::thread(renew_lic_thread_runner);

    return true;
  }

  static void renew_lic_thread_runner()
  {
    bool is_first_renew = true;
    hasp_status_t status;
    char* session_info = nullptr;
    int timeout = 0;

    int rw_offset = 0;
    int accu_req_cnt = 0;
    int ACCU_REQ_CNT_OFFSET = 0x06B8 + rw_offset;
    int ACCU_REQ_CNT_LENGTH = 4;

    int accu_req_cnt_per_day = 0;
    int ACCU_REQ_CNT_PER_DAY_OFFSET = 0x06C2 + rw_offset;
    int ACCU_REQ_CNT_PER_DAY_LENGTH = 4;

    // int max_req_cnt = req_cnt_quota();
    int REQ_CNT_INTERVAL = 1000;
    int REQ_CNT_PER_DAY_INTERVAL = 100;

    auto last_renew_time = std::chrono::system_clock::now() -
                           std::chrono::seconds(qps_quota_renew_interval());
    while (!stop_monitor_thread()) {
      std::this_thread::sleep_for(std::chrono::seconds(monitor_interval_sec()));

      // qps
      if (qps_quota_renew_interval() != 0 && qps_quota_cnt_limit() != 0) {
        auto cur_renew_time = std::chrono::system_clock::now();
        std::chrono::duration<double> duration =
            cur_renew_time - last_renew_time;
        if (duration.count() >= qps_quota_renew_interval()) {
          std::unique_lock<std::mutex> lock(qps_mutex());
          last_renew_time = cur_renew_time;
          qps_quota_remain() = qps_quota_cnt_limit();
          qps_cv().notify_all();
        }
      }

      int max_req_cnt = req_cnt_quota();
      // check req cnt quota
      if (max_req_cnt > 0 && req_cnt() >= REQ_CNT_INTERVAL) {
        status = hasp_read(
            hasp_hdl(), HASP_FILEID_RW, ACCU_REQ_CNT_OFFSET,
            ACCU_REQ_CNT_LENGTH, &accu_req_cnt);
        if (status != HASP_STATUS_OK) {
          LOG_INFO << "Fetch accu req cnt from License failed, ret code: "
                    << status;
          break;
        }

        accu_req_cnt += REQ_CNT_INTERVAL;
        status = hasp_write(
            hasp_hdl(), HASP_FILEID_RW, ACCU_REQ_CNT_OFFSET,
            ACCU_REQ_CNT_LENGTH, &accu_req_cnt);
        if (status != HASP_STATUS_OK) {
          LOG_INFO << "Update accu req cnt " << accu_req_cnt
                    << " to License failed, ret code: " << status;
          break;
        }
        if (accu_req_cnt >= max_req_cnt) {
          LOG_INFO << "ERROR: req cnt exceed quota " << max_req_cnt
                    << " .PLEASE GET AUTH AGAIN!";
          deauth();
          raise(SIGINT);
        } else {
          reset_req_cnt();
        }
      }

      // check req cnt quota per day
      int max_req_cnt_per_day = req_cnt_quota_per_day();
      if (max_req_cnt_per_day > 0){
        std::chrono::system_clock::time_point today = std::chrono::system_clock::now();
        time_t tt = std::chrono::system_clock::to_time_t ( today );
        struct tm *p = localtime(&tt);
        if(p->tm_hour == 0 && p->tm_min == 0 && p->tm_sec < 5){
          reset_req_cnt_per_day();
          accu_req_cnt_per_day = 0;
          status = hasp_write(
              hasp_hdl(), HASP_FILEID_RW, ACCU_REQ_CNT_PER_DAY_OFFSET,
              ACCU_REQ_CNT_PER_DAY_LENGTH, &accu_req_cnt_per_day);
          if (status != HASP_STATUS_OK) {
            LOG_INFO << "0:0:0 Reset accu req cnt per day to License failed, ret code: " << status;
            break;
          }
        }
      }

      if (max_req_cnt_per_day > 0 && req_cnt_per_day() >= REQ_CNT_PER_DAY_INTERVAL) {
        status = hasp_read(
            hasp_hdl(), HASP_FILEID_RW, ACCU_REQ_CNT_PER_DAY_OFFSET,
            ACCU_REQ_CNT_PER_DAY_LENGTH, &accu_req_cnt_per_day);
        if (status != HASP_STATUS_OK) {
          LOG_INFO << "Fetch accu req cnt per day from License failed, ret code: "
                    << status;
          break;
        }

        accu_req_cnt_per_day += REQ_CNT_PER_DAY_INTERVAL;
        status = hasp_write(
            hasp_hdl(), HASP_FILEID_RW, ACCU_REQ_CNT_PER_DAY_OFFSET,
            ACCU_REQ_CNT_PER_DAY_LENGTH, &accu_req_cnt_per_day);
        if (status != HASP_STATUS_OK) {
          LOG_INFO << "Update accu req cnt per day " << accu_req_cnt_per_day
                    << " to License failed, ret code: " << status;
          break;
        }
        if (accu_req_cnt_per_day >= max_req_cnt_per_day) {
          LOG_INFO << "ERROR: req cnt per day exceed quota " << max_req_cnt_per_day
                    << " .PLEASE GET AUTH AGAIN!";
          deauth();
          raise(SIGINT);
        } else {
          reset_req_cnt_per_day();
        }
      }

      timeout -= monitor_interval_sec();
      if (timeout > 0)
        continue;
      timeout = 0;
      if (session_info != nullptr) {
        hasp_free(session_info);
        session_info = nullptr;
      }

      if (stop_monitor_thread())
        break;

      if (!is_first_renew) {
        hasp_logout(hasp_hdl());
        status =
            hasp_login(LICENSE_FEATURE_ID, LICENSE_VENDOR_CODE, &(hasp_hdl()));
        if (status != HASP_STATUS_OK) {
          LOG_INFO << "WARNING: renew license failed, login with ret code: "
                    << status;
          LOG_INFO << "ERROR: LICENSE IS EXPIRED, PLEASE GET AUTH AGAIN!";
          raise(SIGINT);
        }

        status = hasp_read(
            hasp_hdl(), HASP_FILEID_RW, ACCU_REQ_CNT_OFFSET,
            ACCU_REQ_CNT_LENGTH, &accu_req_cnt);
        if (status != HASP_STATUS_OK) {
          LOG_INFO << "Fetch accu req cnt from License failed, ret code: "
                    << status;
          hasp_logout(hasp_hdl());
          break;
        }
        if(max_req_cnt > 0 && accu_req_cnt >= max_req_cnt){
          LOG_INFO << "ERROR: req cnt exceed quota " << max_req_cnt
                    << " .PLEASE GET AUTH AGAIN!";
          hasp_logout(hasp_hdl());
          break;
        }
        LOG_INFO << "renew license success";
      } else {
        // skip logout and login op in first renew process
        is_first_renew = false;
      }

      status =
          hasp_get_sessioninfo(hasp_hdl(), HASP_SESSIONINFO, &session_info);
      if (status != HASP_STATUS_OK) {
        LOG_INFO << "WARNING: renew license failed, cannot get session info, "
                     "ret code: "
                  << status;
        continue;
      }

      std::string session_info_bak(session_info);

      rapidxml::xml_document<> session_info_doc;
      session_info_doc.parse<0>(session_info);
      rapidxml::xml_node<>* sess_info_root =
          session_info_doc.first_node("hasp_info");
      if (sess_info_root == nullptr) {
        LOG_INFO << "WARNING: renew license failed, parse session info "
                     "failed, cannot get root node";
        continue;
      }

      rapidxml::xml_node<>* feature_node =
          sess_info_root->first_node("feature");
      if (feature_node == nullptr) {
        LOG_INFO << "WARNING: renew license failed, parse session info "
                     "failed, cannot get feature node";
        continue;
      }

      // for remote cl mode, session info is not exists
      rapidxml::xml_node<>* session_node = feature_node->first_node("session");
      if (session_node == nullptr) {
        // without session info
        LOG_INFO << "USING REMOTE CL MODE";
        // timeout = 43200;
        timeout = 100;
        continue;
      } else {
        // with session info, local cl mode or non-cl mode
        rapidxml::xml_node<>* timeout_node =
            session_node->first_node("idle_remaining");
        if (timeout_node == nullptr) {
          LOG_INFO << "WARNING: renew license failed, parse session info "
                       "failed, cannot get idle_remaining node";
          continue;
        }
        if (!safe_lexical_cast(timeout_node->value(), timeout)) {
          LOG_INFO
              << "WARNING: renew license failed, cannot cast timeout to int";
          continue;
        }
      }

    }  // while

    deauth();
    raise(SIGINT);
  }

  static std::string rtrim(const std::string& s)
  {
    const std::string WHITESPACE = " \n\r\t\f\v";
    size_t end = s.find_last_not_of(WHITESPACE);
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
  }

  static std::string& get_machine()
  {
    static std::string machine = "";
    return machine;
  }
  static std::string& license_pc_code()
  {
    static std::string g_pc_code = "";
    return g_pc_code;
  }

  static std::unique_ptr<license::LicenseClient>& get_dclient()
  {
    static std::unique_ptr<license::LicenseClient> g_client =
        std::unique_ptr<license::LicenseClient>(new license::LicenseClient(""));
    return g_client;
  }

  static bool dataelem_login()
  {
    std::string hostname = "", ipaddr = "";
    if (!get_machine_info(hostname, ipaddr)) {
      LOG_INFO << "License get machine info failed!";
      return false;
    }

    uint64_t sec = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
    std::string timestamp = std::to_string(int(sec) % 10000);

    get_machine() = hostname + ":" + timestamp;
    license::Conn conn;
    conn.machine = get_machine();
    conn.product = get_product_name();
    conn.address = ipaddr;
    conn.host = hostname;
    conn.timestamp = int(sec);
    if (!get_dclient()->license_login(conn)) {
      LOG_INFO << "License login failed!";
      return false;
    }

    license::KeyInfo key_info;
    if (!get_dclient()->license_read(key_info)) {
      LOG_INFO << "License read info failed!";
      return false;
    }

    license_pc_code() = key_info.pc_code;
    if (key_info.expire_date == "99999999") {
      LOG_INFO << "License Info: Lifecycle Use";
    } else {
      LOG_INFO << "License Info: expire_date:" << key_info.expire_date;
    }

    LOG_INFO << "feat_id:" << key_info.feat_id
              << " qps_quota:" << key_info.qps_quota 
              << " scene_quota:" << key_info.scene_quota
              << " stations:" << key_info.stations
              << " gpu_num:" << key_info.gpu_num;


    if (key_info.feat_id < 0 || key_info.feat_id > 128) {
      LOG_INFO << "feat id error:" << key_info.feat_id;
      return false;
    }

    if (key_info.qps_quota < 0 || key_info.qps_quota > 16000) {
      LOG_INFO << "qps_quota error:" << key_info.qps_quota;
      return false;
    }

    if (key_info.scene_quota < 0 || key_info.scene_quota > 10000) {
      LOG_INFO << "scene_quota error:" << key_info.scene_quota;
      return false;
    }

    if (key_info.stations < 0 || key_info.stations > 30) {
      LOG_INFO << "stations error:" << key_info.stations;
      return false;
    }

    if (key_info.gpu_num < 0 || key_info.gpu_num > 10) {
      LOG_INFO << "gpu_num error:" << key_info.gpu_num;
      return false;
    }

    req_cnt_quota() = key_info.qps_quota * 10000;
    qps_quota_renew_interval() = 0;

    license::DynamicInfo dyi;
    if (!get_dclient()->license_read_dyn(license_pc_code(), dyi)) {
      LOG_INFO << "Fetch accu req cnt from License failed";
      return false;
    }

    int max_req_cnt = req_cnt_quota();
    int accu_req_cnt = dyi.value1;
    if (accu_req_cnt >= max_req_cnt && max_req_cnt != 0) {
      LOG_INFO << "ERROR: req cnt exceed quota " << max_req_cnt
                    << " .PLEASE GET AUTH AGAIN!";
      return false;
    }

    return true;
  }

  static void dataelem_logout()
  {
    std::string hostname = "", ipaddr = "";
    if (!get_machine_info(hostname, ipaddr)) {
      LOG_INFO << "License get machine info failed!";
      return;
    }
    uint64_t sec = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
    license::Conn conn;
    conn.machine = get_machine();
    conn.product = get_product_name();
    conn.address = ipaddr;
    conn.host = hostname;
    conn.timestamp = int(sec);
    if (!get_dclient()->license_logout(conn)) {
      LOG_INFO << "License logout failed!";
    } else {
      LOG_INFO << "Logout dataelem license";
    }

    get_machine() = "";
    license_pc_code() = "";
  }


  static bool get_machine_info(std::string& hostname, std::string& ipaddr)
  {
    char host[256];
    int host_id = gethostname(host, sizeof(host));
    if (host_id == -1) {
      return false;
    }

    struct hostent* host_entry = gethostbyname(host);
    if (host_entry == NULL) {
      return false;
    }

    char* ip = inet_ntoa(*((struct in_addr*)host_entry->h_addr_list[0]));
    if (ip == NULL) {
      ipaddr = "0.0.0.0";
    } else {
      ipaddr = std::string(ip);
    }

    hostname = std::string(host);
    return true;
  }

  static std::string get_product_name()
  {
    return "Dataelem Studio专业版-GPU版";
  }

  static void renew_lic_thread_runner_dataelem()
  {
    int timeout = 43200;
    int timeout_heart = 10;
    int accu_req_cnt = 0;
    license::DynamicInfo dyi;
    int REQ_CNT_INTERVAL = 1000;
    auto last_renew_time = std::chrono::system_clock::now() -
                           std::chrono::seconds(qps_quota_renew_interval());
    while (!stop_monitor_thread()) {
      std::this_thread::sleep_for(std::chrono::seconds(monitor_interval_sec()));

      // qps
      if (qps_quota_renew_interval() != 0 && qps_quota_cnt_limit() != 0) {
        auto cur_renew_time = std::chrono::system_clock::now();
        std::chrono::duration<double> duration =
            cur_renew_time - last_renew_time;
        if (duration.count() >= qps_quota_renew_interval()) {
          std::unique_lock<std::mutex> lock(qps_mutex());
          last_renew_time = cur_renew_time;
          qps_quota_remain() = qps_quota_cnt_limit();
          qps_cv().notify_all();
        }
      }

      int max_req_cnt = req_cnt_quota();
      // check req cnt quota
      if (max_req_cnt > 0 && req_cnt() >= REQ_CNT_INTERVAL) {
        if (!get_dclient()->license_read_dyn(license_pc_code(), dyi)) {
          LOG_INFO << "Fetch accu req cnt from License failed";
          break;
        }
        accu_req_cnt = dyi.value1;
        accu_req_cnt += REQ_CNT_INTERVAL;
        dyi.value1 = accu_req_cnt;
        if (!get_dclient()->license_update_dyn(license_pc_code(), dyi)) {
          LOG_INFO << "Update accu req cnt " << accu_req_cnt
                    << " to License failed";
          break;
        }
        if (accu_req_cnt >= max_req_cnt) {
          LOG_INFO << "ERROR: req cnt exceed quota " << max_req_cnt
                    << " .PLEASE GET AUTH AGAIN!";
          deauth_dataelem();
          raise(SIGINT);
        } else {
          reset_req_cnt();
        }
      }

      timeout_heart -= monitor_interval_sec();
      if(timeout_heart <= 0){
        timeout_heart = 10;
        license::Conn conn;
        conn.machine = get_machine();
        get_dclient()->license_heart(conn);
      }

      timeout -= monitor_interval_sec();
      if (timeout > 0)
        continue;
      timeout = 43200;

      if (stop_monitor_thread())
        break;

      dataelem_logout();
      if (!dataelem_login()) {
        LOG_INFO << "WARNING: renew license failed ";
        LOG_INFO << "ERROR: LICENSE IS EXPIRED, PLEASE GET AUTH AGAIN!";
        raise(SIGINT);
      }
      LOG_INFO << "renew license success";
    }  // while

    deauth_dataelem();
    raise(SIGINT);
  }

  static bool GetGpuDeviceCnt(int& device_cnt)
  {
#ifdef TRITON_ENABLE_GPU
    cudaError_t cuerr = cudaGetDeviceCount(&device_cnt);
    if ((cuerr == cudaErrorNoDevice) ||
        (cuerr == cudaErrorInsufficientDriver)) {
      device_cnt = 0;
    } else if (cuerr != cudaSuccess) {
      return false;
    }
#else
    device_cnt = 0;
#endif  // TRITON_ENABLE_GPU

    return true;
  }
};

}}  // namespace dataelem::common
