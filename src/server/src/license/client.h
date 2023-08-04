#include <string>

namespace license {

struct Conn {
  std::string machine;
  std::string product;
  std::string address;
  std::string host;
  int timestamp;
};

struct DynamicInfo {
  int value1;
  int value2;
  int value3;
  int value4;
  int value5;
  int value6;
  int value7;
  int value8;
  int value9;
};

struct KeyInfo {
  std::string pc_code;
  std::string expire_date;
  int feat_id;
  int qps_quota;
  int scene_quota;
  int stations;
  int gpu_num;
  std::string reserver1;
  std::string reserver2;
  std::string reserver3;
  std::string reserver4;
  std::string reserver5;
};

class LicenseClient {
 public:
  LicenseClient(const std::string& url) : url_(url){};
  void set_url(const std::string& url) { url_ = url; };
  bool license_read(KeyInfo& info);
  bool license_login(const Conn& conn);
  bool license_logout(const Conn& conn);
  bool license_heart(const Conn& conn);
  bool license_update_dyn(const std::string& pc_code, const DynamicInfo& info);
  bool license_read_dyn(const std::string& pc_code, DynamicInfo& info);

 private:
  std::string url_;
};

}  // namespace license