#ifndef DATAELEM_FRAMEWORK_APP_FACTORY_H_
#define DATAELEM_FRAMEWORK_APP_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "dataelem/framework/app.h"
#include "triton/backend/backend_common.h"

namespace dataelem { namespace alg {

class AppRegistry {
 public:
  typedef Application* (*Creator)();
  typedef std::map<std::string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry()
  {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const std::string& type, Creator creator)
  {
    CreatorRegistry& registry = Registry();
    if (registry.count(type) != 0) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("app type: ") + type + " already registed.").c_str());
    }
    registry[type] = creator;
  }

  static Application* CreateApp(const std::string& type)
  {
    CreatorRegistry& registry = Registry();
    if (registry.count(type) != 1) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("unknown app type:") + type).c_str());
      return nullptr;
    }

    return registry[type]();
  }

  static std::vector<std::string> TypeList()
  {
    CreatorRegistry& registry = Registry();
    std::vector<std::string> types;
    for (auto iter = registry.begin(); iter != registry.end(); ++iter) {
      types.push_back(iter->first);
    }
    return types;
  }

 private:
  // Solver registry should never be instantiated - everything is done with its
  // static variables.
  AppRegistry() {}

  static std::string TypeListString()
  {
    std::vector<std::string> types = TypeList();
    std::string types_str;
    for (auto iter = types.begin(); iter != types.end(); ++iter) {
      if (iter != types.begin()) {
        types_str += ", ";
      }
      types_str += *iter;
    }
    return types_str;
  }
};

class AppRegisterer {
 public:
  AppRegisterer(const std::string& type, Application* (*creator)())
  {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("registering app type: ") + type).c_str());
    AppRegistry::AddCreator(type, creator);
  }
};

#define REGISTER_APP_CREATOR(type, creator) \
  static AppRegisterer gCreator##type(#type, creator);

#define REGISTER_APP_CLASS(type)                      \
  Application* Creator##type() { return new type(); } \
  REGISTER_APP_CREATOR(type, Creator##type)

}}  // namespace dataelem::alg

#endif  // DATAELEM_FRAMEWORK_APP_FACTORY_H_
