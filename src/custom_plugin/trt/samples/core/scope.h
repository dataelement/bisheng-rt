#ifndef TRT_ISCOPE_H
#define TRT_ISCOPE_H

#include <string>
#include <iostream>
#include <map>
#include <unordered_map>

class IScope {

 public:
  IScope() {
    _scope_name = "";
  }

  IScope(const std::string& name) {
    _scope_name = name;
  }

  std::string getOpName() const {
    return _scope_name + ":0";
  }

  std::string getScopeName() const {
    return _scope_name;
  }

  IScope& operator=(const IScope& other) {
    _scope_name = other.getScopeName();
    return *this;
  }

  IScope subIScope(const std::string& child_scope_name) const;

  void updateNameMap(std::string name);

  ~IScope() {}

 private:
  std::string _scope_name;
  //TODO multi op support
  std::unordered_map<std::string, int> name_map;
};

#endif
