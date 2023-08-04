#include "scope.h"

// scope
IScope IScope::subIScope(const std::string& child_scope_name) const {
  std::string new_name;
  if (_scope_name != "") {
    new_name = _scope_name + "/" + child_scope_name;
  } else {
    new_name = child_scope_name;
  }
  return IScope(new_name);
}

void IScope::updateNameMap(std::string name) {
  auto flag = name_map.find(name);
  if (flag != name_map.end()) {
    name_map[name] += 1;
  } else {
    name_map[name] = 0;
  }
}

