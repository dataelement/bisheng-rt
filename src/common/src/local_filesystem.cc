#include "triton/common/local_filesystem.h"

#ifdef _WIN32
// suppress the min and max definitions in Windef.h.
#define NOMINMAX
#include <Windows.h>

// _CRT_INTERNAL_NONSTDC_NAMES 1 before including Microsoft provided C Runtime
// library to expose declarations without "_" prefix to match POSIX style.
#define _CRT_INTERNAL_NONSTDC_NAMES 1
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cerrno>
#include <fstream>
#include <mutex>

constexpr uint64_t NANOS_PER_SECOND = 1000000000;
constexpr uint64_t NANOS_PER_MILLIS = 1000000;

#define TIMESPEC_TO_NANOS(TS) ((TS).tv_sec * NANOS_PER_SECOND + (TS).tv_nsec)
#define TIMESPEC_TO_MILLIS(TS) (TIMESPEC_TO_NANOS(TS) / NANOS_PER_MILLIS)

// If status is non-OK, return the Status.
#define RETURN_IF_ERROR(S)        \
  do {                            \
    const Status& status__ = (S); \
    if (!status__.IsOk()) {       \
      return status__;            \
    }                             \
  } while (false)


#ifdef _WIN32
// <sys/stat.h> in Windows doesn't define S_ISDIR macro
#if !defined(S_ISDIR) && defined(S_IFMT) && defined(S_IFDIR)
#define S_ISDIR(m) (((m)&S_IFMT) == S_IFDIR)
#endif
#define F_OK 0
#endif

namespace triton { namespace common {


// FIXME: Windows support '/'? If so, the below doesn't need to change
bool
IsAbsolutePath(const std::string& path)
{
  return !path.empty() && (path[0] == '/');
}

std::string
JoinPath(std::initializer_list<std::string> segments)
{
  std::string joined;

  for (const auto& seg : segments) {
    if (joined.empty()) {
      joined = seg;
    } else if (IsAbsolutePath(seg)) {
      if (joined[joined.size() - 1] == '/') {
        joined.append(seg.substr(1));
      } else {
        joined.append(seg);
      }
    } else {  // !IsAbsolutePath(seg)
      if (joined[joined.size() - 1] != '/') {
        joined.append("/");
      }
      joined.append(seg);
    }
  }

  return joined;
}

std::string
BaseName(const std::string& path)
{
  if (path.empty()) {
    return path;
  }

  size_t last = path.size() - 1;
  while ((last > 0) && (path[last] == '/')) {
    last -= 1;
  }

  if (path[last] == '/') {
    return std::string();
  }

  const size_t idx = path.find_last_of("/", last);
  if (idx == std::string::npos) {
    return path.substr(0, last + 1);
  }

  return path.substr(idx + 1, last - idx);
}

std::string
DirName(const std::string& path)
{
  if (path.empty()) {
    return path;
  }

  size_t last = path.size() - 1;
  while ((last > 0) && (path[last] == '/')) {
    last -= 1;
  }

  if (path[last] == '/') {
    return std::string("/");
  }

  const size_t idx = path.find_last_of("/", last);
  if (idx == std::string::npos) {
    return std::string(".");
  }
  if (idx == 0) {
    return std::string("/");
  }

  return path.substr(0, idx);
}

Status
IsPathDirectory(const std::string& path, bool* is_dir)
{
  *is_dir = false;

  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    return Status(Status::Code::INTERNAL, "failed to stat file " + path);
  }

  *is_dir = S_ISDIR(st.st_mode);
  return Status::Success;
}

Status
FileExists(const std::string& path, bool* exists)
{
  *exists = (access(path.c_str(), F_OK) == 0);
  return Status::Success;
}

Status
IsDirectory(const std::string& path, bool* is_dir)
{
  return IsPathDirectory(path, is_dir);
}

Status
FileModificationTime(const std::string& path, int64_t* mtime_ns)
{
  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    return Status(Status::Code::INTERNAL, "failed to stat file " + path);
  }

#ifdef _WIN32
  // In Windows, st_mtime is in time_t
  *mtime_ns = st.st_mtime;
#else
  *mtime_ns = TIMESPEC_TO_NANOS(st.st_mtim);
#endif
  return Status::Success;
}

Status
GetDirectoryContents(const std::string& path, std::set<std::string>* contents)
{
#ifdef _WIN32
  WIN32_FIND_DATA entry;
  // Append "*" to obtain all files under 'path'
  HANDLE dir = FindFirstFile(JoinPath({path, "*"}).c_str(), &entry);
  if (dir == INVALID_HANDLE_VALUE) {
    return Status(Status::Code::INTERNAL, "failed to open directory " + path);
  }
  if ((strcmp(entry.cFileName, ".") != 0) &&
      (strcmp(entry.cFileName, "..") != 0)) {
    contents->insert(entry.cFileName);
  }
  while (FindNextFile(dir, &entry)) {
    if ((strcmp(entry.cFileName, ".") != 0) &&
        (strcmp(entry.cFileName, "..") != 0)) {
      contents->insert(entry.cFileName);
    }
  }

  FindClose(dir);
#else
  DIR* dir = opendir(path.c_str());
  if (dir == nullptr) {
    return Status(Status::Code::INTERNAL, "failed to open directory " + path);
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string entryname = entry->d_name;
    if ((entryname != ".") && (entryname != "..")) {
      contents->insert(entryname);
    }
  }

  closedir(dir);
#endif
  return Status::Success;
}

Status
GetDirectorySubdirs(const std::string& path, std::set<std::string>* subdirs)
{
  RETURN_IF_ERROR(GetDirectoryContents(path, subdirs));

  // Erase non-directory entries...
  for (auto iter = subdirs->begin(); iter != subdirs->end();) {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(JoinPath({path, *iter}), &is_dir));
    if (!is_dir) {
      iter = subdirs->erase(iter);
    } else {
      ++iter;
    }
  }

  return Status::Success;
}

Status
GetDirectoryFiles(const std::string& path, std::set<std::string>* files)
{
  RETURN_IF_ERROR(GetDirectoryContents(path, files));

  // Erase directory entries...
  for (auto iter = files->begin(); iter != files->end();) {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(JoinPath({path, *iter}), &is_dir));
    if (is_dir) {
      iter = files->erase(iter);
    } else {
      ++iter;
    }
  }

  return Status::Success;
}

Status
ReadTextFile(const std::string& path, std::string* contents)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    return Status(
        Status::Code::INTERNAL,
        "failed to open text file for read " + path + ": " + strerror(errno));
  }

  in.seekg(0, std::ios::end);
  contents->resize(in.tellg());
  in.seekg(0, std::ios::beg);
  in.read(&(*contents)[0], contents->size());
  in.close();

  return Status::Success;
}


Status
WriteTextFile(const std::string& path, const std::string& contents)
{
  std::ofstream out(path, std::ios::out | std::ios::binary);
  if (!out) {
    return Status(
        Status::Code::INTERNAL,
        "failed to open text file for write " + path + ": " + strerror(errno));
  }

  out.write(&contents[0], contents.size());
  out.close();

  return Status::Success;
}

Status
WriteBinaryFile(
    const std::string& path, const char* contents, const size_t content_len)
{
  std::ofstream out(path, std::ios::out | std::ios::binary);
  if (!out) {
    return Status(
        Status::Code::INTERNAL, "failed to open binary file for write " + path +
                                    ": " + strerror(errno));
  }

  out.write(contents, content_len);

  return Status::Success;
}

Status
MakeDirectory(const std::string& dir, const bool recursive)
{
#ifdef _WIN32
  if (mkdir(dir.c_str()) == -1)
#else
  if (mkdir(dir.c_str(), S_IRWXU) == -1)
#endif
  {
    // Only allow the error due to parent directory does not exist
    // if 'recursive' is requested
    if ((errno == ENOENT) && (!dir.empty()) && recursive) {
      RETURN_IF_ERROR(MakeDirectory(DirName(dir), recursive));
      // Retry the creation
#ifdef _WIN32
      if (mkdir(dir.c_str()) == -1)
#else
      if (mkdir(dir.c_str(), S_IRWXU) == -1)
#endif
      {
        return Status(
            Status::Code::INTERNAL, "Failed to create directory '" + dir +
                                        "', errno:" + strerror(errno));
      }
    } else {
      return Status(
          Status::Code::INTERNAL,
          "Failed to create directory '" + dir + "', errno:" + strerror(errno));
    }
  }

  return Status::Success;
}

Status
MakeTemporaryDirectory(std::string* temp_dir)
{
#ifdef _WIN32
  char temp_path[MAX_PATH + 1];
  size_t temp_path_length = GetTempPath(MAX_PATH + 1, temp_path);
  if (temp_path_length == 0) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to get local directory for temporary files");
  }
  // There is no single operation like 'mkdtemp' in Windows, thus generating
  // unique temporary directory is a process of getting temporary file name,
  // deleting the file (file creation is side effect fo getting name), creating
  // corresponding directory, so mutex is used to avoid possible race condition.
  // However, it doesn't prevent other process on creating temporary file and
  // thus the race condition may still happen. One possible solution is
  // to reserve a temporary directory for the process and generate temporary
  // model directories inside it.
  static std::mutex mtx;
  std::lock_guard<std::mutex> lk(mtx);
  // Construct a std::string as filled 'temp_path' is not C string,
  // and so that we can reuse 'temp_path' to hold the temp file name.
  std::string temp_path_str(temp_path, temp_path_length);
  if (GetTempFileName(temp_path_str.c_str(), "folder", 0, temp_path) == 0) {
    return Status(Status::Code::INTERNAL, "Failed to create local temp folder");
  }
  *temp_dir = temp_path;
  DeleteFile(temp_dir->c_str());
  if (CreateDirectory(temp_dir->c_str(), NULL) == 0) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to create local temp folder: " + *temp_dir);
  }
#else
  std::string folder_template = "/tmp/folderXXXXXX";
  char* res = mkdtemp(const_cast<char*>(folder_template.c_str()));
  if (res == nullptr) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to create local temp folder: " + folder_template +
            ", errno:" + strerror(errno));
  }
  *temp_dir = res;
#endif
  return Status::Success;
}

Status
DeleteDirectory(const std::string& path)
{
  std::set<std::string> contents;
  RETURN_IF_ERROR(GetDirectoryContents(path, &contents));

  for (const auto& content : contents) {
    std::string full_path = JoinPath({path, content});
    bool is_dir = false;
    RETURN_IF_ERROR(IsDirectory(full_path, &is_dir));
    if (is_dir) {
      DeleteDirectory(full_path);
    } else {
      remove(full_path.c_str());
    }
  }
  rmdir(path.c_str());

  return Status::Success;
}

bool
FileExists(const std::string& path)
{
  return (access(path.c_str(), F_OK) == 0);
}

bool
IsDirectory(const std::string& path)
{
  bool is_dir = false;
  IsPathDirectory(path, &is_dir);
  return is_dir;
}


}}  // namespace triton::common
