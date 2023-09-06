#pragma once

#include <set>
#include <string>

#include "triton/common/error.h"

namespace triton { namespace common {

// typedef Error Status;

/// Is a path an absolute path?
/// \param path The path.
/// \return true if absolute path, false if relative path.
bool IsAbsolutePath(const std::string& path);

/// Join path segments into a longer path
/// \param segments The path segments.
/// \return the path formed by joining the segments.
std::string JoinPath(std::initializer_list<std::string> segments);

/// Get the basename of a path.
/// \param path The path.
/// \return the last segment of the path.
std::string BaseName(const std::string& path);

/// Get the dirname of a path.
/// \param path The path.
/// \return all but the last segment of the path.
std::string DirName(const std::string& path);

/// Does a file or directory exist?
/// \param path The path to check for existance.
/// \param exists Returns true if file/dir exists
/// \return Error status if unable to perform the check
Status FileExists(const std::string& path, bool* exists);

/// Is a path a directory?
/// \param path The path to check.
/// \param is_dir Returns true if path represents a directory
/// \return Error status
Status IsDirectory(const std::string& path, bool* is_dir);

/// Get file modification time in nanoseconds.
/// \param path The path.
/// \param mtime_ns Returns the file modification time. For some filesystems a
/// file/folder may not have a modification time, in that case return 0.
/// \return Error status
Status FileModificationTime(const std::string& path, int64_t* mtime_ns);

/// Get the contents of a directory.
/// \param path The directory path.
/// \param subdirs Returns the directory contents.
/// \return Error status
Status GetDirectoryContents(
    const std::string& path, std::set<std::string>* contents);

/// Get the sub-directories of a path.
/// \param path The path.
/// \param subdirs Returns the names of the sub-directories.
/// \return Error status
Status GetDirectorySubdirs(
    const std::string& path, std::set<std::string>* subdirs);

/// Get the files contained in a directory.
/// \param path The directory.
/// \param skip_hidden_files Ignores the hidden files in the directory.
/// \param files Returns the names of the files.
/// \return Error status
Status GetDirectoryFiles(
    const std::string& path, const bool skip_hidden_files,
    std::set<std::string>* files);

/// Read a text file into a string.
/// \param path The path of the file.
/// \param contents Returns the contents of the file.
/// \return Error status
Status ReadTextFile(const std::string& path, std::string* contents);

/// Write a string to a file.
/// \param path The path of the file.
/// \param contents The contents to write to the file.
/// \return Error status
Status WriteTextFile(const std::string& path, const std::string& contents);

/// Write binary to a file.
/// \param path The path of the file.
/// \param contents The contents to write to the file.
/// \param content_len The size of the content.
/// \return Error status
Status WriteBinaryFile(
    const std::string& path, const char* contents, const size_t content_len);

/// Create a directory of the specified path.
/// \param dir The path to the directory.
/// \param recursive Whether the parent directories will be created
/// if not exist.
/// \return Error status if the directory can't be created
Status MakeDirectory(const std::string& dir, const bool recursive);

/// Create a temporary directory of the specified filesystem type.
/// \param type The type of the filesystem.
/// \param temp_dir Returns the path to the temporary directory.
/// \return Error status
Status MakeTemporaryDirectory(std::string* temp_dir);

/// Delete a directory.
/// \param path Returns the path to the directory.
/// \return Error status
Status DeleteDirectory(const std::string& path);


class ScopedTemporaryDirectory {
 public:
  ScopedTemporaryDirectory() { MakeTemporaryDirectory(&temp_dir_); }
  ~ScopedTemporaryDirectory()
  {
    if (temp_dir_.rfind("/tmp/", 0) == 0) {
      DeleteDirectory(temp_dir_);
    }
  }
  const std::string Get() { return temp_dir_; }
  std::string temp_dir_ = "";
};

/// Does a file or directory exist?
/// \param path The path to check for existance.
/// \param exists Returns true if file/dir exists
/// \return Error status if unable to perform the check
bool FileExists(const std::string& path);

/// Is a path a directory?
/// \param path The path to check.
/// \param is_dir Returns true if path represents a directory
/// \return Error status
bool IsDirectory(const std::string& path);


}}  // namespace triton::common
