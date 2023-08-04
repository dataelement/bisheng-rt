#ifndef _AES_HPP_
#define _AES_HPP_

#ifndef __cplusplus
#error Do not include the hpp header in a c project!
#endif  //__cplusplus

extern "C" {
#include "aes.h"
}

#include <stdio.h>

#include <fstream>
#include <iostream>
#include <streambuf>

#include "picosha2.hpp"


struct MemBuffer : std::streambuf {
  MemBuffer(char* begin, char* end) { this->setg(begin, begin, end); }
};


namespace cipher {

#define DEFAULT_KEY_SIZE 32
typedef std::array<uint8_t, DEFAULT_KEY_SIZE> KeyBytes;
static const int32_t AES_BLOCK_SIZE = AES_BLOCKLEN;
static const int32_t AES_INIT_VECTOR_SIZE = AES_BLOCK_SIZE;


template <typename In>
int
DecryptAES(
    const KeyBytes& key_bytes, const uint32_t content_size,
    std::vector<In>& input_content)
{
  if (input_content.size() < AES_INIT_VECTOR_SIZE) {
    return -1;
  }
  struct AES_ctx ctx;
  const std::vector<uint8_t> iv_bytes(
      input_content.begin(), input_content.begin() + AES_INIT_VECTOR_SIZE);

  AES_init_ctx_iv(&ctx, key_bytes.data(), iv_bytes.data());

  input_content.erase(
      input_content.begin(), input_content.begin() + AES_INIT_VECTOR_SIZE);

  AES_CBC_decrypt_buffer(
      &ctx, reinterpret_cast<uint8_t*>(input_content.data()),
      content_size - AES_INIT_VECTOR_SIZE);

  const int size = input_content.size();
  const int last_index = (int)input_content[size - 1];

  if (last_index < 0 || last_index >= size) {
    return -2;
  }
  size_t size_without_padding = size - last_index;
  input_content.resize(size_without_padding);
  return 0;
}

inline void
genRandomBytes(std::vector<uint8_t>& bytes, int n)
{
  for (int i = 0; i < n; i++) {
    bytes[i] = rand() & 0xff;
  }
}

template <typename In>
int
EncryptAES(const KeyBytes& key_bytes, std::vector<In>& content)
{
  struct AES_ctx ctx;
  std::vector<uint8_t> iv_bytes(AES_BLOCK_SIZE);
  genRandomBytes(iv_bytes, AES_BLOCK_SIZE);

  AES_init_ctx_iv(&ctx, key_bytes.data(), iv_bytes.data());

  // padding the input
  int ori_size = content.size();
  int pad_len = DEFAULT_KEY_SIZE - ori_size % DEFAULT_KEY_SIZE;
  int new_size = ori_size + pad_len;
  if (pad_len > 0) {
    content.resize(new_size);
    for (int i = 0; i < pad_len; i++) {
      content[ori_size + i] = (In)pad_len;
    }
  }

  AES_CBC_encrypt_buffer(
      &ctx, reinterpret_cast<uint8_t*>(content.data()), new_size);
  const In* ptr = reinterpret_cast<In*>(iv_bytes.data());
  std::vector<In> out(ptr, ptr + AES_BLOCK_SIZE);
  out.insert(out.end(), content.begin(), content.end());
  content = out;
  return 0;
}

inline int
ReadAESBinary(const std::string& file_path, std::vector<char>& bytes)
{
  std::string private_key = "a9ad0566-cfb3-4427-8f8f-c672fa2b5fcf";
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.eof() && !file.fail()) {
    file.seekg(0, std::ios_base::end);
    std::streampos fileSize = file.tellg();
    bytes.resize(fileSize);
    file.seekg(0, std::ios_base::beg);
    file.read(&bytes[0], fileSize);
    file.close();
  } else {
    return -1;
  }
  cipher::KeyBytes hashKey;
  picosha2::hash256_bytes(private_key, hashKey);
  if (DecryptAES(hashKey, bytes.size(), bytes) != 0) {
    return -2;
  }
  return 0;
}

inline int
WriteAESBinary(const std::string& file_path, std::vector<char>& bytes)
{
  std::string private_key = "a9ad0566-cfb3-4427-8f8f-c672fa2b5fcf";
  cipher::KeyBytes hashKey;
  picosha2::hash256_bytes(private_key, hashKey);
  if (EncryptAES(hashKey, bytes) != 0) {
    return -1;
  }
  std::fstream file;
  file.open(file_path, std::ios::out | std::ios::binary);
  file.write(&bytes[0], bytes.size());
  file.close();
  return 0;
}

inline int
WriteSimpleEncBinary(const std::string& file_path, std::vector<char>& bytes)
{
  std::vector<int> RANDOM_PRIME_NUMS = {
      0,
      3,
      23,
      37,
      107,
      139,
      701,
      19273,
      192737,
      301927,
      541927,
      631927,
      761927,
      1000000 + 1927 + 0,
      2000000 + 1927 + 3,
      5000000 + 1927 + 23,
      10000000 + 1927 + 37,
      50000000 + 1927 + 107,
      100000000 + 1927 + 139,
      300000000 + 1927 + 701};

  std::vector<int> RANDOM_PRIME_INDEX = {4, 19, 15, 17, 7,  3, 10, 1,  8,  14,
                                         5, 16, 6,  2,  13, 0, 9,  18, 12, 11};

  const int size = bytes.size();

  int bias_index = 0;
  for (size_t i = 0; i < RANDOM_PRIME_NUMS.size(); i++) {
    if (RANDOM_PRIME_NUMS[i] >= size) {
      bias_index = i;
      break;
    }
  }
  if (bias_index > 1) {
    bias_index--;
  }
  if (bias_index == 0) {
    bias_index = RANDOM_PRIME_NUMS.size() - 1;
  }

  std::vector<int> sel_index;
  for (int& r : RANDOM_PRIME_INDEX) {
    if (r <= bias_index) {
      sel_index.push_back(r);
    }
  }

  const auto& last_bias = RANDOM_PRIME_NUMS[bias_index];

  std::fstream file;
  file.open(file_path, std::ios::out | std::ios::binary);
  for (auto& i : sel_index) {
    int s = RANDOM_PRIME_NUMS[i];
    int e = (s == last_bias) ? size : RANDOM_PRIME_NUMS[i + 1];
    if ((e - s) > 0) {
      file.write(&bytes[0] + s, e - s);
    };
  }
  file.close();
  return 0;
}

inline int
ReadSimpleEncBinary(const std::string& file_path, std::vector<char>& bytes)
{
  std::vector<int> RANDOM_PRIME_NUMS = {
      0,
      3,
      23,
      37,
      107,
      139,
      701,
      19273,
      192737,
      301927,
      541927,
      631927,
      761927,
      1000000 + 1927 + 0,
      2000000 + 1927 + 3,
      5000000 + 1927 + 23,
      10000000 + 1927 + 37,
      50000000 + 1927 + 107,
      100000000 + 1927 + 139,
      300000000 + 1927 + 701};

  std::vector<int> RANDOM_PRIME_INDEX = {4, 19, 15, 17, 7,  3, 10, 1,  8,  14,
                                         5, 16, 6,  2,  13, 0, 9,  18, 12, 11};

  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.eof() && !file.fail()) {
    file.seekg(0, std::ios_base::end);
    std::streampos file_size = file.tellg();
    int size = int(file_size);

    int bias_index = 0;
    for (size_t i = 0; i < RANDOM_PRIME_NUMS.size(); i++) {
      if (RANDOM_PRIME_NUMS[i] >= size) {
        bias_index = i;
        break;
      }
    }

    if (bias_index > 1) {
      bias_index--;
    }
    if (bias_index == 0) {
      bias_index = RANDOM_PRIME_NUMS.size() - 1;
    }

    std::vector<int> sel_index;
    for (int& r : RANDOM_PRIME_INDEX) {
      if (r <= bias_index) {
        sel_index.push_back(r);
      }
    }

    const auto& last_bias = RANDOM_PRIME_NUMS[bias_index];

    file.seekg(0, std::ios_base::beg);
    bytes.resize(file_size);
    for (auto& i : sel_index) {
      int s = RANDOM_PRIME_NUMS[i];
      int e = (s == last_bias) ? int(file_size) : RANDOM_PRIME_NUMS[i + 1];
      if (e - s > 0) {
        file.read(&bytes[0] + s, e - s);
      }
    }
    file.close();
  } else {
    return -1;
  }

  return 0;
}


inline int
write_plain_binary(const std::string& file_path, std::vector<char>& bytes)
{
  std::vector<int> RANDOM_PRIME_NUMS = {0, 3, 23, 37, 107, 139, 701};
  std::vector<int> RANDOM_PRIME_INDEX = {3, 2, 5, 1, 6, 4, 0};

  const int n = RANDOM_PRIME_INDEX.size();
  const auto& last_bias = RANDOM_PRIME_NUMS[n - 1];
  const int size = bytes.size();
  std::fstream file;
  file.open(file_path, std::ios::out | std::ios::binary);
  for (auto& i : RANDOM_PRIME_INDEX) {
    int s = RANDOM_PRIME_NUMS[i];
    int e = (s == last_bias) ? size : RANDOM_PRIME_NUMS[i + 1];
    file.write(&bytes[0] + s, e - s);
  }
  file.close();
  return 0;
}


inline int
read_plain_binary(const std::string& file_path, std::vector<char>& bytes)
{
  std::vector<int> RANDOM_PRIME_NUMS = {0, 3, 23, 37, 107, 139, 701};
  std::vector<int> RANDOM_PRIME_INDEX = {3, 2, 5, 1, 6, 4, 0};

  const int n = RANDOM_PRIME_INDEX.size();
  const auto& last_bias = RANDOM_PRIME_NUMS[n - 1];
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.eof() && !file.fail()) {
    file.seekg(0, std::ios_base::end);
    std::streampos file_size = file.tellg();
    file.seekg(0, std::ios_base::beg);
    bytes.resize(file_size);
    for (auto& i : RANDOM_PRIME_INDEX) {
      int s = RANDOM_PRIME_NUMS[i];
      int e = (s == last_bias) ? int(file_size) : RANDOM_PRIME_NUMS[i + 1];
      file.read(&bytes[0] + s, e - s);
    }
    file.close();
  } else {
    return -1;
  }

  return 0;
}

}  // namespace cipher

#endif  //_AES_HPP_
