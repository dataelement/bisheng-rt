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
  std::string private_key = "50ba58cd-9ab4-40ca-aa76-7c2fd4179321";
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
  std::string private_key = "50ba58cd-9ab4-40ca-aa76-7c2fd4179321";
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


}  // namespace cipher

#endif  //_AES_HPP_
