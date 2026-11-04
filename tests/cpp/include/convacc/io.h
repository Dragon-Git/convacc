// Filesystem helpers shared by the native drivers.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace convacc {

// Reads a whole file into memory. Returns an empty vector when the file cannot
// be opened, which every caller already treats as "this case is unusable".
inline std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  std::vector<uint8_t> buf;
  std::ifstream in(path, std::ios::binary);
  if (!in) return buf;
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  in.seekg(0, std::ios::beg);
  buf.resize(static_cast<size_t>(size));
  in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
  return buf;
}

}  // namespace convacc
