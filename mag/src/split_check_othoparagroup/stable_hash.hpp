#pragma once
#include <cstdint>
#include <string>

/** FNV-1a 32-bit — matches src.preprocessing.stable_hash / build_hash_table.py */
inline uint32_t stable_hash(const std::string& s) {
  uint32_t h = 2166136261u;
  for (unsigned char ch : s) {
    h ^= ch;
    h *= 16777619u;
  }
  return h;
}
