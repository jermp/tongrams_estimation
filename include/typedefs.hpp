#pragma once

#include <chrono>
#include <vector>
#include <sparsehash/dense_hash_map>

namespace tongrams {
typedef uint32_t ngram_id;
typedef uint32_t word_id;
typedef uint32_t range_id;
typedef uint32_t occurrence;
typedef uint64_t iterator;
typedef float float_t;
typedef std::vector<uint64_t> counts;
typedef std::vector<float_t> floats;
typedef google::dense_hash_map<uint64_t, word_id> words_map;
typedef std::chrono::high_resolution_clock clock_type;
}  // namespace tongrams
