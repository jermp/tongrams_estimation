#pragma once

#include <thread>

#include "constants.hpp"

namespace tongrams {

struct configuration {
    configuration()
        : block_size(0.5 * essentials::MB)
        , text_chunk_size(1 * essentials::GB)
        , max_order(5)
        , num_threads(std::thread::hardware_concurrency())
        , tmp_dirname(constants::default_tmp_dirname)
        , vocab_tmp_subdirname(tmp_dirname + "/vocab")
        , vocab_filename("/vocabulary")
        , output_filename(constants::default_output_filename)
        , compress_blocks(false)
        , probs_quantization_bits(global::default_probs_quantization_bits)
        , backoffs_quantization_bits(
              global::default_backoffs_quantization_bits) {}

    uint64_t RAM;
    uint64_t page_size;
    uint64_t block_size;
    uint64_t text_chunk_size;
    uint64_t max_order;  // <--- TODO: change this name to N
    uint64_t num_threads;
    uint64_t text_size;
    std::string tmp_dirname;
    std::string vocab_tmp_subdirname;
    std::string vocab_filename;
    std::string text_filename;
    std::string output_filename;
    bool compress_blocks;
    uint8_t probs_quantization_bits;
    uint8_t backoffs_quantization_bits;
};

}  // namespace tongrams
