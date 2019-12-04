#pragma once

#include "util.hpp"
#include "util_types.hpp"
#include "../external/tongrams/include/utils/util_types.hpp"

namespace tongrams {
namespace constants {

static const uint64_t invalid_hash = 0;

namespace file_extension {
static const std::string counts("c");
static const std::string merged("m");
}  // namespace file_extension

static const std::string default_tmp_dirname("./tmp_dir");
static const std::string default_output_filename("out.bin");

static const std::string empty_token("</>");
static const word_id empty_token_word_id = 0;

static const byte_range empty_token_byte_range{
    reinterpret_cast<uint8_t const*>(empty_token.c_str()),
    reinterpret_cast<uint8_t const*>(empty_token.c_str()) + empty_token.size()};

}  // namespace constants
}  // namespace tongrams
