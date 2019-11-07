#pragma once

#include "util.hpp"
#include "../external/tongrams/include/utils/util_types.hpp"

namespace tongrams {
namespace constants {

namespace file_extension {
static const std::string counts("c");
static const std::string merged("m");
}  // namespace file_extension

static const std::string default_tmp_dirname(".");
static const std::string default_output_filename("index");

static const std::string empty("</>");

static const byte_range empty_byte_range{
    reinterpret_cast<uint8_t const*>(empty.c_str()),
    reinterpret_cast<uint8_t const*>(empty.c_str()) + empty.size()};

}  // namespace constants
}  // namespace tongrams
