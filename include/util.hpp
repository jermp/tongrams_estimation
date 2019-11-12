#pragma once

#include "../external/tongrams/include/utils/iterators.hpp"
#include "../external/tongrams/include/utils/util_types.hpp"

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/filesystem.hpp>

#include <sys/mman.h>  // for POSIX_MADV_SEQUENTIAL and POSIX_MADV_RANDOM
#include <thread>
#include <fstream>

namespace tongrams {
namespace util {

void write(std::ofstream& os, byte_range br) {
    os.write(reinterpret_cast<const char*>(br.first),
             (br.second - br.first) * sizeof(char));
}

size_t file_size(const char* filename) {
    boost::filesystem::path filepath(filename);
    return boost::filesystem::file_size(filepath);
}

bool exists(const char* filename) {
    boost::filesystem::path filepath(filename);
    return boost::filesystem::exists(filepath);
}

template <typename File>
void check_file(File const& file) {
    if (not file.is_open()) {
        throw std::runtime_error(
            "Error in opening file: it may not exist or be malformed.");
    }
}

template <typename Address>
void optimize_access(Address addr, size_t len, int MODE) {
    auto ret = posix_madvise((void*)addr, len, MODE);
    if (ret) {
        std::cerr << "Error in calling madvice: " << errno << std::endl;
    }
}

#define optimize_sequential_access(addr, len) \
    optimize_access(addr, len, POSIX_MADV_SEQUENTIAL)
#define optimize_random_access(addr, len) \
    optimize_access(addr, len, POSIX_MADV_RANDOM)

template <typename File>
uint8_t const* open_file_partition(File& file, std::string const& filename,
                                   size_t partition_size,
                                   size_t offset  // in bytes
) {
    file.open(filename.c_str(), partition_size, offset);
    util::check_file(file);
    assert(file.size() == partition_size);
    return reinterpret_cast<uint8_t const*>(file.data());
}

void clean_temporaries(std::string const& tmp_dirname) {
    boost::filesystem::remove_all(boost::filesystem::path(tmp_dirname.c_str()));
}

template <typename Funct, typename... Args>
auto async_call(Funct& f, Args&&... args) {
    return std::make_unique<std::thread>(f, args...);
}

void wait(std::unique_ptr<std::thread>& handle_ptr) {
    if (handle_ptr and handle_ptr->joinable()) handle_ptr->join();
}

}  // namespace util
}  // namespace tongrams
