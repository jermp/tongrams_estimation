#pragma once

#include <fstream>
#include <numeric>

#include "util.hpp"
#include "ngrams_block.hpp"
#include "front_coding.hpp"

#include "../external/tongrams/include/utils/util_types.hpp"

namespace tongrams {
namespace stream {

typedef ngrams_block uncompressed_block_type;
typedef fc::ngrams_block<context_order_comparator_type> compressed_block_type;

template <typename Block>
struct async_ngrams_file_source {
    async_ngrams_file_source() : m_file_size(0), m_handle_ptr(nullptr) {}

    async_ngrams_file_source(std::string const& filename)
        : m_file_size(0), m_handle_ptr(nullptr) {
        open(filename);
    }

    void open(std::string const& filename) {
        m_filename = filename;
        m_is.open(filename.c_str(), std::ifstream::binary);
        if (not m_is.good()) {
            throw std::runtime_error(
                "Error in opening binary file, it may not exist or be "
                "malformed.");
        }
        m_is.seekg(0, m_is.end);
        m_file_size = static_cast<size_t>(m_is.tellg());
        m_is.seekg(0, m_is.beg);
    }

    void close() {
        util::wait(m_handle_ptr);
        if (m_is.is_open()) m_is.close();
    }

    void close_and_remove() {
        close();
        std::remove(m_filename.c_str());
    }

    size_t size() const {
        return m_buffer.size();
    }

    bool empty() const {
        return m_buffer.empty();
    }

    Block* get_block() {
        if (empty()) util::wait(m_handle_ptr);
        assert(size());
        return &m_buffer.front();
    }

    void release_block() {
        m_buffer.front().release();
        m_buffer.pop_front();
    }

protected:
    std::string m_filename;
    std::ifstream m_is;
    size_t m_file_size;
    std::deque<Block> m_buffer;
    std::unique_ptr<std::thread> m_handle_ptr;
};

struct uncompressed_stream_generator
    : async_ngrams_file_source<uncompressed_block_type> {
    typedef uncompressed_block_type block_type;

    uncompressed_stream_generator() {}

    uncompressed_stream_generator(uint8_t ngram_order)
        : m_read_bytes(0), m_N(ngram_order), m_eos(false), m_I_time(0.0) {}

    void open(std::string const& filename) {
        async_ngrams_file_source::open(filename);
    }

    void async_fetch_next_block(size_t num_bytes) {
        util::wait(m_handle_ptr);
        m_handle_ptr =
            util::async_call(uncompressed_stream_generator::fetch, num_bytes);
    }

    void fetch_next_block(size_t num_bytes) {
        fetch(num_bytes);
    }

    double I_time() const {
        return m_I_time;
    }

    bool eos() const {
        return m_eos;
    }

private:
    size_t m_read_bytes;
    uint8_t m_N;
    bool m_eos;
    double m_I_time;

    std::function<void(size_t)> fetch = [&](size_t bytes) {
        if (eos()) return;
        auto s = clock_type::now();
        block_type block(m_N);
        if (m_read_bytes + bytes >= m_file_size) {
            bytes = m_file_size - m_read_bytes;
            m_eos = true;
        }
        m_read_bytes += bytes;
        assert(bytes % block.record_size() == 0);
        uint64_t num_ngrams = bytes / block.record_size();
        char* begin = block.initialize_memory(bytes);
        block.read_bytes(m_is, begin, bytes);
        block.materialize_index(num_ngrams);
        m_buffer.push_back(std::move(block));
        auto e = clock_type::now();
        std::chrono::duration<double> elapsed = e - s;
        m_I_time += elapsed.count();
    };
};

struct compressed_stream_generator
    : async_ngrams_file_source<compressed_block_type> {
    typedef compressed_block_type block_type;

    compressed_stream_generator() {}

    compressed_stream_generator(uint8_t ngram_order)
        : m_read_bytes(0)
        , m_N(ngram_order)
        , m_w(0)
        , m_v(0)
        , m_eos(false)
        , m_I_time(0.0) {}

    void open(std::string const& filename) {
        async_ngrams_file_source::open(filename);
        essentials::load_pod(m_is, m_w);
        essentials::load_pod(m_is, m_v);
        m_read_bytes = sizeof(m_w) + sizeof(m_v);
    }

    void async_fetch_next_block(size_t /*num_bytes*/) {
        util::wait(m_handle_ptr);
        m_handle_ptr = util::async_call(compressed_stream_generator::fetch);
    }

    void fetch_next_block(size_t /*num_bytes*/) {
        fetch();
    }

    double I_time() const {
        return m_I_time;
    }

    bool eos() const {
        return m_eos;
    }

private:
    size_t m_read_bytes;
    uint8_t m_N;
    uint8_t m_w;
    uint8_t m_v;
    bool m_eos;
    double m_I_time;

    std::function<void(void)> fetch = [&]() {
        if (eos()) return;
        auto s = clock_type::now();
        size_t size = 0;
        essentials::load_pod(m_is, size);
        m_read_bytes += sizeof(size);
        assert(size > 0);
        block_type block(m_N, size, m_w, m_v);
        size_t bytes = fc::BLOCK_BYTES;
        if (m_read_bytes + bytes >= m_file_size) {
            bytes = m_file_size - m_read_bytes;
            m_eos = true;
        }
        m_read_bytes += bytes;
        block.read(m_is, bytes);
        m_buffer.push_back(std::move(block));
        auto e = clock_type::now();
        std::chrono::duration<double> elapsed = e - s;
        m_I_time += elapsed.count();
    };
};

struct writer {
    writer(uint8_t order) : m_order(order) {}

    template <typename Iterator>
    void write_block(std::ofstream& os, Iterator begin, Iterator end, size_t,
                     ngrams_block_statistics const&) {
        std::streamsize record_size = ngrams_block::record_size(m_order);
        for (auto it = begin; it != end; ++it) {
            auto ptr = *it;
            os.write(reinterpret_cast<char const*>(ptr.data), record_size);
        }
    }

private:
    uint8_t m_order;
};

template <typename T = uint16_t>
struct floats_vec {
    typedef T value_type;
    typedef typename std::vector<T>::iterator iterator;

    floats_vec(size_t n) : m_floats(n) {
        m_reint.uint_value = 0;
    }

    void clear() {
        m_floats.clear();
    }

    void reserve(size_t n) {
        m_floats.reserve(n);
    }

    void resize(size_t n) {
        m_floats.resize(n);
    }

    void push_back(float x) {
        m_reint.uint_value = 0;
        m_reint.float_value = x;
        m_floats.push_back(m_reint.uint_value);
    }

    inline float operator[](size_t i) {
        m_reint.uint_value = m_floats[i];
        return m_reint.float_value;
    }

    size_t size() const {
        return m_floats.size();
    }

    auto* data() {
        return m_floats.data();
    }

    void swap(floats_vec<T>& other) {
        m_floats.swap(other.m_floats);
        std::swap(m_reint, other.m_reint);
    }

    iterator begin() {
        return m_floats.begin();
    }

    iterator end() {
        return m_floats.end();
    }

private:
    bits::reinterpret<T> m_reint;
    std::vector<T> m_floats;
};

}  // namespace stream
}  // namespace tongrams
