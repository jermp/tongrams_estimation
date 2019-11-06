#pragma once

#include "util.hpp"
#include "stream.hpp"
#include "front_coding.hpp"

namespace tongrams {
namespace stream {

typedef stream::ngrams_block_partition uncompressed_block_type;
typedef fc::ngrams_block<context_order_comparator_type> compressed_block_type;

template <typename Block>
struct async_ngrams_file_source {
    async_ngrams_file_source()
        : m_file_size(0), m_buffer_size(0), m_handle_ptr(nullptr) {}

    async_ngrams_file_source(std::string const& filename)
        : m_file_size(0), m_buffer_size(0), m_handle_ptr(nullptr) {
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

    // why a pointer and not a reference???
    Block* get() {
        if (empty()) util::wait(m_handle_ptr);
        assert(size());
        Block* ptr = nullptr;
        size_t processed_blocks = 0;
        for (auto& x : m_buffer) {
            if (not x.prd) {
                if (!ptr) ptr = &x;
            } else {
                ++processed_blocks;
            }
        }
        assert(m_buffer.size() >= processed_blocks);
        m_buffer_size = m_buffer.size() - processed_blocks;
        return ptr;
    }

    // unsused by counting and adjusting
    void release_processed_blocks() {
        while (not m_buffer.empty() and m_buffer.front().prd) {
            m_buffer.front().release();
            m_buffer.pop_front();
        }
        m_buffer_size = m_buffer.size();
    }

    // unsused by counting and adjusting
    void processed(Block* ptr) {
        ptr->prd = true;
        --m_buffer_size;
    }

    size_t size() const {
        return m_buffer_size;  // then just use m_buffer.size()
    }

    bool empty() const {
        return size() == 0;
    }

    Block* get_block() {
        if (empty()) util::wait(m_handle_ptr);
        assert(size());
        return &m_buffer.front();
    }

    void release_block() {
        m_buffer.front().release();
        m_buffer.pop_front();
        --m_buffer_size;  // then just use m_buffer.size()
    }

protected:
    std::string m_filename;
    std::ifstream m_is;
    size_t m_file_size;
    size_t m_buffer_size;
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
        block.range.begin = 0;
        if (m_read_bytes + bytes > m_file_size) {
            bytes = m_file_size - m_read_bytes;
            m_eos = true;
        }
        m_read_bytes += bytes;
        assert(bytes % block.record_size() == 0);
        block.range.end = bytes / block.record_size();
        char* begin = block.initialize_memory(bytes);
        block.read_bytes(m_is, begin, bytes);
        m_buffer.push_back(std::move(block));
        ++m_buffer_size;
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
        if (m_read_bytes + bytes > m_file_size) {
            bytes = m_file_size - m_read_bytes;
            m_eos = true;
        }
        m_read_bytes += bytes;
        block.read(m_is, bytes);
        m_buffer.push_back(std::move(block));
        ++m_buffer_size;
        auto e = clock_type::now();
        std::chrono::duration<double> elapsed = e - s;
        m_I_time += elapsed.count();
    };
};

}  // namespace stream
}  // namespace tongrams
