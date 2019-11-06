#pragma once

#include <fstream>
#include <numeric>

#include "ngrams_block.hpp"
#include "../external/tongrams/include/utils/util_types.hpp"

namespace tongrams {
namespace stream {

namespace constants {
static const int invalid_id = -1;
}

// NOTE: this class is useless, we can do everything with ngrams_block
struct ngrams_block_partition : ngrams_block {
    ngrams_block_partition(uint8_t ngram_order) {
        init(ngram_order);
    }

    typedef count_type value;
    typedef ngram_pointer pointer;

    void init(uint8_t ngram_order) {
        ngrams_block::init(ngram_order);
        range = {0, 0};
        prd = false;
    }

    ngrams_block_partition(ngrams_block_partition&& rhs) {
        *this = std::move(rhs);
    }

    inline ngrams_block_partition& operator=(ngrams_block_partition&& rhs) {
        if (this != &rhs) swap(rhs);
        return *this;
    };

    ngrams_block_partition(ngrams_block_partition const&) : ngrams_block() {
        assert(false);
    }

    ngrams_block_partition& operator=(ngrams_block_partition const&) {
        assert(false);
        return *this;
    };

    uint64_t size() const {
        assert(range.end >= range.begin);
        return range.end - range.begin;
    }

    void swap(ngrams_block_partition& other) {
        ngrams_block::swap(other);
        std::swap(range, other.range);
        std::swap(prd, other.prd);
    }

    void write_memory(std::ofstream& os) {
        assert(!m_memory.empty());
        size_t offset = range.begin * record_size();
        std::streamsize num_bytes = size() * record_size();
        os.write(reinterpret_cast<char const*>(m_memory.data() + offset),
                 num_bytes);
    }

    char* initialize_memory(size_t num_bytes) {
        m_memory.resize(num_bytes);
        return reinterpret_cast<char*>(m_memory.data());
    }

    char* read_bytes(std::ifstream& is, char* dest, size_t num_bytes) {
        is.read(dest, static_cast<std::streamsize>(num_bytes));
        dest += num_bytes;
        return dest;
    }

    void materialize_index() {
        m_index.clear();
        uint64_t num_ngrams = range.end - range.begin;
        m_index.reserve(num_ngrams);
        assert(m_memory.size() > 0);
        for (uint64_t i = range.begin; i != range.end; ++i) {
            auto ptr = m_allocator.allocate(m_memory, i);
            push_back(ptr);
        }
        assert(size() == num_ngrams);
    }

    pointer_range range;
    bool prd;  // processed
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

    floats_vec(size_t n) : m_floats(n) {}

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
