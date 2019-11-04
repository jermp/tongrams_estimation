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
struct ngrams_block_partition : ngrams_block<count_type> {
    ngrams_block_partition(uint8_t ngram_order, uint64_t num_values) {
        init(ngram_order, num_values);
    }

    typedef count_type value;
    typedef ngram_pointer<count_type> pointer;
    typedef ngrams_block<count_type>::iterator ngrams_iterator;

    void init(uint8_t ngram_order, uint64_t num_values) {
        ngrams_block<count_type>::init(ngram_order, num_values);
        range = {0, 0};
        bid = constants::invalid_id;
        sid = constants::invalid_id;
        prd = false;
        eos = false;
    }

    ngrams_block_partition(ngrams_block_partition&& rhs) {
        *this = std::move(rhs);
    }

    inline ngrams_block_partition& operator=(ngrams_block_partition&& rhs) {
        if (this != &rhs) {
            swap(rhs);
        }
        return *this;
    };

    ngrams_block_partition(ngrams_block_partition const&)
        : ngrams_block<count_type>() {
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
        ngrams_block<count_type>::swap(other);
        std::swap(range, other.range);
        std::swap(bid, other.bid);
        std::swap(sid, other.sid);
        std::swap(prd, other.prd);
        std::swap(eos, other.eos);
    }

    void write_memory(std::ofstream& os) {
        assert(not m_memory.empty());
        size_t offset = range.begin * m_record_size;
        std::streamsize num_bytes = size() * m_record_size;
        os.write(reinterpret_cast<char const*>(m_memory.data() + offset),
                 num_bytes);
    }

    // write current partition
    void write_index(std::ofstream& os) {
        assert(std::distance(begin(), end()));
        // std::cout << "flushing index[" << range.begin << ", " << range.end <<
        // ")" << std::endl; std::cout << "write_index: m_record_size = " <<
        // m_record_size << std::endl;
        for (auto it = begin(); it != end(); ++it) {
            // it->print(5,0);
            os.write(reinterpret_cast<char const*>(it->data),
                     (std::streamsize)(m_record_size));
        }
    }

    char* initialize_memory(size_t num_bytes) {
        // std::cout << "resizing to " << num_bytes << std::endl;
        m_memory.resize(num_bytes);  // .resize(num_bytes, 0)
        return reinterpret_cast<char*>(m_memory.data());
    }

    // TODO: pass a ngram_pointer here
    char* copy(word_id* data, size_t num_bytes, char* dest) {
        auto addr = reinterpret_cast<char*>(data);
        std::copy(addr, addr + num_bytes, dest);
        return dest + num_bytes;
    }

    char* read_bytes(std::ifstream& is, char* dest, size_t num_bytes) {
        is.read(dest, (std::streamsize)(num_bytes));
        dest += num_bytes;
        return dest;
    }

    void materialize_index(uint64_t num_values) {
        m_index.clear();
        uint64_t num_ngrams = range.end - range.begin;
        // std::cout << "materializing index[" << range.begin << ", " <<
        // range.end << ") with: " << num_ngrams << std::endl;
        m_index.reserve(num_ngrams);
        // uint64_t sum = 0;
        // std::cout << "m_allocator.allocated() = " << m_allocator.allocated()
        // << std::endl;
        assert(m_memory.size());

        // m_allocator.print_stats();
        // m_allocator.init(order());
        // m_allocator.print_stats();

        for (uint64_t i = range.begin; i != range.end; ++i) {
            auto ptr = m_allocator.allocate(m_memory, num_values, i);
            // sum += (ptr.value(order(), num_values - 1))->value;
            push_back(ptr);
        }
        assert(size() == num_ngrams);
    }

    // uint64_t sum_bytes() const {
    //     return std::accumulate(m_memory.begin(), m_memory.end(),
    //     uint64_t(0));
    // }

    // uint64_t sum_values(uint64_t num_values = 1) const {
    //     uint64_t sum = 0;
    //     std::for_each(m_index.begin(), m_index.end(),
    //         [&](ngram_pointer<count_type> const& ptr) {
    //             sum += (ptr.value(order(), num_values - 1))->value;
    //         }
    //     );
    //     return sum;
    // }

    void print_info() const {
        std::cout << "bid = " << bid << "; sid = " << sid
                  << "; prd = " << int(prd) << "\n";
        std::cout << "[" << range.begin << ", " << range.end << ")"
                  << std::endl;
    }

    void advance(uint64_t n) {
        // std::cout << "advancing " << range.begin << "/" << range.end << " by
        // " << n << std::endl;
        range.begin += n;
        assert(range.begin <= range.end);
    }

    pointer_range range;  // represent a partition of the block index
    int bid;              // BLOCK ID
    int sid;              // STREAM ID
    bool prd;  // PROCESSED: must become true when all partitions of the indexed
               // block have been processed
    bool eos;  // END OF STREAM: indicated that the partition belongs to the
               // last block of the stream
};

struct writer {
    writer(uint8_t N)
        : m_record_size(ngrams_block<count_type>::record_size(N, 1)) {}

    template <typename Iterator>
    void write_block(std::ofstream& os, Iterator begin, Iterator end, size_t,
                     ngrams_block_statistics const&) {
        for (auto it = begin; it != end; ++it) {
            auto ptr = *it;
            os.write(reinterpret_cast<char const*>(ptr.data), m_record_size);
        }
    }

private:
    std::streamsize m_record_size;
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
