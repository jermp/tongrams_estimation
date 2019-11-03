#pragma once

#include "typedefs.hpp"
#include "util.hpp"
#include "../external/tongrams/include/utils/iterators.hpp"

namespace tongrams {

// TODO: delete this class and use ngram_cache instead

namespace constants {
static const uint64_t invalid_hash = 0;
}

// namespace util {
//     // NOTE: adapted from
//     //
//     http://stackoverflow.com/questions/1646807/quick-and-simple-hash-code-combinations
//     static uint64_t mix_hashes(std::vector<word_id> const& hashes) {
//         uint64_t hash = 17;
//         for (auto x: hashes) {
//             hash = hash * 31 + x;
//         }
//         return hash;
//     }
// }

struct ngram {
    ngram() {}

    ngram(uint8_t n) {
        resize(n);
    }

    ngram(byte_range const& br, uint8_t n) {
        resize(n);
        init(br, n);
    }

    void init(byte_range const& br, uint8_t n) {
        forward_byte_range_iterator it;
        it.init(br);
        // ++it;  // advance to get first word
        // for (uint8_t i = 0; i < n; ++i, ++it) {
        //     data[i] = util::toul(*it);
        // }
        for (uint8_t i = 0; i < n; ++i) {
            data[i] = util::toul(it.next());
        }
    }

    template <typename Iterator>
    ngram(Iterator begin, uint8_t n) {
        resize(n);
        fill(begin, n);
    }

    void fill(word_id hash) {
        data.assign(data.size(), hash);
    }

    template <typename Iterator>
    void fill(Iterator begin, uint8_t n) {
        for (uint8_t i = 0; i < n; ++i, ++begin) {
            data[i] = *begin;
        }
    }

    void resize(uint8_t n) {
        data.resize(n, 0);
    }

    // debug purposes
    void print() const {
        for (auto v : data) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    static ngram invalid(uint8_t n) {
        std::vector<word_id> hashes(n, constants::invalid_hash);
        ngram ng(hashes.begin(), n);
        return ng;
    }

    static uint64_t size_of(uint8_t order) {
        return sizeof(word_id) * order;
    }

    inline uint64_t size_of() const {
        return ngram::size_of(size());
    }

    inline word_id operator[](size_t i) const {
        assert(i < size());
        return data[i];
    }

    inline void set_back(word_id hash) {
        data.back() = hash;
    }

    inline size_t size() const {
        return data.size();
    }

    inline word_id front() const {
        return data.front();
    }

    inline word_id back() const {
        return data.back();
    }

    // TODO: use SIMD here??
    bool operator==(ngram const& other) const {
        return memcmp(other.data.data(), this->data.data(), size_of()) == 0;
    }

    // void operator=(ngram const& other) {
    //     memcpy(other.data.data(), this->data.data(), size_of());
    // }

    // 16 bytes of overhead :(, due to:
    // - pointer to storage oh heap
    // - size of std::vector
    std::vector<word_id> data;
};
}  // namespace tongrams
