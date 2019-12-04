#pragma once

#include "constants.hpp"

namespace tongrams {
namespace hash_utils {

/*
    This code is an adaptation from
    https://github.com/aappleby/smhasher/blob/master/src/MurmurHash2.cpp
        by Austin Appleby
*/
uint64_t murmur_hash64(const void* key, size_t len, uint64_t seed) {
    const uint64_t m = 0xc6a4a7935bd1e995ULL;
    const int r = 47;

    uint64_t h = seed ^ (len * m);

#if defined(__arm) || defined(__arm__)
    const size_t ksize = sizeof(uint64_t);
    const unsigned char* data = (const unsigned char*)key;
    const unsigned char* end = data + (std::size_t)(len / 8) * ksize;
#else
    const uint64_t* data = (const uint64_t*)key;
    const uint64_t* end = data + (len / 8);
#endif

    while (data != end) {
#if defined(__arm) || defined(__arm__)
        uint64_t k;
        memcpy(&k, data, ksize);
        data += ksize;
#else
        uint64_t k = *data++;
#endif

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    const unsigned char* data2 = (const unsigned char*)data;

    switch (len & 7) {
        // fall through
        case 7:
            h ^= uint64_t(data2[6]) << 48;
        // fall through
        case 6:
            h ^= uint64_t(data2[5]) << 40;
        // fall through
        case 5:
            h ^= uint64_t(data2[4]) << 32;
        // fall through
        case 4:
            h ^= uint64_t(data2[3]) << 24;
        // fall through
        case 3:
            h ^= uint64_t(data2[2]) << 16;
        // fall through
        case 2:
            h ^= uint64_t(data2[1]) << 8;
        // fall through
        case 1:
            h ^= uint64_t(data2[0]);
            h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

static inline uint64_t byte_range_hash64(byte_range const& br) {
    return murmur_hash64(br.first, br.second - br.first, 0);
}

static inline uint64_t hash64(const void* data, size_t bytes) {
    return murmur_hash64(data, bytes, 0);
}

static const uint64_t hash_empty_token =
    byte_range_hash64(constants::empty_token_byte_range);
static constexpr float probing_space_multiplier = 1.5;

struct linear_prober {
    linear_prober(iterator position, uint64_t universe)
        : m_position(position % universe), m_universe(universe) {}

    inline iterator operator*() {
        if (m_position == m_universe) m_position = 0;  // fall back
        return m_position;
    }

    inline void operator++() {
        ++m_position;
    }

private:
    iterator m_position;
    uint64_t m_universe;
};

}  // namespace hash_utils
}  // namespace tongrams
