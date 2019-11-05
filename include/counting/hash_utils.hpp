#pragma once

#include "constants.hpp"

namespace tongrams {
namespace hash_utils {

/*
    This code is an adaptation from
    https://github.com/aappleby/smhasher/blob/master/src/MurmurHash2.cpp
        by Austin Appleby
*/
uint32_t murmur_hash32(const void* key, int len, uint32_t seed) {
    // 'm' and 'r' are mixing constants generated offline.
    // They're not really 'magic', they just happen to work well.

    const uint32_t m = 0x5bd1e995;
    const int r = 24;

    // Initialize the hash to a 'random' value

    uint32_t h = seed ^ len;

    // Mix 4 bytes at a time into the hash

    const unsigned char* data = (const unsigned char*)key;

    while (len >= 4) {
        uint32_t k = *(uint32_t*)data;

        k *= m;
        k ^= k >> r;
        k *= m;

        h *= m;
        h ^= k;

        data += 4;
        len -= 4;
    }

    // Handle the last few bytes of the input array

    switch (len) {
        // fall through
        case 3:
            h ^= data[2] << 16;
        // fall through
        case 2:
            h ^= data[1] << 8;
        // fall through
        case 1:
            h ^= data[0];
            h *= m;
    };

    // Do a few final mixes of the hash to ensure the last few
    // bytes are well-incorporated.

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h;
}

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

uint32_t hash_bytes32(byte_range const& br) {
    return murmur_hash32(br.first, br.second - br.first, 0);
}

uint64_t hash_bytes64(byte_range const& br) {
    return murmur_hash64(br.first, br.second - br.first, 0);
}

static const uint64_t hash_empty = hash_bytes64(constants::empty_byte_range);
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
