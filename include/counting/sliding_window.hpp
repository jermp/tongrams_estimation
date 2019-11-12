#pragma once

#include <cstring>  // for std::memmove

#include "../external/tongrams/include/utils/iterators.hpp"

#include "util.hpp"
#include "hash_utils.hpp"
#include "constants.hpp"

namespace tongrams {

struct sliding_window {
    sliding_window(uint8_t capacity)
        : m_end(2), m_buff(capacity), m_time(0.0) {}

    void init(byte_range text, uint64_t pos = 2) {
        m_end = pos;
        m_iterator.init(text);
        m_time = 0.0;
    }

    void fill(word_id id) {
        m_buff.assign(m_buff.size(), id);
    }

    // void print() const {
    //     for (auto x : m_buff.data) {
    //         std::cout << x << " ";
    //     }
    //     std::cout << std::endl;
    // }

    struct word {
        void init(uint64_t h, byte_range br) {
            hash = h;
            range = br;
        }

        uint64_t hash;
        byte_range range;
    };

    void advance() {
        std::memmove(&m_buff[0], &m_buff[1],
                     sizeof_ngram(m_buff.size() - 1));  // shift left by one
        uint64_t hash = hash_utils::hash_empty;
        byte_range range = constants::empty_byte_range;
        size_t range_len = 0;

        while (range_len == 0) {  // skip blank lines
            if (m_iterator.has_next()) {
                auto start = clock_type::now();
                range = m_iterator.next();
                auto end = clock_type::now();
                std::chrono::duration<double> elapsed = end - start;
                m_time += elapsed.count();
                range_len = range.second - range.first;
            } else {
                m_end += 2;
                m_last.init(hash, range);
                return;
            }
        }

        ++range_len;
        hash = hash_utils::hash_bytes64(range);
        m_end += range_len;
        m_last.init(hash, range);
    }

    void eat(word_id id) {
        m_buff.back() = id;
    }

    ngram_type const& get() {
        return m_buff;
    }

    word_id const* data() {
        return m_buff.data();
    }

    // return the position of the end of current window,
    // i.e., by discarding the whitespace and the beginning of the next word
    inline uint64_t end() const {
        return m_end - 2;
    }

    inline auto const& last() const {
        return m_last;
    }

    inline word_id front() const {
        return m_buff.back();
    }

    inline word_id back() const {
        return m_buff.front();
    }

    double time() const {
        return m_time;
    }

private:
    uint64_t m_end;  // beginning of next word
    word m_last;
    forward_byte_range_iterator m_iterator;
    ngram_type m_buff;
    double m_time;
};

}  // namespace tongrams
