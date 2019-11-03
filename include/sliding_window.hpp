#pragma once

#include "../external/tongrams/include/utils/iterators.hpp"
#include "util.hpp"
#include "hash_utils.hpp"
#include "constants.hpp"
#include "ngram.hpp"

#include <cstring>  // for std::memmove

namespace tongrams {

struct sliding_window {
    sliding_window(uint8_t capacity)
        : m_end(2), m_buff(capacity), m_time(0.0) {}

    void init(byte_range const& text, uint64_t pos = 2) {
        m_end = pos;
        m_iterator.init(text);
        m_time = 0.0;
    }

    void fill(word_id id) {
        m_buff.data.assign(m_buff.size(), id);
    }

    void print() const {
        for (auto x : m_buff.data) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }

    struct word {
        void init(uint64_t h, byte_range br) {
            hash = h;
            range = br;
        }

        uint64_t hash;
        byte_range range;
    };

    void advance() {
        std::memmove(  // shift left by one
            &m_buff.data[0], &m_buff.data[1],
            ngram::size_of(m_buff.size() - 1));
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
        m_buff.set_back(id);
    }

    inline auto const& get() {
        return m_buff;
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
        return m_buff.data.back();
    }

    inline word_id back() const {
        return m_buff.data.front();
    }

    double time() const {
        return m_time;
    }

private:
    uint64_t m_end;  // beginning of next word
    word m_last;
    forward_byte_range_iterator m_iterator;
    ngram m_buff;
    double m_time;
};
}  // namespace tongrams
