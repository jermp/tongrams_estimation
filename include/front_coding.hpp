#pragma once

#include "util_types.hpp"
#include "../external/tongrams/include/vectors/bit_vector.hpp"

#include <fstream>

namespace tongrams {
namespace fc {

const static std::streamsize BLOCK_BYTES = 64 * essentials::MiB;
const static std::streamsize BLOCK_BITS = BLOCK_BYTES * 8;

template <typename Comparator>
struct writer {
    writer(uint8_t N) : m_comparator(N) {}

    template <typename Iterator>
    void write_block(std::ofstream& os, Iterator begin, Iterator end, size_t n,
                     ngrams_block_statistics const& stats) {
        // in bytes
        std::cerr << "stats.max_word_id = " << stats.max_word_id << std::endl;
        std::cerr << "stats.max_count = " << stats.max_count << std::endl;
        uint8_t l = 1;
        uint8_t w = (util::ceil_log2(stats.max_word_id + 1) + 7) / 8;
        uint8_t v = (util::ceil_log2(stats.max_count + 1) + 7) / 8;
        essentials::save_pod(os, w);
        essentials::save_pod(os, v);

        std::cerr << "w = " << int(w) << "; v = " << int(v) << std::endl;
        std::cerr << "\tsaving " << n << " records" << std::endl;

        // in bits
        l *= 8;
        w *= 8;
        v *= 8;
        uint8_t N = m_comparator.order();
        size_t max_record_size = l + N * w + v;  // in bits

        m_buffer.init();
        m_buffer.reserve(BLOCK_BITS);

        auto explicit_write = [&](ngram_pointer const& ptr) {
            for (int i = 0; i < N; ++i) {
                m_buffer.append_bits(ptr[i], w);
            }
            m_buffer.append_bits(*(ptr.value(N)), v);
        };

        size_t ngrams_in_block = 1;  // since first is written explicitly
        auto save = [&](std::vector<uint64_t> const& bits, size_t bytes) {
            // std::cerr << "saving " << ngrams_in_block << " ngrams in block"
            // << std::endl;
            essentials::save_pod(os, ngrams_in_block);
            os.write(reinterpret_cast<char const*>(bits.data()), bytes);
        };

        auto prev_ptr = *begin;
        explicit_write(prev_ptr);
        ++begin;

        size_t written = 0;
        for (size_t encoded = 0; begin != end;
             ++begin, ++encoded, ++ngrams_in_block) {
            int lcp = 0;
            auto ptr = *begin;

            if (BLOCK_BITS - m_buffer.size() < max_record_size) {
                // flush current m_buffer, inserting padding
                // always flush exactly BLOCK_BYTES bytes
                save(m_buffer.data(), BLOCK_BYTES);
                m_buffer.init();
                m_buffer.reserve(BLOCK_BITS);
                written = encoded;
                ngrams_in_block = 0;
                explicit_write(ptr);

            } else {
                // std::cerr << "prev_ptr: ";
                // prev_ptr.print(5);
                // std::cerr << "ptr: ";
                // ptr.print(5);
                lcp = m_comparator.lcp(ptr, prev_ptr);
                // std::cerr << "lcp = " << lcp << std::endl;
                assert(lcp < N);

                m_buffer.append_bits(lcp, l);

                if (lcp == 0) {
                    explicit_write(ptr);
                } else {
                    int i = m_comparator.begin();
                    m_comparator.advance(i, lcp);
                    for (;; m_comparator.next(i)) {
                        m_buffer.append_bits(ptr[i], w);
                        if (i == m_comparator.end()) break;
                    }
                    m_buffer.append_bits(*(ptr.value(N)), v);
                }
            }

            prev_ptr = ptr;
        }

        // save last block if needed
        if (written != n) {
            std::streamsize bytes = (m_buffer.size() + 7) / 8;
            save(m_buffer.data(), bytes);
        }
    }

private:
    Comparator m_comparator;
    bit_vector_builder m_buffer;
};

struct cache {
    cache() : pos(nullptr), m_begin(nullptr), m_data(0, 0) {}

    cache(uint8_t N) : m_data(ngrams_block::record_size(N), 0) {
        init();
    }

    inline void init() {
        m_begin = m_data.data();
        pos = m_begin;
    }

    inline uint8_t* begin() const {
        return m_begin;
    }

    void store(uint8_t const* src, size_t n) {
        std::memcpy(pos, src, n);
    }

    void swap(cache& other) {
        std::swap(pos, other.pos);
        std::swap(m_begin, other.m_begin);
        m_data.swap(other.m_data);
        // init();
    }

    // void print() const {
    //     for (auto x: m_data) {
    //         std::cerr << int(x) << " ";
    //     }
    //     std::cerr << std::endl;
    // }

    uint8_t* pos;

private:
    uint8_t* m_begin;
    std::vector<uint8_t> m_data;
};

template <typename Comparator>
struct ngrams_block {
    ngrams_block() {}

    struct fc_iterator {
        const static size_t W = sizeof(word_id);

        fc_iterator(uint8_t N, size_t pos, size_t size,
                    ngrams_block<Comparator>& m_block)
            : m_it(m_block.m_memory.data())
            , m_comparator(N)
            , m_back(N)
            , m_pos(pos)
            , m_size(size)
            , m_w(m_block.m_w)
            , m_v(m_block.m_v) {
            // std::cerr << "initializing iterator with:\n";
            // std::cerr << "N = " << int(N) << "\n";
            // std::cerr << "m_pos = " << m_pos << "\n";
            // std::cerr << "m_size = " << m_size << "\n";
            // std::cerr << "m_w = " << int(m_w) << "\n";
            // std::cerr << "m_v = " << int(m_v) << std::endl;
            // std::cerr << "init iterator at pos " << pos << "/" << size <<
            // std::endl;
            if (pos != size) {
                decode_explicit();
                // auto ptr = operator*();
                // ptr.print(5);
            }
        }

        void swap(fc_iterator& other) {
            std::swap(m_it, other.m_it);
            m_comparator.swap(other.m_comparator);
            m_back.swap(other.m_back);
            m_back.init();
            std::swap(m_pos, other.m_pos);
            std::swap(m_size, other.m_size);
            std::swap(m_w, other.m_w);
            std::swap(m_v, other.m_v);
        }

        fc_iterator(fc_iterator&& rhs) {
            // std::cerr << "move constr" << std::endl;
            *this = std::move(rhs);
        }

        inline fc_iterator& operator=(fc_iterator&& rhs) {
            // std::cerr << "move assign" << std::endl;
            if (this != &rhs) {
                swap(rhs);
            }
            return *this;
        };

        fc_iterator(fc_iterator const& rhs) {
            // std::cerr << "copy constr" << std::endl;
            *this = rhs;
        }

        fc_iterator& operator=(fc_iterator const& rhs) {
            // std::cerr << "copy assign" << std::endl;
            if (this != &rhs) {
                m_it = rhs.m_it;
                m_comparator = rhs.m_comparator;
                m_back = rhs.m_back;
                m_back.init();
                m_pos = rhs.m_pos;
                m_size = rhs.m_size;
                m_w = rhs.m_w;
                m_v = rhs.m_v;
            }
            return *this;
        };

        bool operator==(fc_iterator const& rhs) {
            return m_pos == rhs.m_pos;
        }

        bool operator!=(fc_iterator const& rhs) {
            return not(*this == rhs);
        }

        inline auto operator*() const {
            ngram_pointer ptr;
            ptr.data = reinterpret_cast<word_id*>(m_back.begin());
            return ptr;
        }

        void operator++() {
            // std::cerr << "pos " << m_pos << "/" << m_size << std::endl;
            if (m_pos == m_size - 1) {
                ++m_pos;  // one-past the end
                // std::cerr << "scanned all positions: thus now one-past the
                // end: " << m_pos << std::endl;
                return;
            }
            decode();
            ++m_pos;
        }

    private:
        uint8_t* m_it;
        Comparator m_comparator;
        cache m_back;
        size_t m_pos, m_size;
        uint8_t m_w, m_v;

        void decode_value() {
            m_back.store(reinterpret_cast<uint8_t const*>(m_it), m_v);
            m_back.pos += sizeof(count_type);
            m_it += m_v;
        }

        void decode_explicit() {
            uint8_t N = m_comparator.order();
            assert(m_back.pos == m_back.begin());
            for (uint8_t i = 0; i < N; ++i) {
                m_back.store(reinterpret_cast<uint8_t const*>(m_it), m_w);
                m_back.pos += W;
                m_it += m_w;
            }
            decode_value();
        }

        void decode() {
            m_back.init();

            uint8_t lcp = 0;
            std::memcpy(&lcp, m_it, 1);
            // std::cerr << "lcp = " << int(lcp) << std::endl;
            m_it += 1;

            if (lcp == 0) {
                decode_explicit();
                return;
            }

            int i = m_comparator.begin();

            m_comparator.advance(i, lcp);
            m_back.pos = m_back.begin() + i * W;

            uint8_t N = m_comparator.order();
            assert(lcp < N);

            // store into [m_back] the other [N] - [lcp] word_ids
            for (int j = 0; j < N - lcp; ++j) {
                m_back.store(reinterpret_cast<uint8_t const*>(m_it), m_w);
                m_comparator.next(i);
                m_back.pos = m_back.begin() + i * W;
                m_it += m_w;
            }

            m_back.pos = m_back.begin() + N * W;
            decode_value();
        }
    };

    typedef fc_iterator iterator;
    typedef ngram_pointer pointer;

    ngrams_block(uint8_t N, size_t size, uint8_t w, uint8_t v)
        : m_size(size), m_N(N), m_w(w), m_v(v) {}

    void read(std::ifstream& is, size_t bytes) {
        // std::cerr << "reading " << bytes << " bytes from file" << std::endl;
        m_memory.resize(bytes);
        is.read(reinterpret_cast<char*>(m_memory.data()), bytes);
    }

    template <typename C>
    bool is_sorted(iterator begin, iterator end) {
        C comparator(m_N);
        auto it = begin;

        size_t record_bytes = tongrams::ngrams_block::record_size(m_N);
        cache prev(m_N);
        prev.init();
        prev.store(reinterpret_cast<uint8_t const*>((*it).data), record_bytes);
        ngram_pointer prev_ptr;
        prev_ptr.data = reinterpret_cast<word_id*>(prev.begin());

        ++it;
        bool ret = true;
        for (size_t i = 1; it != end; ++i, ++it) {
            auto curr_ptr = *it;
            int cmp = comparator.compare(prev_ptr, curr_ptr);
            if (cmp == 0) {
                std::cerr << "Error at " << i << "/" << size() << ":\n";
                prev_ptr.print(m_N);
                curr_ptr.print(m_N);
                std::cerr << "Repeated ngrams" << std::endl;
            }

            if (cmp > 0) {
                std::cerr << "Error at " << i << "/" << size() << ":\n";
                prev_ptr.print(m_N);
                curr_ptr.print(m_N);
                std::cerr << std::endl;
                ret = false;
            }
            prev.init();
            prev.store(reinterpret_cast<uint8_t const*>(curr_ptr.data),
                       record_bytes);
            prev_ptr.data = reinterpret_cast<word_id*>(prev.begin());
        }
        return ret;
    }

    void materialize_index() {}

    void swap(ngrams_block<Comparator>& other) {
        m_memory.swap(other.m_memory);
        std::swap(m_size, other.m_size);
        std::swap(m_N, other.m_N);
        std::swap(m_w, other.m_w);
        std::swap(m_v, other.m_v);
    }

    void release() {
        fc::ngrams_block<Comparator>().swap(*this);
    }

    friend struct fc_iterator;

    inline auto begin() {
        return fc_iterator(m_N, 0, m_size, *this);
    }

    inline auto end() {
        return fc_iterator(m_N, m_size, m_size, *this);
    }

    size_t size() const {
        return m_size;
    }

private:
    std::vector<uint8_t> m_memory;
    size_t m_size;
    uint8_t m_N;
    uint8_t m_w, m_v;
};

}  // namespace fc
}  // namespace tongrams
