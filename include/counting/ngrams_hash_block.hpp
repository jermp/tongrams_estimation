#pragma once

#ifndef __APPLE__
#include <parallel/algorithm>
#endif

#include "util.hpp"
#include "hash_utils.hpp"
#include "ngrams_block.hpp"
#include "parallel_radix_sort.hpp"

namespace tongrams {

template <typename Value, typename Prober = hash_utils::linear_prober>
struct ngrams_hash_block {
    static constexpr ngram_id invalid_ngram_id = ngram_id(-1);

    ngrams_hash_block() : m_size(0), m_num_bytes(0) {
        resize(0);
    }

    void init(uint8_t ngram_order, uint64_t size) {
        m_num_bytes = ngram_order * sizeof(word_id);
        m_block.init(ngram_order);
        resize(size);
    }

    void resize(uint64_t size) {
        uint64_t buckets = size * hash_utils::probing_space_multiplier;
        m_data.resize(buckets, invalid_ngram_id);
        m_block.resize_memory(size);
        m_block.resize_index(size);
    }

    bool find_or_insert(ngram const& key, iterator hint, ngram_id& at) {
        Prober prober(hint, buckets());
        iterator start = *prober;
        iterator it = start;

        while (m_data[it] != invalid_ngram_id) {
            assert(it < buckets());
            if (equal_to(m_block[m_data[it]].data, key.data.data(),
                         m_num_bytes)) {
                at = m_data[it];
                return true;
            }
            ++prober;
            it = *prober;
            if (it == start) {  // back to starting point:
                                // thus all positions have been checked
                std::cerr << "ERROR: all positions have been checked"
                          << std::endl;
                at = invalid_ngram_id;
                return false;
            }
        }

        // insert
        m_data[it] = m_size++;
        at = m_data[it];
        m_block.set(at, key.data.begin(), key.data.end(), Value(1));
        return false;
    }

    template <typename Comparator>
    void sort(Comparator const& comparator) {
        std::cerr << "block size = " << m_size << std::endl;
        auto begin = m_block.begin();
        auto end = begin + size();

#ifdef LSD_RADIX_SORT
        (void)comparator;
        uint32_t max_digit = statistics().max_word_id;
        uint32_t num_digits = m_block.order();
        // std::cerr << "max_digit = " << max_digit
        //           << "; num_digits = " << num_digits << std::endl;
        parallel_lsd_radix_sorter<typename ngrams_block<Value>::iterator>
            sorter(max_digit, num_digits);
        sorter.sort(begin, end);
#else

#ifdef __APPLE__
        std::sort
#else
        __gnu_parallel::sort
#endif
            (begin, end, [&](auto l, auto r) { return comparator(l, r); });
#endif

        assert(m_block.template is_sorted<Comparator>(begin, end));
    }

    inline typename Value::value_type& operator[](ngram_id at) {
        assert(at < size());
        return m_block.value(at);
    }

    inline uint64_t size() const {
        return m_size;
    }

    inline bool empty() const {
        return size() == 0;
    }

    inline uint64_t buckets() const {
        return m_data.size();
    }

    double load_factor() const {
        return static_cast<double>(size()) / buckets();
    }

    auto begin() {
        return enumerator(m_block);
    }

    auto end() {
        return enumerator(m_block, size());
    }

    struct enumerator {
        enumerator(ngrams_block<Value>& block, size_t pos = 0)
            : m_pos(pos), m_block(block) {}

        bool operator==(enumerator const& rhs) {
            return m_pos == rhs.m_pos;
        }

        bool operator!=(enumerator const& rhs) {
            return not(*this == rhs);
        }

        void operator++() {
            ++m_pos;
        }

        auto operator*() {
            return m_block[m_pos];
        }

    private:
        size_t m_pos;
        ngrams_block<Value>& m_block;
    };

    void swap(ngrams_hash_block<Value, Prober>& other) {
        std::swap(m_size, other.m_size);
        std::swap(m_num_bytes, other.m_num_bytes);
        m_data.swap(other.m_data);
        m_block.swap(other.m_block);
    }

    void release_hash_index() {
        std::vector<ngram_id>().swap(m_data);
    }

    void release() {
        ngrams_hash_block().swap(*this);
    }

    auto& statistics() {
        return m_block.stats;
    }

private:
    uint64_t m_size;
    size_t m_num_bytes;
    std::vector<ngram_id> m_data;
    ngrams_block<Value> m_block;
};

}  // namespace tongrams
