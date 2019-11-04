#pragma once

#ifndef __APPLE__
#include <parallel/algorithm>
#endif

#include "util.hpp"
#include "hash_utils.hpp"
#include "ngrams_block.hpp"
#include "parallel_radix_sort.hpp"

namespace tongrams {

template <typename Value, typename Prober = hash_utils::linear_prober,
          typename EqualPred = equal_to>
struct ngrams_hash_block {
    ngrams_hash_block() : m_size(0), m_num_bytes(0) {
        resize(0);
    }

    void init(uint8_t ngram_order, uint64_t size, Value default_value,
              Prober const& prober = Prober(),
              EqualPred const& equal_to = EqualPred()) {
        m_prober = prober;
        m_equal_to = equal_to;
        m_default_value = default_value;
        m_num_bytes = ngram_order * sizeof(word_id);
        m_block.init(ngram_order, 1);
        resize(size);
    }

    void resize(uint64_t size) {
        uint64_t buckets = size * hash_utils::probing_space_multiplier;
        m_data.resize(buckets, ngram_id(-1));
        m_block.resize_memory(size, 1);

#ifdef LSD_RADIX_SORT
        m_block.resize_index(size);
#else
        m_block.resize_index(0);
#endif
    }

    bool find_or_insert(ngram const& key, iterator hint, ngram_id& at) {
        m_prober.init(hint, buckets());
        iterator start = *m_prober;
        iterator it = start;

        while (valid(m_data[it])) {
            assert(it < buckets());
            if (m_equal_to(this->operator[](m_data[it]).data, &(key.data[0]),
                           m_num_bytes)) {
                at = m_data[it];
                return true;
            }
            ++m_prober;
            it = *m_prober;
            if (it == start) {  // back to starting point:
                                // thus all positions have been checked
                std::cerr << "ERROR: all positions have been checked"
                          << std::endl;
                at = ngram_id(-1);
                return false;
            }
        }

        m_data[it] = m_size;
        at = m_size;
        m_block.set(m_size, key.data.begin(), key.data.end(), m_default_value);
        ++m_size;
        return false;
    }

    template <typename Comparator>
    void sort(Comparator const& cmp) {
        std::cerr << "block size = " << m_size << std::endl;
#ifdef LSD_RADIX_SORT
        auto begin = m_block.begin();
        auto end = begin + size();
        uint32_t max_digit = statistics().max_word_id;
        uint32_t num_digits = m_block.order();
        // std::cerr << "max_digit = " << max_digit
        //           << "; num_digits = " << num_digits << std::endl;
        parallel_lsd_radix_sorter<typename ngrams_block<Value>::iterator>
            sorter(max_digit, num_digits);
        sorter.sort(begin, end);
        assert(m_block.template is_sorted<Comparator>(begin, end));
#else
        m_index.resize(size());
        for (size_t i = 0; i != size(); ++i) {
            m_index[i] = i;
        }

#ifdef __APPLE__
        std::sort
#else
        __gnu_parallel::sort
#endif
            (m_index.begin(), m_index.end(), [&](auto const& i, auto const& j) {
                return cmp(m_block.access(i), m_block.access(j));
            });
#endif
    }

    void write_index(std::ofstream& os) {
#ifdef LSD_RADIX_SORT
        auto begin = m_block.begin();
        auto end = begin + size();
        for (auto it = begin; it != end; ++it) {
            os.write(reinterpret_cast<char const*>(it->data),
                     (std::streamsize)(m_block.record_size()));
        }
#else
        auto begin = m_index.begin();
        auto end = begin + size();
        std::for_each(begin, end, [&](ngram_id const& id) {
            os.write(reinterpret_cast<char const*>(m_block.access(id).data),
                     (std::streamsize)(m_block.record_size()));
        });
#endif
    }

    inline auto operator[](ngram_id at) {
#ifdef LSD_RADIX_SORT
        return m_block[at];
#else
        return m_block.access(at);
#endif
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

    inline bool valid(ngram_id id) {
        return id != ngram_id(-1);
    }

    auto& block() {
        return m_block;
    }

    auto begin() {
        return enumerator(*this);
    }

    auto end() {
        return enumerator(*this, m_size);
    }

    auto& index() {
        return m_index;
    }

    struct enumerator {
        enumerator(ngrams_hash_block<Value, Prober, EqualPred>& block,
                   size_t pos = 0)
            : m_pos(pos), m_block(block), m_index(block.index()) {}

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
#ifdef LSD_RADIX_SORT
            return m_block[m_pos];
#else
            return m_block[m_index[m_pos]];
#endif
        }

    private:
        size_t m_pos;
        ngrams_hash_block<Value, Prober, EqualPred>& m_block;
        std::vector<ngram_id>& m_index;
    };

    void swap(ngrams_hash_block<Value, Prober, EqualPred>& other) {
        std::swap(m_size, other.m_size);
        std::swap(m_num_bytes, other.m_num_bytes);
        m_data.swap(other.m_data);
        m_block.swap(other.m_block);
        m_index.swap(other.m_index);
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

    Prober m_prober;
    EqualPred m_equal_to;
    Value m_default_value;

    std::vector<ngram_id> m_data;
    ngrams_block<Value> m_block;
    std::vector<ngram_id> m_index;
};

}  // namespace tongrams
