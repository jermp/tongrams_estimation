#pragma once

#include "../external/tongrams/include/vectors/compact_vector.hpp"
#include "../external/tongrams/include/sequences/pointer_sequence.hpp"
#include "../external/tongrams/include/utils/util.hpp"

namespace tongrams {

template <typename Grams, typename Pointers>
struct trie_level {
    struct builder {
        builder() {}

        template <typename T>
        void build_words_ids(uint8_t order, trie_level& level, T& partition) {
            level.m_words_ids.build(words_ids.begin(), words_ids.size(),
                                    partition, order);
            compact_vector::builder().swap(words_ids);
        }

        void build_probs_backoffs_ranks(trie_level& level) {
            probs_backoffs_ranks.build(level.m_probs_backoffs_ranks);
            compact_vector::builder().swap(probs_backoffs_ranks);
        }

        void build_pointers(trie_level& level) {
            level.m_pointers.build(pointers);
            compact_vector::builder().swap(pointers);
        }

        compact_vector::builder words_ids;
        compact_vector::builder probs_backoffs_ranks;
        compact_vector::builder pointers;
    };

    trie_level() {}

    trie_level(size_t size) : m_size(size) {}

    inline pointer_range range(uint64_t pos) {
#ifndef NDEBUG
        // for better diagnostic of
        // test/check_count_model.cpp
        if (pos == global::not_found) {
            throw std::runtime_error("not found");
        }
#endif
        assert(pos < size());
        return m_pointers[pos];
    }

    inline uint64_t next(pointer_range& r, uint64_t id) {
        uint64_t pos = position(r, id);
        r = range(pos);
        return pos;
    }

    inline uint64_t prob_backoff_rank(uint64_t pos) const {
        assert(pos < size());
        return m_probs_backoffs_ranks.access(pos);
    }

    inline uint64_t position(pointer_range r, uint64_t id) {
        uint64_t pos = 0;
        m_words_ids.find(r, id, &pos);
        return pos;
    }

    auto* words_ids() {
        return &m_words_ids;
    }

    auto const& probs_backoffs_ranks() const {
        return m_probs_backoffs_ranks;
    }

    auto const* ptrs() const {
        return &m_pointers;
    }

    size_t size() const {
        return m_size;
    }

    // void print_stats(uint8_t order,
    //                  pointer_sequence<Pointers> const* ranges) const;

    size_t grams_bytes() const {
        return m_words_ids.bytes();
    }

    size_t probs_backoffs_ranks_bytes() const {
        return m_probs_backoffs_ranks.bytes();
    }

    size_t pointers_bytes() const {
        return m_pointers.bytes();
    }

    void save(std::ostream& os, uint8_t order, int value_t) const {
        essentials::save_pod(os, m_size);

        if (order != 1) {
            m_words_ids.save(os);
        }

        switch (value_t) {
            case value_type::prob_backoff:
                m_probs_backoffs_ranks.save(os);
                break;
            case value_type::none:
                break;
            default:
                assert(false);
        }

        m_pointers.save(os);
    }

    void load(std::istream& is, uint8_t order, int value_t) {
        essentials::load_pod(is, m_size);

        if (order != 1) {
            m_words_ids.load(is);
        }

        switch (value_t) {
            case value_type::prob_backoff:
                m_probs_backoffs_ranks.load(is);
                break;
            case value_type::none:
                break;
            default:
                assert(false);
        }

        m_pointers.load(is);
    }

private:
    size_t m_size;
    Grams m_words_ids;
    compact_vector m_probs_backoffs_ranks;
    pointer_sequence<Pointers> m_pointers;
};
}  // namespace tongrams
