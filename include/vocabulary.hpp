#pragma once

#include "constants.hpp"
#include "util_types.hpp"
#include "iterators.hpp"

#include "../external/tongrams/include/utils/pools.hpp"

namespace tongrams {

struct vocabulary {
    struct builder {
        builder() {}

        builder(size_t vocab_size, size_t bytes = 0)
            : m_vocab_size(vocab_size) {
            m_unigram_strings.reserve(bytes);
            m_offsets.reserve(vocab_size + 1);
            m_offsets.push_back(0);
        }

        void reserve(size_t bytes) {
            m_unigram_strings.reserve(bytes);
        }

        void push_empty() {
            m_offsets.push_back(m_unigram_strings.bytes());
        }

        void push_back(byte_range br) {
            m_unigram_strings.append(br);
            m_offsets.push_back(m_unigram_strings.bytes());
        }

        void load(std::string const& vocab_filename) {
            lines_iterator it(vocab_filename.c_str());
            for (uint64_t i = 0; i < m_vocab_size; ++i) {
                auto unigram = *it;
                byte_range br(unigram.first,
                              unigram.second - 1);  // discard '\n'
                if (bytes::equal_bytes(br, constants::empty_byte_range)) {
                    push_empty();
                } else {
                    push_back(br);
                }
                ++it;
            }
        }

        void swap(builder& other) {
            std::swap(m_vocab_size, other.m_vocab_size);
            m_unigram_strings.swap(other.m_unigram_strings);
            m_offsets.swap(other.m_offsets);
        }

        void build(vocabulary& vocab) {
            vocab.m_unigram_strings.swap(m_unigram_strings);
            vocab.m_unigram_strings.shrink_to_fit();
            vocab.m_base_addr = vocab.m_unigram_strings.base_addr();
            vocab.m_offsets.swap(m_offsets);
            builder().swap(*this);
        }

        size_t size() const {
            return m_offsets.size() - 1;
        }

    private:
        size_t m_vocab_size;
        strings_pool m_unigram_strings;
        std::vector<size_t> m_offsets;
    };

    vocabulary() {}

    byte_range operator[](word_id id) const {
        assert(id < m_offsets.size() - 1);
        uint64_t begin = m_offsets[id];
        uint64_t end = m_offsets[id + 1];
        if (LIKELY(begin != end)) {
            return m_unigram_strings.get_bytes(m_base_addr, begin, end);
        } else {
            return constants::empty_byte_range;
        }
    }

private:
    uint8_t const* m_base_addr;
    strings_pool m_unigram_strings;
    std::vector<size_t> m_offsets;
};

}  // namespace tongrams
