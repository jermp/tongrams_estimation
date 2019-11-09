#pragma once

#include "suffix_trie.hpp"

namespace tongrams {

template <typename Vocabulary, typename Values, typename Grams,
          typename Pointers>
struct suffix_trie<Vocabulary, Values, Grams, Pointers>::estimation_builder {
    estimation_builder() {}

    estimation_builder(uint8_t order, configuration const& config,
                       statistics& stats)
        : m_order(order)
        , m_unk_prob(stats.unk_prob())
        , m_levels(order)
        , m_next_positions(order, 0) {
        building_util::check_order(m_order);

        uint64_t vocab_size = stats.num_ngrams(1);
        size_t log_vocab_size = util::ceil_log2(vocab_size + 1);
        m_vocab_values.resize(vocab_size,
                              64);  // uni-grams' values are not quantized

        m_levels.front().pointers.resize(
            vocab_size + 1, util::ceil_log2(stats.num_ngrams(2) + 1));
        m_probs.resize(order - 1);
        m_backoffs.resize(order - 2);
        std::vector<float> probs;
        std::vector<float> backoffs;

        size_t probs_levels = uint64_t(1) << config.probs_quantization_bits;
        size_t backoffs_levels = uint64_t(1)
                                 << config.backoffs_quantization_bits;
        float prob_quantum = 1.0 / probs_levels;
        float backoff_quantum = 1.0 / backoffs_levels;

        for (uint8_t order = 2; order <= m_order; ++order) {
            uint64_t n = stats.num_ngrams(order);
            auto& level = m_levels[order - 1];
            level.words_ids.resize(n, log_vocab_size);
            level.probs_backoffs_ranks.resize(
                n, config.probs_quantization_bits +
                       ((order != m_order) ? config.backoffs_quantization_bits
                                           : 0));
            probs.resize(probs_levels, 0.0);
            float prev = 0.0;
            for (auto& x : probs) {
                x += prev + prob_quantum;
                prev = x;
            }

            m_probs.add_sequence(order - 1, config.probs_quantization_bits,
                                 probs);

            if (order != m_order) {
                backoffs.resize(backoffs_levels + 1, 0.0);
                float prev = 0.0;
                auto begin = backoffs.begin() + 1;
                for (; begin != backoffs.end(); ++begin) {
                    auto& x = *begin;
                    x += prev + backoff_quantum;
                    prev = x;
                }
                m_backoffs.add_sequence(
                    order - 1, config.backoffs_quantization_bits, backoffs);
                uint64_t pointer_bits =
                    util::ceil_log2(stats.num_ngrams(order + 1) + 1);
                level.pointers.resize(n + 1, pointer_bits);
            }
        }
    }

    void set_next_word(uint8_t n, word_id id) {
        assert(n >= 2 and n <= m_order);
        m_levels[n - 1].words_ids.push_back(id);
    }

    void set_next_pointer(uint8_t n, uint64_t pointer) {
        assert(n >= 1 and n < m_order);
        m_levels[n - 1].pointers.push_back(pointer);
    }

    void set_next_backoff(uint8_t n, float backoff) {
        assert(n >= 2 and n < m_order);
        uint64_t backoff_rank = m_backoffs.rank(n - 2, backoff, 1);
        uint64_t& next_pos = m_next_positions[n - 1];
        uint64_t prob_backoff_rank =
            m_levels[n - 1].probs_backoffs_ranks[next_pos];
        uint64_t probs_quantization_bits = m_probs.quantization_bits(n - 2);
        assert(probs_quantization_bits);
        prob_backoff_rank |= (backoff_rank << probs_quantization_bits);
        m_levels[n - 1].probs_backoffs_ranks.push_back(prob_backoff_rank);
        ++next_pos;
    }

    void set_backoff(uint8_t n, uint64_t pos, float backoff) {
        assert(n >= 2 and n < m_order);
        uint64_t backoff_rank = m_backoffs.rank(n - 2, backoff, 1);
        uint64_t prob_backoff_rank = m_levels[n - 1].probs_backoffs_ranks[pos];
        uint64_t probs_quantization_bits = m_probs.quantization_bits(n - 2);
        assert(probs_quantization_bits);
        prob_backoff_rank |= (backoff_rank << probs_quantization_bits);
        m_levels[n - 1].probs_backoffs_ranks.set(pos, prob_backoff_rank);
    }

    void set_next_unigram_values(float prob, float backoff) {
        uint64_t packed = 0;
        bits::pack(packed, prob, backoff);
        m_vocab_values.push_back(packed);
    }

    void set_unigram_values(uint64_t pos, float prob, float backoff) {
        uint64_t packed = 0;
        bits::pack(packed, prob, backoff);
        m_vocab_values.set(pos, packed);
    }

    void set_word(uint8_t n, uint64_t pos, word_id id) {
        assert(n >= 2 and n <= m_order);
        m_levels[n - 1].words_ids.set(pos, id);
    }

    void set_pointer(uint8_t n, uint64_t pos, uint64_t pointer) {
        assert(n >= 1 and n < m_order);
        m_levels[n - 1].pointers.set(pos, pointer);
    }

    void set_prob(uint8_t n, uint64_t pos, float prob) {
        assert(n >= 2 and n <= m_order);
        uint64_t prob_backoff_rank = m_levels[n - 1].probs_backoffs_ranks[pos];
        uint64_t prob_rank = m_probs.rank(n - 2, prob, 0);
        prob_backoff_rank |= prob_rank;
        m_levels[n - 1].probs_backoffs_ranks.set(pos, prob_backoff_rank);
    }

    void build(suffix_trie<Vocabulary, Values, Grams, Pointers>& trie,
               configuration const& config) {
        trie.m_order = m_order;
        trie.m_unk_prob = m_unk_prob;

        parallel_executor p(2);
        task_region(*(p.executor), [&](task_region_handle& trh) {
            trh.run([&] {
                essentials::logger("building vocabulary");
                uint64_t vocab_size = m_vocab_values.size();
                vocabulary vocab;
                {
                    size_t num_bytes =
                        sysconf(_SC_PAGESIZE) * sysconf(_SC_PHYS_PAGES);
                    vocabulary::builder vocab_builder(vocab_size, num_bytes);
                    vocab_builder.load(config.vocab_tmp_subdirname +
                                       config.vocab_filename);
                    vocab_builder.build(vocab);
                }

                std::vector<byte_range> bytes;
                bytes.reserve(vocab_size);
                compact_vector::builder vocab_ids(
                    vocab_size, util::ceil_log2(vocab_size + 1));
                for (uint64_t id = 0; id < vocab_size; ++id) {
                    bytes.emplace_back(vocab[id]);
                    vocab_ids.push_back(id);
                }

                trie.m_vocab.build(bytes,
                                   compact_vector(),  // use default hash-keys
                                   compact_vector(vocab_ids),
                                   compact_vector(m_vocab_values),
                                   identity_adaptor());
            });

            trh.run([&] {
                m_probs.build(trie.m_probs_averages);
                m_backoffs.build(trie.m_backoffs_averages);

                trie.m_levels.resize(m_order);

                // #pragma omp parallel for
                for (uint8_t n = 2; n <= m_order; ++n) {
                    if (n == m_order) {
                        // prefix sums pointers for N-grams
                        std::cerr << "prefix summing pointers for "
                                  << int(m_order) << "-grams" << std::endl;
                        auto& pointers = m_levels[n - 2].pointers;
                        uint64_t prev = 0;
                        for (uint64_t pos = 1; pos < pointers.size(); ++pos) {
                            prev += pointers[pos];
                            pointers.set(pos, prev);
                        }
                    }

                    std::cerr << "building " << int(n) << "-level words_ids"
                              << std::endl;
                    std::cerr << "m_levels[" << int(n) - 2
                              << "].pointers.back() = "
                              << m_levels[n - 2].pointers.back() << "; ";
                    std::cerr << "m_levels[" << int(n) - 1
                              << "].words_ids.size() = "
                              << m_levels[n - 1].words_ids.size() << std::endl;
                    assert(m_levels[n - 2].pointers.back() ==
                           m_levels[n - 1].words_ids.size());

                    m_levels[n - 1].build_words_ids(n, trie.m_levels[n - 1],
                                                    m_levels[n - 2].pointers);
                    std::cerr << "DONE" << std::endl;

                    m_levels[n - 1].build_probs_backoffs_ranks(
                        trie.m_levels[n - 1]);
                }

                // #pragma omp parallel for
                for (uint8_t n = 1; n < m_order; ++n) {
                    std::cerr << "building " << int(n) << "-level pointers"
                              << std::endl;
                    m_levels[n - 1].build_pointers(trie.m_levels[n - 1]);
                    std::cerr << "DONE" << std::endl;
                }
            });
        });

        estimation_builder().swap(*this);
    }

    // for bebug
    // void print_stats() const {
    //     int n = 1;
    //     for (auto& l : m_levels) {
    //         std::cerr << "===========\n";
    //         std::cerr << "level-" << n << " statistics:\n";
    //         for (auto x : l.words_ids) {
    //             std::cerr << x << " ";
    //         }
    //         std::cerr << std::endl;
    //         for (auto x : l.probs_backoffs_ranks) {
    //             std::cerr << x << " ";
    //         }
    //         std::cerr << std::endl;
    //         for (auto x : l.pointers) {
    //             std::cerr << x << " ";
    //         }
    //         std::cerr << std::endl;
    //         ++n;
    //     }
    //     std::cerr << std::endl;
    // }

    void swap(estimation_builder& other) {
        std::swap(m_order, other.m_order);
        std::swap(m_unk_prob, other.m_unk_prob);
        m_vocab_values.swap(other.m_vocab_values);
        m_probs.swap(other.m_probs);
        m_backoffs.swap(other.m_backoffs);
        m_levels.swap(other.m_levels);
    }

private:
    uint8_t m_order;
    float m_unk_prob;
    compact_vector::builder m_vocab_values;
    typename Values::builder m_probs;
    typename Values::builder m_backoffs;
    std::vector<typename level_type::builder> m_levels;
    std::vector<uint64_t> m_next_positions;
};

}  // namespace tongrams