#pragma once

#include "../external/tongrams/external/emphf/base_hash.hpp"
#include "../external/tongrams/include/utils/mph_tables.hpp"
#include "../external/tongrams/include/sequences/ef_sequence.hpp"
#include "../external/tongrams/include/sequences/sequence_collections.hpp"
#include "../external/tongrams/include/sequences/uniform_pef_sequence.hpp"
#include "../external/tongrams/include/state.hpp"

#include "trie_level.hpp"
#include "configuration.hpp"
#include "statistics.hpp"

/*

    NOTE: this class will evolve in just a builder for
    the already coded trie_prob_lm.hpp.
    But, for the moment being, we use this other implementation.

*/

namespace tongrams {

template <typename Vocabulary, typename Values, typename Grams,
          typename Pointers>
struct suffix_trie {
    typedef suffix_trie<Vocabulary, Values, Grams, Pointers> trie_type;
    typedef trie_level<Grams, Pointers> level_type;
    typedef prob_model_state<uint64_t> state_type;

    struct builder {
        builder() {}

        builder(uint8_t order, configuration const& config, statistics& stats)
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
                    n,
                    config.probs_quantization_bits +
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
                    // std::cerr << "backoffs quantization book" << std::endl;
                    for (; begin != backoffs.end(); ++begin) {
                        auto& x = *begin;
                        x += prev + backoff_quantum;
                        prev = x;
                    }
                    // for (auto x: backoffs) {
                    //     std::cerr << x << " ";
                    // }
                    // std::cerr << std::endl;

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
            // if (m_levels[n - 1].words_ids.set_positions % 64 == 0) {
            //     std::cerr << "set " << m_levels[n -
            //     1].words_ids.set_positions
            //               << " word ids for " << int(n) << "-grams" <<
            //               std::endl;
            // }
        }

        void set_next_pointer(uint8_t n, uint64_t pointer) {
            assert(n >= 1 and n < m_order);
            m_levels[n - 1].pointers.push_back(pointer);
            // if (m_levels[n - 1].pointers.set_positions % 64 == 0) {
            //     std::cerr << "set " << m_levels[n - 1].pointers.set_positions
            //               << " pointers for " << int(n) << "-grams" <<
            //               std::endl;
            // }
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
            // std::cerr << "setting backoff rank for " << int(n) << "-gram" <<
            // std::endl;
            m_levels[n - 1].probs_backoffs_ranks.push_back(prob_backoff_rank);
            ++next_pos;
        }

        void set_backoff(uint8_t n, uint64_t pos, float backoff) {
            assert(n >= 2 and n < m_order);
            uint64_t backoff_rank = m_backoffs.rank(n - 2, backoff, 1);
            uint64_t prob_backoff_rank =
                m_levels[n - 1].probs_backoffs_ranks[pos];
            uint64_t probs_quantization_bits = m_probs.quantization_bits(n - 2);
            assert(probs_quantization_bits);
            prob_backoff_rank |= (backoff_rank << probs_quantization_bits);
            m_levels[n - 1].probs_backoffs_ranks.set(pos, prob_backoff_rank);
        }

        void set_next_unigram_values(float prob, float backoff) {
            uint64_t packed = 0;
            // std::cerr << "prob = " << prob << "; backoff = " << backoff <<
            // std::endl; if (prob == 0.0) {
            //     std::cerr << "prob is 0.0" << std::endl;
            // }
            // if (backoff == 0.0) {
            //     std::cerr << "backoff is 0.0" << std::endl;
            // }
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
            // std::cerr << "set pointer " << pointer << " for " << int(n) <<
            // "-gram at pos " << pos << std::endl;
        }

        void set_prob(uint8_t n, uint64_t pos, float prob) {
            assert(n >= 2 and n <= m_order);
            uint64_t prob_backoff_rank =
                m_levels[n - 1].probs_backoffs_ranks[pos];
            // std::cerr << "calculating ranking for " << int(n) << "-prob " <<
            // prob << std::endl;
            uint64_t prob_rank = m_probs.rank(n - 2, prob, 0);
            // std::cerr << "prob_rank: " << prob_rank << std::endl;
            prob_backoff_rank |= prob_rank;
            // std::cerr << "setting prob rank in position " << pos <<
            // std::endl;
            m_levels[n - 1].probs_backoffs_ranks.set(pos, prob_backoff_rank);
        }

        void build(trie_type& trie, configuration const& config) {
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
                        vocabulary::builder vocab_builder(vocab_size,
                                                          num_bytes);
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

                    trie.m_vocab.build(
                        bytes,
                        compact_vector(),  // use default hash-keys
                        compact_vector(vocab_ids),
                        compact_vector(m_vocab_values), identity_adaptor());
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
                            for (uint64_t pos = 1; pos < pointers.size();
                                 ++pos) {
                                prev += pointers[pos];
                                // std::cerr << prev << " ";
                                pointers.set(pos, prev);
                            }
                            // std::cerr << std::endl;
                        }

                        std::cerr << "building " << int(n) << "-level words_ids"
                                  << std::endl;
                        std::cerr << "m_levels[" << int(n) - 2
                                  << "].pointers.back() = "
                                  << m_levels[n - 2].pointers.back() << "; ";
                        std::cerr << "m_levels[" << int(n) - 1
                                  << "].words_ids.size() = "
                                  << m_levels[n - 1].words_ids.size()
                                  << std::endl;
                        assert(m_levels[n - 2].pointers.back() ==
                               m_levels[n - 1].words_ids.size());

                        // if (n != m_order) // NOTE: do not build last level
                        // for DEBUG
                        m_levels[n - 1].build_words_ids(
                            n, trie.m_levels[n - 1], m_levels[n - 2].pointers);
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

            builder().swap(*this);
        }

        // for bebug
        void print_stats() const {
            int n = 1;
            for (auto& l : m_levels) {
                std::cerr << "===========\n";
                std::cerr << "level-" << n << " statistics:\n";
                for (auto x : l.words_ids) {
                    std::cerr << x << " ";
                }
                std::cerr << std::endl;
                for (auto x : l.probs_backoffs_ranks) {
                    std::cerr << x << " ";
                }
                std::cerr << std::endl;
                for (auto x : l.pointers) {
                    std::cerr << x << " ";
                }
                std::cerr << std::endl;
                ++n;
            }
            std::cerr << std::endl;
        }

        void swap(builder& other) {
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

    suffix_trie() : m_order(0), m_unk_prob(global::default_unk_prob) {}

    state_type state() {
        return state_type(order());
    }

    void score(state_type& state, byte_range const& word, bool& is_OOV,
               float& prob) {
        uint64_pair word_id;
        m_vocab.lookup(word, word_id, identity_adaptor());
        // std::cerr << std::string(word.first, word.second) << ": word_id = "
        // << word_id.first << std::endl;
        state.add_word(word_id.first);

        uint8_t longest_matching_history_len = 0;
        uint64_t order_m1 = 1;

        // STEP (1): determine longest matching history
        if (word_id.first != global::not_found) {
            float backoff;
            bits::unpack(word_id.second, prob, backoff);
            // std::cerr << "1-prob = " << prob << std::endl;
            // std::cerr << "1-backoff = " << backoff << std::endl;

            if (backoff) {
                backoff = std::log10(backoff);
            }
            // else {
            //     // longest_matching_history_len = 1;
            //     backoff = 0.0;
            // }

            state.add_backoff(backoff);

            longest_matching_history_len = 1;

            auto words_rbegin = state.words.rbegin();
            ++words_rbegin;  // skip just added word id

            auto r = m_levels[0].range(word_id.first);
            // std::cerr << "range = (" << r.begin << ", " << r.end << ")" <<
            // std::endl;

            if (state.length == m_order) {  // saturate: at most m_order - 1
                                            // searches can be done in the trie
                --state.length;
            }

            // std::cerr << "searching the trie for a = " << int(state.length) +
            // 1 << "-gram" << std::endl;

            for (; order_m1 <= state.length;
                 ++order_m1, ++words_rbegin, ++longest_matching_history_len) {
                state.advance();

                if (r.end - r.begin == 0) {
                    // no extension to the left, i.e.,
                    // no successors in reversed trie
                    break;
                }

                uint64_t id = *words_rbegin;
                // std::cerr << "next id = " << id << std::endl;

                uint64_t pos = m_levels[order_m1].position(r, id);
                if (pos == global::not_found) {
                    // std::cerr << "NOT FOUND" << std::endl;
                    break;
                }

                uint64_t probs_quantization_bits =
                    m_probs_averages.quantization_bits(order_m1 - 1);
                uint64_t mask = (uint64_t(1) << probs_quantization_bits) - 1;
                uint64_t prob_backoff_rank =
                    m_levels[order_m1].prob_backoff_rank(pos);
                uint64_t prob_rank = prob_backoff_rank & mask;
                uint64_t backoff_rank =
                    prob_backoff_rank >> probs_quantization_bits;
                prob = m_probs_averages.access(order_m1 - 1, prob_rank);
                // std::cerr << (order_m1+1) << "-prob = " << prob << std::endl;

                if (order_m1 != order() - 1) {
                    backoff =
                        m_backoffs_averages.access(order_m1 - 1, backoff_rank);
                    // std::cerr << (order_m1+1) << "-backoff = " << prob <<
                    // std::endl;
                    if (backoff) {
                        backoff = std::log10(backoff);
                        // longest_matching_history_len = order_m1 + 1;
                    }
                    // else {
                    //     backoff = 0.0;
                    // }
                    state.add_backoff(backoff);
                    r = m_levels[order_m1].range(pos);
                    // if (backoff) {
                    //     longest_matching_history_len = order_m1 + 1;
                    // }
                }
            }

            // std::cerr << "longest_matching_history_len = " <<
            // int(longest_matching_history_len) << std::endl;

            prob = std::log10(prob);

        } else {  // unseen word
            ++state.OOVs;
            is_OOV = true;
            prob = m_unk_prob;
            state.add_backoff(0.0);  // for log10 values
            // state.add_backoff(1.0);
        }

        // STEP (2): add backoff weights
        // if we encountered unseen ngrams during STEP (1)
        for (uint64_t i = order_m1 - 1; i < state.length; ++i) {
            // std::cerr << "backoff(" << int(i) << ") = " << state.backoff(i)
            // << std::endl;
            prob += state.backoff(i);  // for log10 values
            // prob *= state.backoff(i);
        }

        assert(longest_matching_history_len <= m_order);
        state.length = longest_matching_history_len;
        state.finalize();
        // std::cerr << "**prob = " << prob << std::endl;
        // assert(prob < 1.0);
        assert(prob < 0.0);  // for log10 values
    }

    inline uint64_t order() const {
        return uint64_t(m_order);
    }

    uint64_t size() const {
        uint64_t size = 0;
        for (auto const& a : m_levels) {
            size += a.size();
        }
        return size;
    }

    void save(std::ostream& os) const {
        essentials::save_pod(os, m_order);
        essentials::save_pod(os, m_unk_prob);
        m_vocab.save(os);
        m_probs_averages.save(os);
        m_backoffs_averages.save(os);
        m_levels.front().save(os, 1, value_type::none);
        for (uint8_t n = 1; n < m_order; ++n) {
            m_levels[n].save(os, n + 1, value_type::prob_backoff);
        }
    }

    void load(std::istream& is) {
        essentials::load_pod(is, m_order);
        // std::cerr << "loaded order = " << int(m_order) << std::endl;
        essentials::load_pod(is, m_unk_prob);
        // std::cerr << "loaded m_unk_prob = " << m_unk_prob << std::endl;
        m_unk_prob = std::log10(m_unk_prob);
        // std::cerr << "log10(m_unk_prob) = " << m_unk_prob << std::endl;
        // std::cerr << "loading vocab" << std::endl;
        m_vocab.load(is);
        // std::cerr << "loading probs avgs" << std::endl;
        m_probs_averages.load(is, m_order - 1);
        // std::cerr << "loading backoffs avgs" << std::endl;
        m_backoffs_averages.load(is, m_order - 2);
        // std::cerr << "resizing levels" << std::endl;
        m_levels.resize(m_order);
        m_levels.front().load(is, 1, value_type::none);
        // std::cerr << "loaded level 1" << std::endl;
        for (uint8_t n = 1; n < m_order; ++n) {
            m_levels[n].load(is, n + 1, value_type::prob_backoff);
            // std::cerr << "loaded level " << int(n + 1) << std::endl;
        }
    }

private:
    uint8_t m_order;
    float m_unk_prob;
    Vocabulary m_vocab;
    Values m_probs_averages;
    Values m_backoffs_averages;
    std::vector<level_type> m_levels;
};
}  // namespace tongrams
