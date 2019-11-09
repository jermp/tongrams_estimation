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

    struct estimation_builder;

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
