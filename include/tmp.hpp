#pragma once

#include "typedefs.hpp"
#include "ngrams_block.hpp"
#include "vocabulary.hpp"

#include <vector>

namespace tongrams {

    namespace tmp {

        struct statistics {

            static const word_id invalid_word_id = word_id(-1);
            static const range_id invalid_range_id = range_id(-1);

            struct word_statistic {
                range_id id;        // current range id to which the word belongs to
                word_id left;       // last seen word to the left of the word
            };

            statistics(uint8_t order)
                : t(order, counts(5, 0))
                , r(order, counts(5, 0))
                , current_range_id(order, 0)

                // order - 1 because modified counts for N-grams are the raw occurrence counts in text
                , occs(order - 1, std::vector<occurrence>(0, 0)) // num. of distinct words appearing
                                                                 // to the left of the word (modified count)
                , stats(order - 1, std::vector<word_statistic>(0, {invalid_range_id, invalid_word_id}))
            {}

            void release(uint8_t n) {
                assert(n > 0);
                stats[n - 1].resize(0, {invalid_range_id,
                                        invalid_word_id});
            }

            void resize(uint8_t n, size_t vocab_size) {
                assert(n > 0);
                occs[n - 1].resize(vocab_size, 0);
                stats[n - 1].resize(vocab_size, {invalid_range_id,
                                                 invalid_word_id});
            }

            void clear() {
                for (uint8_t n = 0; n < t.size(); ++n) {
                    for (uint64_t k = 0; k < 5; ++k) {
                        r[n][k] = 0;
                        t[n][k] = 0;
                    }
                }
            }

            bool was_not_seen(uint8_t n, word_id right) {
                auto& stat = stats[n - 1][right];
                if (stat.id != current_range_id[n - 1]) { // range changes
                    stat.id  = current_range_id[n - 1];
                    return true;
                }
                return false;
            }

            bool update(uint8_t n, word_id left, word_id right) {
                assert(n > 0 and n < t.size());

                auto& stat = stats[n - 1][right];
                auto& occ = occs[n - 1][right];

                if (n != 1) { // do not reset occurrence for uni-grams
                    if (stat.id != current_range_id[n - 1]) { // range changes

                        // update range id if different from the current one
                        // and reset number of occurrences

                        stat.id = current_range_id[n - 1];
                        occ = 0;
                        stat.left = invalid_word_id;
                    }
                }

                if (stat.left != left) {

                    stat.left = left;
                    ++occ;
                    assert(occ);

                    if (occ == 1) {
                        ++r[n - 1][0];
                    } else if (occ > 1 and occ <= 5) {
                        ++r[n - 1][occ - 1];
                        --r[n - 1][occ - 2];
                    }

                    return true;
                }

                return false;
            }

            void combine(uint8_t n) {
                assert(n > 0);
                ++current_range_id[n - 1];
                for (uint64_t k = 0; k < 5; ++k) {
                    uint64_t& c = r[n - 1][k];
                    t[n - 1][k] += c;
                    c = 0;
                }
            }

            void print_stats() { // debug purposes

                std::cerr << "modified counts for unigrams" << std::endl;
                for (auto x: occs[0]) {
                    std::cerr << x << std::endl;
                }

                for (uint8_t n = 1; n <= t.size(); ++n) {
                    for (uint64_t k = 1; k <= 5; ++k) {
                        // std::cerr << "r_" << int(n) << "(" << k << ") = "
                        //           << r[n - 1][k - 1] << std::endl;
                        std::cerr << "t_" << int(n) << "(" << k << ") = "
                                  << t[n - 1][k - 1] << std::endl;
                    }
                }
            }

            std::vector<counts> t; // number of n-grams, for n = 1,...,4, having modified count equal to 1, 2, 3, 4 and 4+ globally (i.e., all ranges)
            std::vector<counts> r; // number of n-grams, for n = 1,...,4, having modified count equal to 1, 2, 3, 4 and 4+ in a range
            std::vector<range_id> current_range_id; // keep track of current range id to know when we switch to the next range
            std::vector<std::vector<occurrence>> occs;
            std::vector<std::vector<word_statistic>> stats;
        };

        struct data {
            data()
                : vocab_builder(0)
            {
                word_ids.set_empty_key(constants::invalid_hash);
                assert(vocab_builder.size() == 0);
            }

            words_map word_ids; // map from unigrams' hashes to word ids

            vocabulary::builder vocab_builder;

            /*
                Offsets at which we will write the computed probabilities
                in the trie levels: these are equivalent to the counts that
                counting sort would compute: we need them for 1 < n <= N.
            */
            std::vector<std::vector<uint64_t>> probs_offsets;

            /*
                Block partitions' offsets.
                Each block corresponds to a partition of the total N-grams file.
            */
            std::vector<std::vector<uint64_t>> blocks_offsets;
        };
    }
}
