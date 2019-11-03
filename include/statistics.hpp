#pragma once

#include <numeric>

#include "tmp.hpp"
#include "configuration.hpp"
#include "util_types.hpp"

namespace tongrams {

struct statistics {
    struct builder {
        builder(configuration const& config, tmp::data& tmp_data,
                tmp::statistics& tmp_stats)
            : m_config(config)
            , m_tmp_stats(tmp_stats)
            , m_tmp_data(tmp_data)
            , m_ngram_cache(config.max_order, 1)
            , m_t(config.max_order, counts(4, 0))
            , m_D(config.max_order, floats(4, 0))
            , m_num_ngrams(config.max_order, 0)
            , m_total_num_words(0)
            , m_unk_prob(0.0) {}

        void init(size_t vocab_size) {
            assert(vocab_size);
            m_vocab_size = vocab_size;
            uint8_t N = m_config.max_order;
            for (uint8_t n = 1; n < N; ++n) {
                m_tmp_stats.resize(n, m_vocab_size);
            }
            m_tmp_data.probs_offsets.resize(N, counts(0, 0));
            for (uint8_t n = 2; n <= N; ++n) {
                m_tmp_data.probs_offsets[n - 1].resize(m_vocab_size, 0);
            }
        }

        template <typename Iterator>
        void compute_smoothing_statistics(Iterator begin, size_t len) {
            uint8_t N = m_config.max_order;
            m_num_ngrams[N - 1] += len;

            auto prev_ptr = *begin;
            if (not m_ngram_cache.empty()) {
                prev_ptr = m_ngram_cache.get();
            }

            for (size_t i = 0; i < len; ++i, ++begin) {
                auto ptr = *begin;
                word_id right = ptr[N - 1];
                // ptr.print(5,1);

                for (uint8_t n = 1; n < N; ++n) {
                    // context changes
                    if (n != 1 and not ptr.equal_to(
                                       prev_ptr, N - n,
                                       N - 1))  // N - 1 is excluded from
                                                // comparison (one-past the end)
                    {
                        m_tmp_stats.combine(n);
                        ++m_num_ngrams[n -
                                       2];  // set num. ngrams of previous order
                    }

                    word_id left = ptr[N - n - 1];
                    if (m_tmp_stats.update(n, left, right)) {
                        ++m_tmp_data.probs_offsets[n][right];
                    }
                }

                if (not ptr.equal_to(prev_ptr, 0, N - 1)) {
                    ++m_num_ngrams[3];
                }

                // N-gram case (special): they do not have modified counts,
                // rather their counts are equal to the occurrence in text
                uint64_t count = (ptr.value(N))->value;
                assert(count);
                m_total_num_words += count;
                if (count <= 4) {
                    ++m_tmp_stats.t[N - 1][count - 1];
                }

                prev_ptr = ptr;
            }

            m_ngram_cache.store(prev_ptr);
        }

        void finalize() {
            ++m_num_ngrams[3];
            for (uint8_t n = 2; n < m_config.max_order; ++n) {
                ++m_num_ngrams[n - 2];
                m_tmp_stats.combine(n);
                m_tmp_stats.release(n);
            }

            for (uint8_t n = 2; n <= m_config.max_order; ++n) {
                for (uint64_t k = 1; k <= 4; ++k) {
                    m_t[n - 1][k - 1] = m_tmp_stats.t[n - 1][k - 1];
                }
            }
        }

        void build(statistics& stats) {
            uint8_t N = m_config.max_order;
            stats.num_ngrams(1) = m_vocab_size;
            // std::cerr << "vocabulary size: " << m_vocab_size << std::endl;
            for (uint8_t n = 2; n <= N; ++n) {
                stats.num_ngrams(n) = m_num_ngrams[n - 1];
            }

            std::cerr << "number of ngrams:\n";
            size_t sum = 0;
            for (uint8_t n = 1; n <= N; ++n) {
                std::cerr << int(n) << "-grams: " << stats.num_ngrams(n)
                          << "\n";
                sum += stats.num_ngrams(n);
            }
            std::cerr << "total num. grams: " << sum << std::endl;

            // NOTE: smoothing statistics for unigrams must be computed globally
            for (auto k : m_tmp_stats.occs[0]) {
                assert(k);
                if (k <= 4) {
                    ++m_t[0][k - 1];
                }
            }

            // m_tmp_stats.print_stats();

            for (uint8_t n = 2; n <= N; ++n) {
                auto& positions = m_tmp_data.probs_offsets[n - 1];
                // compute prefix sums
                for (uint64_t id = 0, sum = 0; id < m_vocab_size; ++id) {
                    uint64_t occ = positions[id];
                    positions[id] = sum;
                    sum += occ;
                }
                // for (auto x: positions) {
                //     std::cerr << x << " ";
                // }
                // std::cerr << std::endl;
            }

            // NOTE: do not compute for small synthetic datasets
            for (uint8_t n = 1; n <= N; ++n) {
                for (uint64_t k = 1; k <= 4; ++k) {
                    try {
                        D(n, k) = compute_discount(n, k);
                    } catch (std::runtime_error const& e) {
                        e.what();
                        complain(n, k);
                        util::clean_temporaries(m_config.tmp_dirname);
                        std::abort();
                    }
                }
            }

            for (uint64_t k = 1; k <= 3; ++k) {
                m_unk_prob += t(1, k) * D(1, k);
            }
            m_unk_prob /= stats.num_ngrams(2);  // uni-grams' denominator
            m_unk_prob /= stats.num_ngrams(
                1);  // interpolate with uniform distribution: 1/|vocabulary|

            stats.m_t.swap(m_t);
            stats.m_D.swap(m_D);
            stats.m_total_num_words = m_total_num_words - N + 1;
            stats.m_unk_prob = m_unk_prob;
            std::cerr << "total num. tokens: " << stats.m_total_num_words
                      << std::endl;
        }

    private:
        configuration const& m_config;
        tmp::statistics& m_tmp_stats;
        tmp::data& m_tmp_data;
        ngram_cache<payload> m_ngram_cache;
        std::vector<counts> m_t;
        std::vector<floats> m_D;
        counts m_num_ngrams;
        uint64_t m_total_num_words;  // total numer of words in the text corpus
        size_t m_vocab_size;
        float m_unk_prob;  // prob of <unk> word, which is backoff(empty) /
                           // vocabulary_size

        float& D(uint8_t n, uint64_t k) {
            assert(k);
            assert(n >= 1 and n <= m_config.max_order);
            if (k >= 3) {
                return m_D[n - 1].back();
            }
            return m_D[n - 1][k - 1];
        }

        inline uint64_t t(uint8_t n, uint64_t k) const {
            assert(n >= 1 and n <= m_config.max_order);
            assert(k > 0 and k <= 4);
            return m_t[n - 1][k - 1];
        }

        float compute_discount(uint8_t n, uint64_t k) {
            assert(k and k <= 4);
            assert(n >= 1 and n <= m_config.max_order);

            if (k <= 3) {
                float d = (t(n, 1) + 2 * t(n, 2)) * t(n, k);
                if (d == 0.0) {
                    throw std::runtime_error("bad discount");
                }
                return static_cast<float>(k) -
                       static_cast<float>((k + 1) * t(n, 1) * t(n, k + 1)) / d;
            }

            return compute_discount(n, 3);
        }

        void complain(uint8_t n, uint64_t k) {
            std::cerr << "Error: could not calculate Kneser-Ney discounts for "
                      << int(n) << "-grams with adjusted count " << k
                      << " because it was not observed any " << int(n)
                      << "-grams with adjusted count:\n";
            check(n, 1);
            check(n, 2);
            check(n, 3);
            std::cerr << "Is this small or artificial data?" << std::endl;
        }

        void check(uint8_t n, uint64_t k) {
            if (!t(n, k)) {
                std::cerr << k << "\n";
            }
        }
    };

    statistics(uint8_t order)
        : m_num_ngrams(order, 0)
        , m_t(order, counts(4, 0))
        , m_D(order, floats(4, 0))
        , m_total_num_words(0)
        , m_unk_prob(0.0) {}

    inline float D(uint8_t n, uint64_t k) const {
        assert(k);
        assert(n >= 1 and n <= order());

        if (k >= 3) {
            return m_D[n - 1].back();
        }
        return m_D[n - 1][k - 1];
    }

    inline uint64_t t(uint8_t n, uint64_t k) const {
        assert(n >= 1 and n <= order());
        assert(k > 0 and k <= 4);
        return m_t[n - 1][k - 1];
    }

    inline uint64_t& num_ngrams(uint8_t n) {
        assert(n >= 1 and n <= order());
        return m_num_ngrams[n - 1];
    }

    inline uint64_t total_words() const {
        return m_total_num_words;
    }

    uint64_t total_grams() const {
        return std::accumulate(m_num_ngrams.begin(), m_num_ngrams.end(),
                               uint64_t(0));
    }

    inline float unk_prob() const {
        return m_unk_prob;
    }

    uint64_t order() const {
        return m_num_ngrams.size();
    }

    void print() {
        std::cerr << "number of ngrams:\n";
        for (uint8_t n = 1; n <= order(); ++n) {
            std::cerr << int(n) << "-grams: " << num_ngrams(n) << "\n";
        }

        std::cerr << "smoothing statistics:\n";
        for (uint8_t n = 1; n <= order(); ++n) {
            uint64_t sum = 0;
            for (uint64_t k = 1; k <= 4; ++k) {
                std::cerr << "t_" << int(n) << "(" << k << ") = " << t(n, k)
                          << "\n";
                sum += t(n, k);
            }
            std::cerr << "sum: " << sum << "\n" << std::endl;
        }

        std::cerr << "discounts:\n";
        for (uint8_t n = 1; n <= order(); ++n) {
            for (uint64_t k = 1; k <= 3; ++k) {
                std::cerr << "D_" << int(n) << "(" << k << ") = " << D(n, k)
                          << " ";
            }
            std::cerr << std::endl;
        }
    }

private:
    counts m_num_ngrams;
    std::vector<counts> m_t;
    std::vector<floats> m_D;
    uint64_t m_total_num_words;
    float m_unk_prob;
};
}  // namespace tongrams
