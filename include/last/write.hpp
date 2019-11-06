#pragma once

#include "last.hpp"

namespace tongrams {

float last::unigram_prob(word_id w) {
    uint64_t uni_gram_count = m_tmp_stats.occs[0][w];
    uint64_t uni_gram_denominator = m_stats.num_ngrams(2);
    float u =
        (static_cast<float>(uni_gram_count) - m_stats.D(1, uni_gram_count)) /
        uni_gram_denominator;
    u += m_stats.unk_prob();  // interpolate
    assert(u <= 1.0);
    return u;
}

void last::write(uint8_t n, last::state& s) {  // write ngram
    uint8_t N = m_config.max_order;
    assert(n >= 2 and n <= N);

    auto& l = s.range_lengths[n - 1];
    if (l == 0) return;
    auto it = s.iterators[n - 1];
    auto prev_ptr = *(it - 1);  // always one-past the end

    if (n != 2) {
        auto left = prev_ptr[N - n];
        m_index_builder.set_next_word(n - 1, left);
    }

    if (n != N) {
        ++m_pointers[n - 2];
        auto pointer = m_pointers[n - 2];
        m_index_builder.set_next_pointer(n - 1, pointer);
    }

    float backoff = 0.0;  // backoff numerator
    // D_n(1) * N_n(1) + D_n(2) * N_n(2) + D_n(3) * N_n(>= 3),
    // where: N_n(c) = # n-grams with modified count equal to c
    // N_n(>= 3) = # n-grams with modified count >= 3
    for (uint64_t k = 1; k <= 5; ++k) {
        auto& c = m_tmp_stats.r[n - 1][k - 1];
        backoff += c * m_stats.D(n, k);  // = D(n, 3) for k >= 3
        c = 0;                           // reset current range counts
    }

    auto& offset = s.probs_offsets[n - 1];
    assert(offset < m_probs[n - 2].size());

    if (n != N) {
        ++m_tmp_stats.current_range_id[n - 1];

        uint64_t denominator = 0;
        std::for_each(it - l, it, [&](auto ptr) {
            auto right = ptr[N - 1];
            if (m_tmp_stats.was_not_seen(n, right)) {
                uint64_t count = m_tmp_stats.occs[n - 1][right];
                denominator += count;
            }
        });
        assert(denominator > 0);
        assert(backoff <= denominator);
        backoff /= denominator;

        ++m_tmp_stats.current_range_id[n - 1];

        std::for_each(it - l, it, [&](auto ptr) {
            auto right = ptr[N - 1];
            uint64_t count = m_tmp_stats.occs[n - 1][right];
            assert(count > 0);
            float prob =
                (static_cast<float>(count) - m_stats.D(n, count)) / denominator;
            prob += backoff * m_probs[n - 2][offset];

            if (m_tmp_stats.was_not_seen(n, right)) {
                auto& pos = m_tmp_data.probs_offsets[n - 1][right];
                m_index_builder.set_prob(n, pos, prob);
                if (n == N - 1) {
                    m_index_builder.set_pointer(n, pos + 1, count);
                }
                ++pos;
            }

            assert(prob <= 1.0);
            m_probs[n - 1].push_back(prob);
            ++offset;
        });

        ++m_tmp_stats.current_range_id[n - 1];

    } else {  // N-gram case

        assert(s.N_gram_denominator > 0);
        assert(backoff <= s.N_gram_denominator);
        backoff /= s.N_gram_denominator;

        std::for_each(it - l, it, [&](auto const& ptr) {
            uint64_t count = *(ptr.value(N));
            assert(count > 0);
            float prob = (static_cast<float>(count) - m_stats.D(N, count)) /
                         s.N_gram_denominator;
            prob += backoff * m_probs[N - 2][offset];  // interpolate
            assert(prob <= 1.0);

            auto right = ptr[N - 1];
            auto& pos = m_tmp_data.probs_offsets[N - 1][right];

            m_index_builder.set_prob(N, pos, prob);
            m_index_builder.set_word(N, pos,
                                     ptr[0]);  // for suffix order
            ++pos;
        });

        if (it != s.end) s.N_gram_denominator = *(it->value(N));
    }

    if (n == 2) {
        auto context = prev_ptr[N - 2];
        float u = unigram_prob(context);
        m_index_builder.set_next_unigram_values(u, backoff);
    } else {
        m_index_builder.set_next_backoff(n - 1, backoff);
    }

    if (n != N) s.probs_offsets[n] = 0;  // reset next order's offset

    l = 0;
};

}  // namespace tongrams