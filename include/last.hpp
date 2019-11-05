#pragma once

#include "constants.hpp"
#include "util.hpp"
#include "../external/tongrams/include/utils/util.hpp"

#include "index_types.hpp"
#include "stream2.hpp"

namespace tongrams {

/*
    single-threaded version
*/

struct last {
    typedef stream::ngrams_block_partition IN;
    typedef stream::floats_vec<> float_vector_type;

    last(configuration const& config, tmp::data& tmp_data,
         tmp::statistics& tmp_stats, statistics& stats)
        : m_config(config)
        , m_stream_generator(config.max_order)
        , m_tmp_data(tmp_data)
        , m_stats(stats)
        , m_tmp_stats(tmp_stats)
        , m_record_size(ngrams_block<count_type>::record_size(config.max_order))
        , m_pointers(config.max_order - 1, 0)
        , m_probs(config.max_order, float_vector_type(0))
        , m_index_builder(config.max_order, config, stats)
        , m_current_block_id(0)
        , m_fetched_block_id(0)
        , m_num_blocks(tmp_data.blocks_offsets.size())
        , m_CPU_time(0.0)
        , m_I_time(0.0)
        , m_O_time(0.0) {
        assert(m_num_blocks);
        uint8_t N = m_config.max_order;
        directory tmp_dir(m_config.tmp_dirname);
        for (auto const& path : tmp_dir) {
            if (not is_directory(path) and is_regular_file(path) and
                path.extension() == constants::file_extension::merged) {
                m_stream_generator.open(path.string());
                async_fetch_next_block();
                break;
            }
        }

        auto start = clock_type::now();
        size_t vocab_size = m_stats.num_ngrams(1);
        for (uint8_t n = 2; n < N; ++n) {
            m_tmp_stats.resize(n, vocab_size);
            m_index_builder.set_next_pointer(n - 1, 0);
        }
        m_index_builder.set_next_pointer(N - 1, 0);
        auto end = clock_type::now();
        std::chrono::duration<double> elapsed = end - start;
        m_CPU_time += elapsed.count();
    }

    void print_stats() const {
        std::cout << "\"CPU\":" << m_CPU_time << ", ";
        std::cout << "\"I\":" << m_I_time << ", ";
        std::cout << "\"O\":" << m_O_time << ", ";
    }

    void async_fetch_next_block() {
        if (m_fetched_block_id != m_num_blocks) {
            auto const& offsets = m_tmp_data.blocks_offsets[m_fetched_block_id];

            // std::cout << "offsets:\n";
            // for (auto off: offsets) {
            //     std::cout << off << std::endl;
            // }

            size_t n = offsets.back();
            assert(n);
            // std::cout << "loading " << n * m_record_size << " bytes" <<
            // std::endl;
            m_stream_generator.async_fetch_next_block(n * m_record_size);
            ++m_fetched_block_id;
        }
    }

    void run() {
        auto start = clock_type::now();

        for (; m_current_block_id < m_num_blocks;) {
            auto* block = m_stream_generator.get();
            async_fetch_next_block();

            uint8_t N = block->order();
            block->materialize_index();

            for (uint8_t n = 1; n < N; ++n) {
                m_probs[n - 1].reserve(block->size());
            }

            /*
                n = 1: (empty context) the denominator is equal to the number of
               bi-grams n = N: the denominator is equal to the sum of the raw
               counts of N-grams having the same context 1 < n < N (otherwise):
               the denominator is equal to the sum of the modified counts of all
               n-grams having the same context
            */
            counts range_lengths(N, 0);
            counts probs_offsets(N, 0);
            uint64_t N_gram_denominator = 0;
            uint64_t uni_gram_denominator = m_stats.num_ngrams(2);

            auto begin = block->begin();
            auto end = block->end();

            m_tmp_stats.clear();
            // m_tmp_stats.print_stats();

            std::vector<IN::ngrams_iterator> iterators(N, begin);

            auto write = [&](uint8_t n) {
                assert(n >= 2 and n <= N);

                // std::cerr << "writing " << int(n) << "-gram" << std::endl;

                auto& l = range_lengths[n - 1];
                if (l == 0) {
                    return;
                }

                auto it = iterators[n - 1];
                auto prev_ptr = *(it - 1);  // always one-past the end

                if (n != 2) {
                    word_id left = prev_ptr[N - n];
                    // util::do_not_optimize_away(left);
                    // NOTE: commented to measure time without index building
                    m_index_builder.set_next_word(n - 1, left);
                }

                if (n != N) {
                    ++m_pointers[n - 2];
                    uint64_t pointer = m_pointers[n - 2];
                    // util::do_not_optimize_away(pointer);
                    // NOTE: commented to measure time without index building
                    m_index_builder.set_next_pointer(n - 1, pointer);
                }

                // compute numerator of backoff (and reset current range
                // counts):
                float backoff = 0.0;
                // D_n(1) * N_n(1) + D_n(2) * N_n(2) + D_n(3) * N_n(>= 3),
                // where: N_n(c) = # n-grams with modified count equal to c
                // N_n(>= 3) = # n-grams with modified count >= 3
                uint64_t sum = 0;
                for (uint64_t k = 1; k <= 5; ++k) {
                    auto& c = m_tmp_stats.r[n - 1][k - 1];
                    backoff += c * m_stats.D(n, k);  // = D(n, 3) for k >= 3
                    sum += c;
                    c = 0;
                }

                auto& offset = probs_offsets[n - 1];
                assert(offset < m_probs[n - 2].size());

                if (n != N) {
                    ++m_tmp_stats.current_range_id[n - 1];

                    uint64_t denominator = 0;
                    std::for_each(it - l, it, [&](auto const& ptr) {
                        word_id right = ptr[N - 1];
                        if (m_tmp_stats.was_not_seen(n, right)) {
                            uint64_t count = m_tmp_stats.occs[n - 1][right];
                            denominator += count;
                        }
                    });

                    // std::cerr << "sum = " << sum << std::endl;
                    // std::cerr << "range_len = " << l << std::endl;
                    // std::cerr << "numerator = " << backoff << std::endl;
                    // std::cerr << "denominator = " << denominator <<
                    // std::endl;
                    assert(denominator);
                    assert(backoff <= denominator);

                    backoff /= denominator;
                    // std::cerr << "backoff = " << backoff << std::endl;

                    ++m_tmp_stats.current_range_id[n - 1];

                    std::for_each(it - l, it, [&](auto const& ptr) {
                        word_id right = ptr[N - 1];
                        uint64_t count = m_tmp_stats.occs[n - 1][right];
                        assert(count);
                        float prob =
                            (static_cast<float>(count) - m_stats.D(n, count)) /
                            denominator;
                        // NOTE: do not interpolate for tiny language models
                        prob +=
                            backoff * m_probs[n - 2][offset];  // interpolate
                        // std::cerr << "prob = " << prob << std::endl;

                        if (m_tmp_stats.was_not_seen(n, right)) {
                            auto& pos = m_tmp_data.probs_offsets[n - 1][right];
                            // NOTE: commented to measure time without index
                            // building
                            m_index_builder.set_prob(n, pos, prob);

                            if (n == N - 1) {
                                m_index_builder.set_pointer(n, pos + 1, count);
                            }

                            ++pos;
                        }

                        assert(prob <= 1.0);
                        m_probs[n - 1].push_back(prob);
                        ++offset;
                        // std::cerr << "looping: " << i << "/" << l <<
                        // std::endl;
                    });

                    // std::cerr << std::endl;
                    ++m_tmp_stats.current_range_id[n - 1];

                } else {  // N-gram case

                    assert(N_gram_denominator);
                    assert(backoff <= N_gram_denominator);
                    backoff /= N_gram_denominator;

                    std::for_each(it - l, it, [&](auto const& ptr) {
                        uint64_t count = (ptr.value(N))->value;
                        assert(count);
                        float prob =
                            (static_cast<float>(count) - m_stats.D(N, count)) /
                            N_gram_denominator;
                        prob +=
                            backoff * m_probs[N - 2][offset];  // interpolate
                        assert(prob <= 1.0);

                        word_id right = ptr[N - 1];
                        auto& pos = m_tmp_data.probs_offsets[N - 1][right];

                        // NOTE: commented to measure time without index
                        // building
                        m_index_builder.set_prob(N, pos, prob);
                        m_index_builder.set_word(N, pos,
                                                 ptr[0]);  // for suffix order

                        ++pos;
                    });

                    if (it != end) {
                        N_gram_denominator = (it->value(N))->value;
                    }
                }

                if (n == 2) {
                    word_id context = prev_ptr[N - 2];
                    uint64_t uni_gram_count = m_tmp_stats.occs[0][context];
                    float u = (static_cast<float>(uni_gram_count) -
                               m_stats.D(1, uni_gram_count)) /
                              uni_gram_denominator;
                    // NOTE: do not interpolate for tiny language models
                    u += m_stats.unk_prob();  // interpolate
                    assert(u <= 1.0);
                    // NOTE: commented to measure time without index building
                    m_index_builder.set_next_unigram_values(u, backoff);
                } else {
                    // NOTE: commented to measure time without index building
                    m_index_builder.set_next_backoff(n - 1, backoff);
                }

                if (n != N) {  // reset next order's offset
                    probs_offsets[n] = 0;
                }
                l = 0;
            };

            while (iterators.back() != end) {
                for (uint8_t n = 2; n <= N; ++n) {
                    auto& it = iterators[n - 1];

                    if (it != end) {
                        auto prev_ptr = *it;

                        for (; it != end; ++it) {
                            auto ptr = *it;

                            // std::cout << "scanning " << int(n) << ": ";
                            // ptr.print(5,1);

                            if (ptr.equal_to(prev_ptr, N - n, N - 1)) {
                                ++range_lengths[n - 1];
                                word_id right = ptr[N - 1];

                                if (n == N) {
                                    uint64_t count = (ptr.value(N))->value;
                                    N_gram_denominator += count;
                                    if (count < 5) {
                                        ++m_tmp_stats.r[N - 1][count - 1];
                                    } else {
                                        ++m_tmp_stats.r[N - 1].back();
                                    }

                                } else {
                                    if (n == 2) {
                                        uint64_t uni_gram_count =
                                            m_tmp_stats.occs[0][right];
                                        float u =
                                            (static_cast<float>(
                                                 uni_gram_count) -
                                             m_stats.D(1, uni_gram_count)) /
                                            uni_gram_denominator;
                                        // NOTE: do not interpolate for tiny
                                        // language models
                                        u += m_stats.unk_prob();  // interpolate
                                        assert(u <= 1.0);
                                        m_probs[0].push_back(u);
                                    }

                                    word_id prev_left = prev_ptr[N - n - 1];
                                    word_id left = ptr[N - n - 1];
                                    m_tmp_stats.update(n, left, right);

                                    if (left !=
                                        prev_left) {  // set pointers for
                                                      // previous order
                                        ++m_pointers[n - 2];
                                    }
                                }

                            } else {
                                break;  // it is one-past-the-end
                            }

                            prev_ptr = ptr;
                        }

                        write(n);
                    }
                }
            }

            // write last entries since [begin, end) is aligned
            // according to unigrams' boundaries
            for (uint8_t n = 2; n <= N; ++n) {
                write(n);
            }

            for (auto& p : m_probs) p.clear();

            ++m_current_block_id;
            if (m_current_block_id % 20 == 0) {
                std::cerr << "processed " << m_current_block_id << "/"
                          << m_num_blocks << " blocks" << std::endl;
            }

            m_stream_generator.processed(block);
            m_stream_generator.release_processed_blocks();
        }

        auto end = clock_type::now();
        std::chrono::duration<double> elapsed = end - start;
        m_CPU_time += elapsed.count();

        std::vector<float_vector_type>().swap(m_probs);

        m_stream_generator.close();  // close but do not destroy

        // m_index_builder.print_stats();
        essentials::logger("compressing index");
        start = clock_type::now();
        suffix_trie_index index;
        m_index_builder.build(index, m_config);
        end = clock_type::now();
        elapsed = end - start;
        std::cerr << "compressing index took: " << elapsed.count() << " [sec]"
                  << std::endl;
        m_CPU_time += elapsed.count();

        essentials::logger("writing index");
        start = clock_type::now();
        binary_header bin_header;
        bin_header.remapping_order = 0;
        bin_header.data_structure_t = data_structure_type::pef_trie;
        bin_header.value_t = value_type::prob_backoff;
        util::save(bin_header.get(), index, m_config.output_filename.c_str());
        end = clock_type::now();
        elapsed = end - start;
        std::cerr << "flushing index took: " << elapsed.count() << " [sec]"
                  << std::endl;
        m_O_time = elapsed.count();
        m_I_time = m_stream_generator.I_time();
    }

private:
    configuration const& m_config;
    stream::uncompressed_stream_generator m_stream_generator;
    tmp::data& m_tmp_data;
    statistics& m_stats;

    tmp::statistics m_tmp_stats;
    size_t m_record_size;

    std::vector<uint64_t> m_pointers;
    std::vector<float_vector_type> m_probs;  // buffer of uncompressed probs

    suffix_trie_index::builder m_index_builder;

    uint64_t m_current_block_id;
    uint64_t m_fetched_block_id;
    uint64_t m_num_blocks;
    double m_CPU_time;
    double m_I_time;
    double m_O_time;
};
}  // namespace tongrams
