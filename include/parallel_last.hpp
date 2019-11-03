#pragma once

#include "constants.hpp"
#include "util.hpp"
#include "../external/tongrams/include/utils/util.hpp"

#include "index_types.hpp"
#include "stream2.hpp"

namespace tongrams {

/*
    multi-threaded version
*/

struct last {
    typedef stream::ngrams_block_partition IN;
    typedef stream::floats_vec<> float_vector_type;

    last(configuration const& config, tmp::data& tmp_data,
         tmp::statistics& tmp_stats, statistics& stats)
        : m_config(config)
        , m_stream_generator(config.max_order, 1)
        , m_tmp_data(tmp_data)
        , m_stats(stats)
        , m_tmp_stats(tmp_stats)
        , m_record_size(ngrams_block<payload>::record_size(config.max_order, 1))
        , m_index_builder(config.max_order, config, stats)

        , m_num_ngrams(config.max_order, 0)
        , m_pointers(config.max_order - 1, 0)

        , m_num_fetched_blocks(0)
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
                m_handle_ptr =
                    util::async_call(fetch_next_blocks, m_config.num_threads);
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

    std::function<void(uint64_t)> fetch_next_blocks = [&](uint64_t num_blocks) {
        m_num_fetched_blocks = 0;
        for (uint64_t i = 0; i < num_blocks; ++i) {
            if (m_fetched_block_id != m_num_blocks) {
                auto const& offsets =
                    m_tmp_data.blocks_offsets[m_fetched_block_id];
                size_t n = offsets.back();
                assert(n);
                m_stream_generator.sync_fetch_next_blocks(1, n * m_record_size);
                ++m_fetched_block_id;
                ++m_num_fetched_blocks;
            }
        }
    };

    ~last() {
        util::wait(m_handle_ptr);
        assert(m_num_fetched_blocks == 0);
    }

    void run() {
        uint64_t num_threads = m_config.num_threads;
        if (num_threads > 1) {
            --num_threads;  // reserve 1 thread for reading from disk
        }
        // std::cerr << "num_threads = " << num_threads << std::endl;

        uint64_t mod = num_threads * 5;
        std::vector<worker> workers;
        uint8_t N = m_config.max_order;

        auto start = clock_type::now();

        for (; m_current_block_id < m_num_blocks;) {
            // std::cerr << "fetching blocks..." << std::endl;
            util::wait(m_handle_ptr);
            uint64_t num_fetched_blocks = m_num_fetched_blocks;
            // std::cerr << "num_fetched_blocks = " << num_fetched_blocks <<
            // std::endl;
            assert(num_fetched_blocks <= num_threads);
            m_current_block_id += num_fetched_blocks;

            m_handle_ptr = util::async_call(fetch_next_blocks, num_threads);

            workers.reserve(num_fetched_blocks);

            for (uint64_t i = 0; i < num_fetched_blocks; ++i) {
                auto* block = m_stream_generator.get();
                m_stream_generator.processed(block);
                workers.emplace_back(m_tmp_stats.occs.front(), m_stats,
                                     m_index_builder, block->order());
                workers[i].block = block;
            }

            parallel_executor p(num_fetched_blocks);

            // 1. each thread calculates its own (relative) partition
            task_region(*(p.executor), [&](task_region_handle& trh) {
                for (uint64_t i = 0; i < num_fetched_blocks; ++i) {
                    trh.run([&, i] { workers[i].partition(); });
                }
            });

            // 2. prefix sum to calculate global partitions
            // std::cerr << "prefix summing on num_ngrams and pointers" <<
            // std::endl;
            for (uint8_t n = 0; n < N; ++n) {
                uint64_t off = m_num_ngrams[n];
                uint64_t ptr = n < N - 1 ? m_num_ngrams[n + 1] : 0;
                for (uint64_t i = 0; i < num_fetched_blocks; ++i) {
                    uint64_t x = workers[i].num_ngrams[n];
                    workers[i].num_ngrams[n] = off;
                    // std::cerr << "workers[" << i << "].num_ngrams[" << int(n)
                    // << "] = " << workers[i].num_ngrams[n] << std::endl;
                    off += x + 1;
                    if (n < N - 1) {
                        uint64_t y = workers[i].num_ngrams[n + 1];
                        workers[i].pointers[n] = ptr;
                        // std::cerr << "workers[" << i << "].pointers[" <<
                        // int(n) << "] = " << workers[i].pointers[n] <<
                        // std::endl;
                        ptr += y + 1;
                    }
                }
                // std::cerr << std::endl;
            }

            uint64_t vocab_size = m_stats.num_ngrams(1);

// std::cerr << "prefix summing on probs_offsets" << std::endl;
#pragma omp parallel for
            for (uint8_t n = 1; n < N;
                 ++n) {  // unigram probs/backoffs are put in a hash table
                for (uint32_t j = 0; j < vocab_size; ++j) {
                    uint64_t sum = m_tmp_data.probs_offsets[n][j];
                    for (uint32_t i = 0; i < num_fetched_blocks; ++i) {
                        uint64_t x = workers[i].probs_offsets[n][j];
                        workers[i].probs_offsets[n][j] = sum;
                        sum += x;
                    }
                }
            }

            // std::cerr << "processing" << std::endl;

            // 3. process
            task_region(*(p.executor), [&](task_region_handle& trh) {
                for (uint64_t i = 0; i < num_fetched_blocks; ++i) {
                    trh.run([&, i] {
                        workers[i].process();
                        workers[i].block = nullptr;
                    });
                }
            });

            if (m_current_block_id % mod == 0) {
                std::cerr << "processed " << m_current_block_id << "/"
                          << m_num_blocks << " blocks" << std::endl;
            }

            // std::cerr << "releasing blocks" << std::endl;
            m_stream_generator.release_processed_blocks();

            // 4. update global offsets with end of partitions
            for (uint8_t n = 0; n < N; ++n) {
                m_num_ngrams[n] = workers.back().num_ngrams[n];
                // std::cerr << "updated m_num_ngrams[" << int(n) << "] = " <<
                // m_num_ngrams[n] << std::endl;
                if (n < N - 1) {
                    m_pointers[n] = workers.back().num_ngrams[n + 1];
                    // std::cerr << "updated m_pointers[" << int(n) << "] = " <<
                    // m_pointers[n] << std::endl;
                }
            }

#pragma omp parallel for
            for (uint8_t n = 1; n < N; ++n) {
                // std::cerr << "updating " << int(n) << "-probs_offsets" <<
                // std::endl;
                for (uint32_t j = 0; j < vocab_size; ++j) {
                    m_tmp_data.probs_offsets[n][j] =
                        workers.back().probs_offsets[n][j];
                }
            }

            // std::cerr << "clearing workers" << std::endl;
            workers.clear();
        }

        auto end = clock_type::now();
        std::chrono::duration<double> elapsed = end - start;
        m_CPU_time += elapsed.count();

        m_stream_generator.close();  // close but no not destroy

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

    struct worker {
        worker(std::vector<occurrence> const& unigrams_counts,
               statistics& stats, suffix_trie_index::builder& index_builder,
               uint8_t N)

            : block(nullptr)
            , num_ngrams(N, 0)
            , probs_offsets(N, counts(0, 0))
            , pointers(N - 1, 0)

            , m_unigrams_counts(unigrams_counts)
            , m_stats(stats)
            // , m_tmp_data(tmp_data)
            , m_index_builder(index_builder)
            , m_tmp_stats(N)
            , m_probs(N, float_vector_type(0)) {
            size_t vocab_size = m_stats.num_ngrams(1);
            for (uint8_t n = 1; n < N; ++n) {
                m_tmp_stats.resize(n, vocab_size);
            }
            probs_offsets[0].resize(0, 0);
            for (uint8_t n = 2; n <= N; ++n) {
                probs_offsets[n - 1].resize(vocab_size, 0);
            }
        }

        void partition() {
            block->materialize_index(1);
            auto begin = block->begin();
            size_t len = block->size();
            // std::cerr << "block len = " << len << std::endl;
            uint8_t N = block->order();
            num_ngrams[N - 1] = len;
            auto prev_ptr = *begin;

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
                        ++num_ngrams[n -
                                     2];  // set num. ngrams of previous order
                    }
                    // else {

                    //     if (n > 1) {
                    //         word_id prev_left = prev_ptr[N - n - 1];
                    //         word_id      left =      ptr[N - n - 1];
                    //         if (left != prev_left) { // set pointers for
                    //         previous order
                    //             ++pointers[n - 2];
                    //         }
                    //     }

                    // }

                    word_id left = ptr[N - n - 1];
                    if (m_tmp_stats.update(n, left, right)) {
                        ++probs_offsets[n][right];
                    }
                }

                if (not ptr.equal_to(prev_ptr, 0, N - 1)) {
                    ++num_ngrams[3];
                }

                prev_ptr = ptr;
            }

            // std::cerr << "num ngrams:\n";
            // for (int i = 0; i < N; ++i) {
            //     std::cerr << "num " << (i + 1) << "-grams: "
            //               << num_ngrams[i] << std::endl;
            // }
            // std::cerr << std::endl;

            // std::cerr << "pointers:\n";
            // for (int i = 0; i < N - 1; ++i) {
            //     std::cerr << "num " << (i + 1) << "-pointers: "
            //               << pointers[i] << std::endl;
            // }
            // std::cerr << std::endl;

            // std::cerr << "probs_offsets:\n";
            // for (uint8_t n = 1; n <= N; ++n) {
            //     auto const& offs = probs_offsets[n - 1];
            //     std::cerr << int(n) << "-offs.size() = " << offs.size() <<
            //     std::endl;
            //     // std::cerr << "probs_offsets for " << int(n) <<
            //     "-grams:\n";
            //     // for (auto x: offs) {
            //     //     std::cerr << x << " ";
            //     // }
            //     // std::cerr << std::endl;
            // }
        }

        void process() {
            uint8_t N = block->order();
            for (uint8_t n = 1; n < N; ++n) {
                m_probs[n - 1].reserve(block->size());
            }

            /*
                n = 1: (empty context) the denominator is equal to the number of
               bi-grams n = N: the denominator is equal to the sum of the raw
               counts of N-grams having the same context otherwise: the
               denominator is equal to the sum of the modified counts of all
               n-grams having the same context
            */
            counts range_lengths(N, 0);
            counts probs_offs(N, 0);
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
                auto& pos =
                    num_ngrams[n - 2];  // always set values for previous order

                if (n != 2) {
                    word_id left = prev_ptr[N - n];
                    m_index_builder.set_word(n - 1, pos, left);
                }

                if (n != N) {
                    ++pointers[n - 2];
                    auto pointer = pointers[n - 2];
                    m_index_builder.set_pointer(n - 1, pos + 1, pointer);
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

                auto& offset = probs_offs[n - 1];
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

                    // int i = 0;
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
                            auto& pos = probs_offsets[n - 1][right];
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
                        // ++i;
                    });

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
                        auto& pos = probs_offsets[N - 1][right];

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
                    uint64_t uni_gram_count = m_unigrams_counts[context];
                    float prob = (static_cast<float>(uni_gram_count) -
                                  m_stats.D(1, uni_gram_count)) /
                                 uni_gram_denominator;
                    // NOTE: do not interpolate for tiny language models
                    prob += m_stats.unk_prob();  // interpolate
                    assert(prob <= 1.0);
                    m_index_builder.set_unigram_values(pos, prob, backoff);
                } else {
                    m_index_builder.set_backoff(n - 1, pos, backoff);
                }

                if (n != N) {  // reset next order's offset
                    probs_offs[n] = 0;
                }
                l = 0;
                ++pos;
            };

            while (iterators.back() != end) {
                for (uint8_t n = 2; n <= N; ++n) {
                    auto& it = iterators[n - 1];

                    if (it != end) {
                        auto prev_ptr = *it;

                        for (; it != end; ++it) {
                            auto ptr = *it;

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
                                            m_unigrams_counts[right];
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
                                        ++pointers[n - 2];
                                    }
                                }

                            } else {
                                break;  // here, iterator is one-past the end
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
        }

        IN* block;
        counts num_ngrams;
        std::vector<counts> probs_offsets;
        counts pointers;

    private:
        std::vector<occurrence> const& m_unigrams_counts;
        statistics& m_stats;
        // tmp::data& m_tmp_data;
        suffix_trie_index::builder& m_index_builder;
        tmp::statistics m_tmp_stats;
        std::vector<float_vector_type> m_probs;
        // uint8_t m_N;
    };

private:
    configuration const& m_config;
    stream::uncompressed_stream_generator m_stream_generator;
    tmp::data& m_tmp_data;
    statistics& m_stats;

    tmp::statistics m_tmp_stats;
    size_t m_record_size;
    suffix_trie_index::builder m_index_builder;

    counts m_num_ngrams;  // global offsets at which we have to write
                          // word_ids, pointers and backoffs

    counts m_pointers;

    std::unique_ptr<std::thread> m_handle_ptr;
    uint64_t m_num_fetched_blocks;
    uint64_t m_current_block_id;
    uint64_t m_fetched_block_id;
    uint64_t m_num_blocks;
    double m_CPU_time;
    double m_I_time;
    double m_O_time;
};
}  // namespace tongrams
