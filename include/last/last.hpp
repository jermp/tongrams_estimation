#pragma once

#include "../external/tongrams/include/utils/util.hpp"

#include "constants.hpp"
#include "util.hpp"
#include "stream.hpp"
#include "indexing/index_types.hpp"

namespace tongrams {

struct last {
    typedef stream::floats_vec<> float_vector_type;

    last(configuration const& config, tmp::data& tmp_data,
         tmp::statistics& tmp_stats, statistics& stats)
        : m_config(config)
        , m_stream_generator(config.max_order)
        , m_tmp_data(tmp_data)
        , m_stats(stats)
        , m_tmp_stats(tmp_stats)
        , m_record_size(ngrams_block::record_size(config.max_order))
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
        std::cout << "processing " << m_num_blocks << " blocks" << std::endl;
        uint8_t N = m_config.max_order;
        {
            essentials::directory tmp_dir(m_config.tmp_dirname);
            for (auto const& filename : tmp_dir) {
                if (filename.extension == constants::file_extension::merged) {
                    m_stream_generator.open(filename.fullpath);
                    async_fetch_next_block();
                    break;
                }
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
            assert(n > 0);
            m_stream_generator.async_fetch_next_block(n * m_record_size);
            ++m_fetched_block_id;
        }
    }

    void run() {
        auto start = clock_type::now();

        for (; m_current_block_id < m_num_blocks;) {
            auto* block = m_stream_generator.get_block();
            async_fetch_next_block();
            uint8_t N = block->order();

            for (uint8_t n = 1; n < N; ++n) {
                m_probs[n - 1].reserve(block->size());
            }

            /*
                - n = 1: (empty context) the denominator is equal to the number
               of bi-grams;
                - n = N: the denominator is equal to the sum of the raw
                counts of N-grams having the same context;
                - 1 < n < N (otherwise):
                the denominator is equal to the sum of the modified counts of
               all n-grams having the same context.
            */
            auto begin = block->begin();
            auto end = block->end();
            state s(N, begin, end);

            m_tmp_stats.clear();
            // m_tmp_stats.print_stats();

            while (s.iterators.back() != end) {
                for (uint8_t n = 2; n <= N; ++n) {
                    auto& it = s.iterators[n - 1];
                    if (it == end) continue;
                    auto prev_ptr = *it;
                    for (; it != end; ++it) {
                        auto ptr = *it;
                        // std::cout << "scanning " << int(n) << ": ";
                        // ptr.print(N);

                        bool context_changes =
                            !ptr.equal_to(prev_ptr, N - n, N - 1);
                        if (context_changes) break;

                        ++s.range_lengths[n - 1];
                        auto right = ptr[N - 1];

                        if (n == N) {
                            uint64_t count = *(ptr.value(N));
                            s.N_gram_denominator += count;
                            if (count < 5) {
                                ++m_tmp_stats.r[N - 1][count - 1];
                            } else {
                                ++m_tmp_stats.r[N - 1].back();
                            }
                        } else {
                            if (n == 2) {
                                float u = unigram_prob(right);
                                m_probs[0].push_back(u);
                            }

                            auto left = ptr[N - n - 1];
                            m_tmp_stats.update(n, left, right);
                            auto prev_left = prev_ptr[N - n - 1];
                            if (left != prev_left) ++m_pointers[n - 2];
                        }

                        prev_ptr = ptr;
                    }
                    write(n, s);
                }
            }

            // write last entries since [begin, end) is aligned
            // according to unigrams' boundaries
            for (uint8_t n = 2; n <= N; ++n) write(n, s);
            for (auto& p : m_probs) p.clear();

            ++m_current_block_id;
            if (m_current_block_id % 20 == 0) {
                std::cerr << "processed " << m_current_block_id << "/"
                          << m_num_blocks << " blocks" << std::endl;
            }

            m_stream_generator.release_block();
        }

        auto end = clock_type::now();
        std::chrono::duration<double> elapsed = end - start;
        m_CPU_time += elapsed.count();

        std::cerr << "processed " << m_current_block_id << "/" << m_num_blocks
                  << " blocks" << std::endl;

        std::vector<float_vector_type>().swap(m_probs);

        // Close but do not destroy: deleting large file from disk is expensive
        // and we can do this after construction is over.
        m_stream_generator.close();
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

    struct state {
        state(uint8_t N, ngrams_block::iterator begin,
              ngrams_block::iterator end)
            : range_lengths(N, 0)
            , probs_offsets(N, 0)
            , iterators(N, begin)
            , end(end)
            , N_gram_denominator(0) {}
        std::vector<uint64_t> range_lengths;
        std::vector<uint64_t> probs_offsets;
        std::vector<ngrams_block::iterator> iterators;
        const ngrams_block::iterator end;
        uint64_t N_gram_denominator;
    };

    float unigram_prob(word_id w);
    void write(uint8_t n, state& s);
};

}  // namespace tongrams