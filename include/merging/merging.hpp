#pragma once

#include "util.hpp"
#include "constants.hpp"
#include "stream.hpp"
#include "merge_utils.hpp"
#include "merging_writer.hpp"

namespace tongrams {

template <typename StreamGenerator>
struct merging {
    typedef cursor_comparator<prefix_order_comparator_type>
        cursor_comparator_type;

    merging(configuration const& config, tmp::data& /*tmp_data*/,
            tmp::statistics& /*tmp_stats*/, statistics& stats)
        : m_config(config)
        , m_writer(config, stats)
        , m_comparator(config.max_order)
        , m_cursors(cursor_comparator_type(config.max_order)) {}

    typedef typename StreamGenerator::block_type input_block_type;

    void run() {
        std::vector<std::string> filenames;
        {
            essentials::directory tmp_dir(m_config.tmp_dirname);
            for (auto const& filename : tmp_dir) {
                if (filename.extension == constants::file_extension::counts) {
                    filenames.push_back(filename.fullpath);
                }
            }
        }

        uint8_t N = m_config.max_order;
        size_t num_files_to_merge = filenames.size();
        assert(num_files_to_merge > 0);
        std::cerr << "merging " << num_files_to_merge << " files" << std::endl;

        uint64_t record_size = ngrams_block::record_size(N);
        uint64_t min_load_size = m_config.RAM / (2 * num_files_to_merge + 1) /
                                 record_size * record_size;
        uint64_t default_load_size =
            (64 * essentials::MiB) / record_size * record_size;
        uint64_t load_size = default_load_size;
        if (min_load_size < default_load_size) {
            std::cerr << "using min. load size of " << min_load_size
                      << " because not enough RAM is available" << std::endl;
            load_size = min_load_size;
        }
        assert(load_size % record_size == 0);

        for (auto const& filename : filenames) {
            m_stream_generators.emplace_back(N);
            auto& gen = m_stream_generators.back();
            gen.open(filename);
            assert(gen.size() == 0);
            gen.fetch_next_block(load_size);
        }

        auto get_block = [](StreamGenerator& gen) {
            auto* block = gen.get_block();
            assert(block->template is_sorted<prefix_order_comparator_type>(
                block->begin(), block->end()));
            return block;
        };

        assert(m_cursors.empty());
        for (uint64_t k = 0; k != m_stream_generators.size(); ++k) {
            auto& gen = m_stream_generators[k];
            auto* block = get_block(gen);
            cursor<typename input_block_type::iterator> c(block->begin(),
                                                          block->end(), k);
            m_cursors.push(c);
        }

        uint64_t num_ngrams_per_block = load_size / record_size;
        std::cerr << "num_ngrams_per_block = " << num_ngrams_per_block
                  << " ngrams" << std::endl;

        ngrams_block result(N);
        result.resize_memory(num_ngrams_per_block);
        result.reserve_index(num_ngrams_per_block);
        uint64_t num_ngrams = 0;

        m_writer.start();

        while (!m_cursors.empty()) {
            auto& top = m_cursors.top();
            auto min = *(top.range.begin);

            if (!result.size()) {
                result.push_back(min.data, min.data + N, *(min.value(N)));
                ++num_ngrams;
            } else {
                auto& back = result.back();
                bool equal = equal_to(min.data, back.data, sizeof_ngram(N));

                if (not equal) {
                    if (result.size() == num_ngrams_per_block) {
                        while (m_writer.size() > 0)
                            ;  // wait for flush
                        m_writer.push(result);

                        result.init(N);
                        result.resize_memory(num_ngrams_per_block);
                        result.reserve_index(num_ngrams_per_block);
                        assert(result.empty());
                    }

                    result.push_back(min.data, min.data + N, *(min.value(N)));
                    ++num_ngrams;

                } else {
                    *(back.value(N)) += *(min.value(N));
                }
            }

            ++(top.range.begin);

            if (top.range.begin == top.range.end) {
                auto& gen = m_stream_generators[top.index];
                gen.release_block();
                if (gen.eos()) {
                    assert(gen.empty());
                    gen.close_and_remove();
                    m_cursors.pop();
                } else {
                    gen.fetch_next_block(load_size);
                    auto* block = get_block(gen);
                    top.range.begin = block->begin();
                    top.range.end = block->end();
                }
            }

            m_cursors.heapify();
        }

        std::cerr << "MERGE DONE: " << num_ngrams << " " << int(N) << "-grams"
                  << std::endl;

        m_writer.push(result);
        m_writer.terminate();
    }

private:
    configuration const& m_config;
    std::deque<StreamGenerator> m_stream_generators;
    merging_writer m_writer;
    prefix_order_comparator_type m_comparator;

    min_heap<cursor<typename input_block_type::iterator>,
             cursor_comparator_type>
        m_cursors;
};

}  // namespace tongrams
