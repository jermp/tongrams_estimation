#pragma once

#include "util.hpp"
#include "constants.hpp"
#include "stream2.hpp"
#include "statistics.hpp"
#include "merge_utils.hpp"

#include "adjusting_common.hpp"
#include "adjusting_writer.hpp"

namespace tongrams {

template <typename StreamGenerator>
struct adjusting {
    adjusting(configuration const& config, tmp::data& tmp_data,
              tmp::statistics& tmp_stats, statistics& stats)
        : m_config(config)
        , m_tmp_data(tmp_data)
        , m_stats(stats)
        , m_stats_builder(config, tmp_data, tmp_stats)
        , m_writer(config, constants::file_extension::merged)
        , m_comparator(config.max_order)
        , m_cursor_comparator(config.max_order)
        , m_cursors(m_cursor_comparator)
        , m_record_size(
              ngrams_block<count_type>::record_size(config.max_order, 1))
        , m_num_bytes(
              ngrams_block<count_type>::record_size(config.max_order, 0))

        , m_CPU_time(0.0)
        , m_I_time(0.0)
        , m_O_time(0.0)
        , m_total_smooth_time(0.0)
        , m_total_time_waiting_for_disk(0.0) {
        auto start = clock_type::now();
        std::vector<std::string> filenames;
        directory tmp_dir(config.tmp_dirname);
        for (auto const& path : tmp_dir) {
            if (!is_directory(path) and is_regular_file(path)) {
                if (path.extension() == constants::file_extension::counts) {
                    filenames.push_back(path.string());
                }
            }
        }

        size_t num_files_to_merge = filenames.size();
        std::cerr << "merging " << num_files_to_merge << " files" << std::endl;

        uint64_t min_load_size = m_config.RAM / (2 * num_files_to_merge + 1) /
                                 m_record_size * m_record_size;
        uint64_t default_load_size =
            (64 * essentials::MiB) / m_record_size * m_record_size;
        m_load_size = default_load_size;
        if (min_load_size < default_load_size) {
            std::cerr << "\tusing min. load size of " << min_load_size
                      << " because not enough RAM is available" << std::endl;
            m_load_size = min_load_size;
        }
        assert(m_load_size % m_record_size == 0);

        for (auto const& fn : filenames) {
            m_stream_generators.emplace_back(m_config.max_order, 1);
            auto& sg = m_stream_generators.back();
            sg.open(fn);
            assert(sg.size() == 0);
            sg.fetch_next_block(m_load_size);
        }

        size_t vocab_size = m_stats.num_ngrams(1);
        if (!vocab_size) {
            throw std::runtime_error("vocabulary size must not be 0");
        }

        std::cerr << "vocabulary size: " << vocab_size << std::endl;
        tmp_stats.resize(1, vocab_size);
        m_stats_builder.init(vocab_size);

        auto end = clock_type::now();
        std::chrono::duration<double> elapsed = end - start;
        m_CPU_time += elapsed.count();
    }

    typedef typename StreamGenerator::block_type input_block_type;

    void print_stats() const {
        std::cout << "\"CPU\":" << m_CPU_time << ", ";
        std::cout << "\"I\":" << m_I_time << ", ";
        std::cout << "\"O\":" << m_O_time << ", ";
    }

    void run() {
        m_writer.start();
        auto start = clock_type::now();

        assert(m_cursors.empty());
        for (uint64_t k = 0; k != m_stream_generators.size(); ++k) {
            auto& sg = m_stream_generators[k];
            auto* ptr = sg.get();
            assert(ptr);
            ptr->materialize_index(1);
            assert(ptr->template is_sorted<context_order_comparator_type>(
                ptr->begin(), ptr->end()));
            cursor<typename input_block_type::iterator> c(ptr->begin(),
                                                          ptr->end(), k);
            m_cursors.push(c);
        }

        // std::cerr << "DONE" << std::endl;

        uint64_t num_ngrams_per_block = m_load_size / m_record_size;
        uint8_t N = m_config.max_order;
        adjusting_step::output_block_type result(N, 1);
        result.resize_memory(num_ngrams_per_block, 1);
        result.reserve_index(num_ngrams_per_block);
        result.range.begin = 0;
        auto& index = result.index();

        uint64_t partition_size = num_ngrams_per_block;
        uint64_t limit = num_ngrams_per_block;

        std::cerr << "num_ngrams_per_block = " << num_ngrams_per_block
                  << " ngrams" << std::endl;
        std::cerr << "partition_size = " << partition_size << " ngrams"
                  << std::endl;
        std::cerr << "limit = " << limit << " ngrams" << std::endl;

        auto compute_smoothing_statistics = [&]() {
            result.range.end = index.size();
            auto start = clock_type::now();
            m_stats_builder.compute_smoothing_statistics(result.begin(),
                                                         result.size());
            auto end = clock_type::now();
            std::chrono::duration<double> elapsed = end - start;
            m_total_smooth_time += elapsed.count();
        };

        uint64_t num_Ngrams = 0;
        uint64_t prev_offset = 0;

        auto save_offsets = [&]() {
            uint64_t offset = num_Ngrams - prev_offset;
            std::vector<uint64_t> offsets = {offset};
            m_tmp_data.blocks_offsets.push_back(std::move(offsets));
            prev_offset = num_Ngrams;
            limit = num_Ngrams + partition_size;
        };

        equal_to equal_pred;

        while (!m_cursors.empty()) {
            auto& top = m_cursors.top();
            auto min = *(top.range.begin);

            // std::cerr << "min element: ";
            // min.print(5, 1);

            if (!index.size()) {
                result.push_back(min.data, min.data + N, *(min.value(N)));
                ++num_Ngrams;

            } else {
                bool equal =
                    equal_pred(min.data, result.back().data, m_num_bytes);

                if (not equal) {
                    bool greater =
                        compare_i<typename input_block_type::pointer>(
                            min, result.back(), m_comparator.begin()) > 0;

                    if (num_Ngrams >= limit and greater) {
                        save_offsets();
                    }

                    if (index.size() == num_ngrams_per_block) {
                        compute_smoothing_statistics();

#ifndef NDEBUG
                        assert(result.template is_sorted<
                               context_order_comparator_type>(result.begin(),
                                                              result.end()));
#endif

                        auto start = clock_type::now();
                        while (m_writer.size() > 0)
                            ;  // wait for flush
                        auto end = clock_type::now();
                        std::chrono::duration<double> elapsed = end - start;
                        m_total_time_waiting_for_disk += elapsed.count();

                        m_writer.push(result);

                        // re-init result block
                        result.init(N, 1);
                        result.resize_memory(num_ngrams_per_block, 1);
                        result.reserve_index(num_ngrams_per_block);
                        result.range.begin = 0;
                        assert(index.empty());
                    }

                    result.push_back(min.data, min.data + N, *(min.value(N)));
                    ++num_Ngrams;

                } else {
                    // combine the two values, by updating the one in result
                    auto& back = result.back();
                    auto const& combined_value = count_type::combine_values(
                        *(back.value(N)), *(min.value(N)));
                    *(back.value(N)) = combined_value;
                }
            }

            ++(top.range.begin);

            if (top.range.begin == top.range.end) {
                // std::cerr << "block exhausted" << std::endl;
                auto& sg = m_stream_generators[top.index];
                auto* ptr = sg.get();
                assert(ptr);
                sg.processed(ptr);
                sg.release_processed_blocks();

                if (sg.empty() and sg.eos()) {
                    sg.close_and_remove();
                    m_cursors.pop();
                } else {
                    sg.fetch_next_block(m_load_size);
                    auto* ptr = sg.get();
                    assert(ptr);
                    ptr->materialize_index(1);
                    assert(
                        ptr->template is_sorted<context_order_comparator_type>(
                            ptr->begin(), ptr->end()));

                    // update top
                    top.range.begin = ptr->begin();
                    top.range.end = ptr->end();
                }
            }

            m_cursors.heapify();
        }

        std::cerr << "MERGE DONE: " << num_Ngrams << " N-grams" << std::endl;
        std::cerr << "\ttime waiting for disk = "
                  << m_total_time_waiting_for_disk << " [sec]\n";
        std::cerr << "\tsmoothing time: " << m_total_smooth_time << " [sec]"
                  << std::endl;

        save_offsets();

        compute_smoothing_statistics();
        m_stats_builder.finalize();

        auto end = clock_type::now();
        std::chrono::duration<double> elapsed = end - start;
        m_CPU_time += elapsed.count();

        m_writer.push(result);

        m_writer.terminate();

        m_CPU_time -= m_total_time_waiting_for_disk;

        // double max_read_IO_time = 0.0;
        // double sum = 0.0;
        for (auto& sg : m_stream_generators) {
            // m_I_time += sg.IO_time();
            m_I_time += sg.I_time();
            // std::cerr << "read IO time: " << sg.IO_time() << " [sec]" <<
            // std::endl; if (sg.IO_time() > max_read_IO_time) {
            //     max_read_IO_time = sg.IO_time();
            // }
        }
        // std::cerr << "sum of read IO times = " << sum << " [sec]" <<
        // std::endl; std::cerr << "max_read_IO_time = " << max_read_IO_time <<
        // " [sec]" << std::endl; m_I_time = sum;

        start = clock_type::now();
        m_stats_builder.build(m_stats);
        end = clock_type::now();
        elapsed = end - start;
        m_CPU_time += elapsed.count();
        m_CPU_time -= m_I_time;
        m_O_time += m_writer.time();
    }

private:
    configuration const& m_config;
    tmp::data& m_tmp_data;
    statistics& m_stats;
    statistics::builder m_stats_builder;
    std::deque<StreamGenerator> m_stream_generators;
    adjusting_writer m_writer;
    context_order_comparator_type m_comparator;

    cursor_comparator<context_order_comparator_type> m_cursor_comparator;

    min_heap<cursor<typename input_block_type::iterator>,
             cursor_comparator<context_order_comparator_type> >
        m_cursors;

    uint64_t m_record_size;
    uint64_t m_num_bytes;
    uint64_t m_load_size;

    double m_CPU_time;
    double m_I_time;
    double m_O_time;
    double m_total_smooth_time;
    double m_total_time_waiting_for_disk;
};
}  // namespace tongrams
