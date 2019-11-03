#pragma once

#include "util.hpp"
#include "constants.hpp"
#include "stream2.hpp"
#include "statistics.hpp"
#include "merge_utils.hpp"

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
        , m_record_size(ngrams_block<payload>::record_size(config.max_order, 1))
        , m_num_bytes(ngrams_block<payload>::record_size(config.max_order, 0))
        , m_cached_blocks_x_file(1)

        , m_CPU_time(0.0)
        , m_I_time(0.0)
        , m_O_time(0.0)
        , m_total_smooth_time(0.0)
        , m_total_time_waiting_for_disk(0.0) {
        auto start = clock_type::now();
        std::vector<std::string> filenames;
        directory tmp_dir(config.tmp_dirname);
        for (auto const& path : tmp_dir) {
            if (not is_directory(path) and is_regular_file(path)) {
                if (path.extension() == constants::file_extension::counts) {
                    filenames.push_back(path.string());
                }
            }
        }

        size_t k = filenames.size();
        std::cerr << "merging " << k << " files" << std::endl;
        m_cursors.init(k, m_cursor_comparator);

        // min values
        // uint64_t min_load_size = uint64_t(-1);
        uint64_t min_load_size =
            m_config.RAM / (2 * k + 1) / m_record_size * m_record_size;
        uint64_t default_load_size =
            64 * essentials::MB / m_record_size * m_record_size;
        // uint64_t default_load_size = m_config.RAM / (k + 1) / m_record_size *
        // m_record_size;

        if (min_load_size < default_load_size) {
            std::cerr << "\tusing min. load size of " << min_load_size
                      << " because not enough RAM is available" << std::endl;
            m_load_size = min_load_size;
        } else {
            m_load_size = default_load_size;
            // std::cerr << "using default load size" << std::endl;
            // m_cached_blocks_x_file = m_config.RAM / (k + 1) / m_load_size;
            // m_cached_blocks_x_file = 1;
            // m_cached_blocks_x_file = 2;
        }

        // std::cerr << "loading " << m_load_size << " bytes x file" <<
        // std::endl; std::cerr << "m_cached_blocks_x_file = " <<
        // m_cached_blocks_x_file << std::endl;

        assert(m_load_size % m_record_size == 0);

        for (auto const& fn : filenames) {
            m_stream_generators.emplace_back(m_config.max_order, 1);
            auto& sg = m_stream_generators.back();
            sg.open(fn);
            assert(sg.size() == 0);

            // auto start = clock_type::now();
            // for (uint64_t k = 0; k < m_cached_blocks_x_file; ++k) {
            sg.sync_fetch_next_blocks(m_cached_blocks_x_file, m_load_size);
            // }
            // auto end = clock_type::now();
            // std::chrono::duration<double> elapsed = end - start;
            // m_I_time += elapsed.count();
            // std::cout << "m_O_time = " << m_O_time << std::endl;
        }

        assert(m_stream_generators.size() == k);

        size_t vocab_size = m_stats.num_ngrams(1);
        // size_t vocab_size = 2438616; // 1Billion
        // size_t vocab_size = 5681625; // Wikipedia
        // size_t vocab_size = 8769460; // Gutenberg
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
    typedef stream::ngrams_block_partition output_block_type;

    void print_stats() const {
        std::cout << "\"CPU\":" << m_CPU_time << ", ";
        std::cout << "\"I\":" << m_I_time << ", ";
        std::cout << "\"O\":" << m_O_time << ", ";
    }

    void run() {
        m_writer.start();

        auto start = clock_type::now();

        assert(m_stream_generators.size() == m_cursors.k());
        assert(m_cursors.empty());

        // std::cerr << "initializing cursors" << std::endl;
        for (uint64_t k = 0; k < m_cursors.k(); ++k) {
            auto& sg = m_stream_generators[k];
            auto* ptr = sg.get();
            assert(ptr);
            ptr->materialize_index(1);
            assert(ptr->template is_sorted<context_order_comparator_type>(
                ptr->begin(), ptr->end()));
            // std::cerr << "creating cursor" << std::endl;
            cursor<typename input_block_type::iterator> c(ptr->begin(),
                                                          ptr->end(), k);
            // std::cerr << "inserting into heap" << std::endl;
            m_cursors.insert(c);
        }

        // std::cerr << "DONE" << std::endl;

        size_t num_ngrams_x_block = m_load_size / m_record_size;
        // size_t num_ngrams_x_block = 64 * 1024 * 1024 / m_record_size;

        uint8_t N = m_config.max_order;
        output_block_type result(N, 1);
        result.resize_memory(num_ngrams_x_block, 1);
        result.reserve_index(num_ngrams_x_block);
        result.range.begin = 0;
        auto& index = result.index();

        // TODO: remove num_threads, since it is always 1
        uint64_t num_threads = 1;
        std::vector<uint64_t> offsets;
        offsets.reserve(num_threads);

        // determines the size of the block during last step
        // size_t partition_size = m_config.RAM / 2 / m_record_size;
        size_t partition_size = num_ngrams_x_block;
        size_t range_len = partition_size / num_threads;
        size_t limit = range_len;

        std::cerr << "num_ngrams_x_block = " << num_ngrams_x_block << " ngrams"
                  << std::endl;
        std::cerr << "partition_size = " << partition_size << " ngrams"
                  << std::endl;
        std::cerr << "range_len = " << range_len << " ngrams" << std::endl;
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
            size_t offset = num_Ngrams - prev_offset;
            offsets.push_back(offset);
            m_tmp_data.blocks_offsets.push_back(std::move(offsets));
            offsets.clear();
            prev_offset = num_Ngrams;
            limit = num_Ngrams + range_len;
        };

        // context_order_comparator_SIMD<ngram_pointer<payload>> cmp(5);

        equal_to equal_pred;

        while (not m_cursors.empty()) {
            auto& top = m_cursors.top();
            auto min = *(top.range.begin);

            // std::cerr << "min element: ";
            // min.print(5, 1);

            if (!index.size()) {
                result.push_back(min.data, min.data + N, *(min.value(N)));
                ++num_Ngrams;

            } else {
                // auto start = clock_type::now();
                bool equal =
                    equal_pred(min.data, result.back().data, m_num_bytes);
                // auto end = clock_type::now();
                // std::chrono::duration<double> elapsed = end - start;
                // comparison_time += elapsed.count();

                if (not equal) {
                    // auto start = clock_type::now();
                    bool greater =
                        compare_i<typename input_block_type::pointer>(
                            min, result.back(), m_comparator.begin()) > 0;
                    // auto end = clock_type::now();
                    // std::chrono::duration<double> elapsed = end - start;
                    // comparison_time += elapsed.count();

                    if (num_Ngrams >= limit and greater) {
                        save_offsets();
                    }

                    if (index.size() == num_ngrams_x_block) {
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
                        result.resize_memory(num_ngrams_x_block, 1);
                        result.reserve_index(num_ngrams_x_block);
                        result.range.begin = 0;
                        assert(index.empty());
                    }

                    result.push_back(min.data, min.data + N, *(min.value(N)));
                    ++num_Ngrams;

                } else {  // combine the two values, by updating the one already
                          // in result
                    auto& back = result.back();
                    auto const& combined_value = payload::combine_values(
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
                    // std::cerr << "loading next block" << std::endl;
                    if (m_cached_blocks_x_file == 1) {
                        sg.sync_fetch_next_blocks(1, m_load_size);
                        // auto start = clock_type::now();
                        // sg.fetch_next(m_load_size);
                        // auto end = clock_type::now();
                        // std::chrono::duration<double> elapsed = end - start;
                        // m_I_time += elapsed.count();
                    }

                    auto* ptr = sg.get();
                    assert(ptr);
                    ptr->materialize_index(1);
                    assert(
                        ptr->template is_sorted<context_order_comparator_type>(
                            ptr->begin(), ptr->end()));

                    // ASYNCH
                    // NOTE: as soon as a block is done, we fetch another,
                    // in order to always have m_cached_blocks_x_file blocks
                    // this can introduce many disk seeks
                    // x file in RAM
                    // if (m_cached_blocks_x_file > 1) {
                    //     sg.async_fetch_next_block(m_load_size);
                    // }

                    // SYNCH
                    // NOTE: fetch all blocks sequentially when cache is empty
                    // if (sg.size() == 1) {
                    //     std::cerr << "fetching blocks" << std::endl;
                    //     sg.fetch_blocks(m_cached_blocks_x_file - 1,
                    //     m_load_size);
                    //     // for (uint64_t k = 0; k < m_cached_blocks_x_file -
                    //     1; ++k) {
                    //     //     sg.fetch_next(m_load_size);
                    //     // }
                    //     std::cerr << "DONE" << std::endl;
                    // }

                    // update top
                    top.range.begin = ptr->begin();
                    top.range.end = ptr->end();
                }
            }

            // auto start = clock_type::now();

            // std::cerr << "heapifying..." << std::endl;
            m_cursors.heapify();
            // std::cerr << "DONE" << std::endl;

            // auto end = clock_type::now();
            // std::chrono::duration<double> elapsed = end - start;
            // comparison_time += elapsed.count();
        }

        std::cerr << "MERGE DONE: " << num_Ngrams << " N-grams" << std::endl;
        // std::cerr << "\ttime for comparisons = "
        //           << comparison_time << " [sec]\n";
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

    struct writer {
        writer(configuration const& config, std::string const& file_extension)
            : m_num_flushes(0), m_time(0.0) {
            m_buffer.open();
            filename_generator gen(config.tmp_dirname, "", file_extension);
            std::string output_filename = gen();
            m_os.open(output_filename.c_str(), std::ofstream::binary |
                                                   std::ofstream::ate |
                                                   std::ofstream::app);
        }

        ~writer() {
            if (not m_buffer.empty()) {
                std::cerr << "Error: some data still need to be written"
                          << std::endl;
                std::terminate();
            }
        }

        void start() {
            m_thread = std::thread(&writer::run, this);
        }

        void terminate() {
            m_buffer.lock();
            m_buffer.close();
            m_buffer.unlock();

            if (m_thread.joinable()) {
                m_thread.join();
            }

            // std::cout << "writer thread terminating" << std::endl;
            assert(not m_buffer.active());
            while (not m_buffer.empty()) {
                flush();
            }

            m_os.close();

            std::cerr << "\twriter thread stats:\n";
            std::cerr << "\tflushed blocks: " << m_num_flushes << "\n";
            std::cerr << "\twrite time: " << m_time << "\n";
        }

        void push(output_block_type& block) {
            m_buffer.lock();
            m_buffer.push(block);
            m_buffer.unlock();
        }

        size_t size() {
            m_buffer.lock();
            size_t s = m_buffer.size();
            m_buffer.unlock();
            return s;
        }

        double time() const {
            return m_time;
        }

    private:
        semi_sync_queue<output_block_type> m_buffer;
        std::ofstream m_os;
        std::thread m_thread;
        uint64_t m_num_flushes;

        bool m_compress;
        double m_time;

        void run() {
            while (m_buffer.active()) {
                flush();
            }
        }

        void flush() {
            m_buffer.lock();
            if (m_buffer.empty()) {
                m_buffer.unlock();
                return;
            }
            auto& block = m_buffer.pick();
            m_buffer.unlock();

            auto start = clock_type::now();
            block.write_memory(m_os);
            auto end = clock_type::now();
            std::chrono::duration<double> elapsed = end - start;
            m_time += elapsed.count();

            block.release();

            m_buffer.lock();
            m_buffer.pop();
            m_buffer.unlock();
            ++m_num_flushes;
            if (m_num_flushes % 20 == 0) {
                std::cerr << "flushed " << m_num_flushes << " blocks"
                          << std::endl;
            }
        }
    };

private:
    configuration const& m_config;
    tmp::data& m_tmp_data;
    statistics& m_stats;
    statistics::builder m_stats_builder;
    std::deque<StreamGenerator> m_stream_generators;
    writer m_writer;
    context_order_comparator_type m_comparator;

    min_heap<cursor<typename input_block_type::iterator>,
             cursor_comparator<context_order_comparator_type> >
        m_cursors;

    cursor_comparator<context_order_comparator_type> m_cursor_comparator;

    uint64_t m_record_size;
    uint64_t m_num_bytes;
    uint64_t m_cached_blocks_x_file;
    uint64_t m_load_size;

    double m_CPU_time;
    double m_I_time;
    double m_O_time;
    double m_total_smooth_time;
    double m_total_time_waiting_for_disk;
};
}  // namespace tongrams
