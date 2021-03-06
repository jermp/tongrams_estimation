#pragma once

#include "vocabulary.hpp"
#include "tmp.hpp"
#include "statistics.hpp"
#include "stream.hpp"
#include "counting/counting.hpp"
#include "merging/merging.hpp"

namespace tongrams {

struct counter {
    counter(configuration const& config)
        : m_config(config)
        , m_tmp_data()
        , m_tmp_stats(config.max_order)
        , m_stats(config.max_order) {
        m_timings.reserve(2);
        std::cout << "{";
        std::cout << "\"dataset\":"
                  << boost::filesystem::path(config.text_filename).stem()
                  << ", ";
        std::cout << "\"order\":" << config.max_order << ", ";
        std::cout << "\"RAM\":" << config.RAM << ", ";
        std::cout << "\"threads\":" << config.num_threads;
    }

    ~counter() {
        std::cout << "}" << std::endl;
    }

    void run() {
        if (m_config.compress_blocks) {
            typedef fc::writer<prefix_order_comparator_type> block_writer_type;
            run<counting<block_writer_type, prefix_order_comparator_type>>(
                "counting");
        } else {
            run<counting<stream::writer, prefix_order_comparator_type>>(
                "counting");
        }

        m_stats.num_ngrams(1) = m_tmp_data.word_ids.size();
        m_tmp_data.word_ids.clear();
        // write_vocab();

        if (m_config.compress_blocks) {
            run<merging<stream::compressed_stream_generator>>("merging");
        } else {
            run<merging<stream::uncompressed_stream_generator>>("merging");
        }
    }

    void print_stats() {
        int step = 1;
        for (auto t : m_timings) {
            std::cerr << "step-" << step << ": " << t << " [sec]\n";
            ++step;
        }
    }

private:
    configuration const& m_config;
    tmp::data m_tmp_data;
    tmp::statistics m_tmp_stats;
    statistics m_stats;
    std::vector<double> m_timings;

    template <typename Step>
    void run(std::string const& name) {
        std::cout << ", ";
        std::cout << "\"" + name + "\": {";
        auto start = clock_type::now();
        Step step(m_config, m_tmp_data, m_tmp_stats, m_stats);
        step.run();
        auto end = clock_type::now();
        std::chrono::duration<double> elapsed = end - start;
        double total_time = elapsed.count();
        m_timings.push_back(total_time);
        std::cout << "\"total\":" << total_time;
        std::cout << "}";
    }

    // std::function<void(void)> write_vocab = [&]() {
    //     std::ofstream os(m_config.vocab_tmp_subdirname +
    //                      m_config.vocab_filename);
    //     size_t vocab_size = m_stats.num_ngrams(1);
    //     vocabulary vocab;
    //     m_tmp_data.vocab_builder.build(vocab);
    //     for (size_t id = 0; id != vocab_size; ++id) {
    //         util::write(os, vocab[id]);
    //         os << "\n";
    //     }
    //     os.close();
    // };
};
}  // namespace tongrams
