#pragma once

#include "configuration.hpp"
#include "tmp.hpp"

namespace tongrams {

struct merging_writer {
    merging_writer(configuration const& config, statistics& stats)
        : m_num_flushes(0), m_order(config.max_order), m_ngrams(0) {
        m_buffer.open();
        m_os.open(config.output_filename.c_str(),
                  std::ofstream::ate | std::ofstream::app);

        size_t vocab_size = stats.num_ngrams(1);
        if (!vocab_size) {
            throw std::runtime_error("vocabulary size must not be 0");
        }
        std::cerr << "vocabulary size: " << vocab_size << std::endl;
        std::cerr << "loading vocabulary..." << std::endl;
        size_t num_bytes =
            (sysconf(_SC_PAGESIZE) * sysconf(_SC_PHYS_PAGES)) / 2;
        vocabulary::builder vocab_builder(vocab_size, num_bytes);
        vocab_builder.load(config.vocab_tmp_subdirname + config.vocab_filename);
        vocab_builder.build(m_vocab);
        std::cerr << "done" << std::endl;

        //                               18446744073709551615\n
        static std::string empty_line = "                    \n";
        m_os.write(empty_line.data(), empty_line.size());

        m_params.path = config.output_filename;
        m_params.offset = 0;
        m_params.length = sysconf(_SC_PAGESIZE);
    }

    ~merging_writer() {
        if (!m_buffer.empty()) {
            std::cerr << "Error: some data still need to be written"
                      << std::endl;
            std::terminate();
        }
    }

    void start() {
        m_thread = std::thread(&merging_writer::run, this);
    }

    void terminate() {
        m_buffer.lock();
        m_buffer.close();
        m_buffer.unlock();
        if (m_thread.joinable()) m_thread.join();
        assert(!m_buffer.active());
        while (!m_buffer.empty()) flush();
        m_os.close();

        // write number of ngrams at the beginning of file
        boost::iostreams::mapped_file_sink tmp(m_params);
        char* data = tmp.data();
        std::string str = std::to_string(m_ngrams);
        memcpy(data, str.data(), str.size());
        tmp.close();

        std::cerr << "\tmerging_writer thread stats:\n";
        std::cerr << "\tflushed blocks: " << m_num_flushes << "\n";
    }

    void push(ngrams_block& block) {
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

private:
    semi_sync_queue<ngrams_block> m_buffer;
    std::ofstream m_os;
    std::thread m_thread;
    uint64_t m_num_flushes;
    uint64_t m_order;
    uint64_t m_ngrams;
    vocabulary m_vocab;

    boost::iostreams::mapped_file_params m_params;

    void run() {
        while (m_buffer.active()) flush();
    }

    void flush() {
        m_buffer.lock();
        if (m_buffer.empty()) {
            m_buffer.unlock();
            return;
        }
        auto& block = m_buffer.pick();
        m_buffer.unlock();

        for (auto ngram : block) {
            for (uint64_t i = 0; i != m_order; ++i) {
                auto br = m_vocab[ngram[i]];
                util::write(m_os, br);
                if (i != m_order - 1) m_os << " ";
            }
            m_os << "\t" << *ngram.value(m_order) << "\n";
        }

        m_ngrams += block.size();
        block.release();

        m_buffer.lock();
        m_buffer.pop();
        m_buffer.unlock();
        ++m_num_flushes;
        if (m_num_flushes % 20 == 0) {
            std::cerr << "flushed " << m_num_flushes << " blocks" << std::endl;
        }
    }
};

}  // namespace tongrams