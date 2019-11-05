#pragma once

#include "counting_common.hpp"
#include "configuration.hpp"
#include "tmp.hpp"
#include "comparators.hpp"

namespace tongrams {

template <typename BlockWriter>
struct counting_writer {
    counting_writer(configuration const& config, tmp::data& tmp_data,
                    std::string const& file_extension)
        : m_tmp_data(tmp_data)
        , m_filename_gen(config.tmp_dirname, "", file_extension)
        , m_O_time(0.0)
        , m_CPU_time(0.0)
        , m_num_flushes(0)
        , m_writer(config.max_order)
        , m_comparator(config.max_order) {
        m_buffer.open();
    }

    ~counting_writer() {
        if (!m_buffer.empty()) {
            std::cerr << "Error: some data still need to be written"
                      << std::endl;
            std::terminate();
        }
    }

    void start() {
        m_thread = std::thread(&counting_writer::run, this);
    }

    void terminate() {
        m_buffer.lock();
        m_buffer.close();
        m_buffer.unlock();
        if (m_thread.joinable()) m_thread.join();
        assert(!m_buffer.active());
        while (!m_buffer.empty()) flush();
        std::cerr << "\tcounting_writer thread stats:\n";
        std::cerr << "\tflushed blocks: " << m_num_flushes << "\n";
        std::cerr << "\tO time: " << m_O_time << "\n";
        std::cerr << "\tCPU time: " << m_CPU_time << "\n";
    }

    void push(counting_step::block_type& block) {
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

    double CPU_time() const {
        return m_CPU_time;
    }

    double O_time() const {
        return m_O_time;
    }

private:
    tmp::data& m_tmp_data;
    semi_sync_queue<counting_step::block_type> m_buffer;
    std::thread m_thread;
    filename_generator m_filename_gen;
    double m_O_time;
    double m_CPU_time;
    uint64_t m_num_flushes;
    BlockWriter m_writer;
    context_order_comparator_type m_comparator;

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

        block.statistics().max_word_id = m_tmp_data.word_ids.size();

        auto start = clock_type::now();
        block.sort(m_comparator);
        auto end = clock_type::now();
        std::chrono::duration<double> elapsed = end - start;
        m_CPU_time += elapsed.count();
        essentials::logger("sorting took " + std::to_string(elapsed.count()));

        start = clock_type::now();
        std::string filename = m_filename_gen();
        std::ofstream os(filename.c_str(), std::ofstream::binary |
                                               std::ofstream::ate |
                                               std::ofstream::app);

        m_writer.write_block(os, block.begin(), block.end(), block.size(),
                             block.statistics());

        os.close();
        end = clock_type::now();
        elapsed = end - start;
        m_O_time += elapsed.count();

        block.release();

        m_buffer.lock();
        m_buffer.pop();
        m_buffer.unlock();
        ++m_num_flushes;
        m_filename_gen.next();
    }
};

}  // namespace tongrams