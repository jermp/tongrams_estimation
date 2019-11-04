#pragma once

#include "adjusting_common.hpp"
#include "configuration.hpp"
#include "tmp.hpp"

namespace tongrams {

struct adjusting_writer {
    adjusting_writer(configuration const& config,
                     std::string const& file_extension)
        : m_num_flushes(0), m_time(0.0) {
        m_buffer.open();
        filename_generator gen(config.tmp_dirname, "", file_extension);
        std::string output_filename = gen();
        m_os.open(output_filename.c_str(), std::ofstream::binary |
                                               std::ofstream::ate |
                                               std::ofstream::app);
    }

    ~adjusting_writer() {
        if (not m_buffer.empty()) {
            std::cerr << "Error: some data still need to be written"
                      << std::endl;
            std::terminate();
        }
    }

    void start() {
        m_thread = std::thread(&adjusting_writer::run, this);
    }

    void terminate() {
        m_buffer.lock();
        m_buffer.close();
        m_buffer.unlock();
        if (m_thread.joinable()) m_thread.join();
        assert(!m_buffer.active());
        while (!m_buffer.empty()) flush();
        m_os.close();
        std::cerr << "\tadjusting_writer thread stats:\n";
        std::cerr << "\tflushed blocks: " << m_num_flushes << "\n";
        std::cerr << "\twrite time: " << m_time << "\n";
    }

    void push(adjusting_step::output_block_type& block) {
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
    semi_sync_queue<adjusting_step::output_block_type> m_buffer;
    std::ofstream m_os;
    std::thread m_thread;
    uint64_t m_num_flushes;
    double m_time;

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
            std::cerr << "flushed " << m_num_flushes << " blocks" << std::endl;
        }
    }
};

}  // namespace tongrams