#pragma once

#include "configuration.hpp"
#include "tmp.hpp"
#include "constants.hpp"
#include "statistics.hpp"
#include "util.hpp"
#include "util_types.hpp"

#include "../external/tongrams/include/utils/util.hpp"

#include "counting_common.hpp"
#include "counting_writer.hpp"
#include "counting_reader.hpp"

namespace tongrams {

template <typename BlockWriter, typename Comparator>
struct counting {
    counting(configuration const& config, tmp::data& tmp_data, tmp::statistics&,
             statistics&)
        : m_config(config)
        , m_CPU_time(0.0)
        , m_I_time(0.0)
        , m_writer(config, tmp_data, constants::file_extension::counts)
        , m_reader(config, tmp_data, m_writer) {
        tmp_data.vocab_builder.push_empty();
        tmp_data.word_ids[hash_utils::hash_empty_token] =
            constants::empty_token_word_id;
    }

    void run() {
        bool file_begin = true;
        bool file_end = false;
        static constexpr uint64_t mm_region_size = 1 * essentials::GiB;
        uint64_t blocks = util::ceil_div(m_config.text_size, mm_region_size);
        uint64_t page_size = sysconf(_SC_PAGESIZE);
        assert(mm_region_size >= page_size and mm_region_size % page_size == 0);

        m_writer.start();

        for (uint64_t block = 0,
                      page_id = 0;  // disk page containing the beginning of
                                    // current file block
             block != blocks; ++block) {
            uint64_t chunk_size = mm_region_size;
            uint64_t offset = page_id * page_size;
            if (offset + chunk_size > m_config.text_size) {
                file_end = true;
                chunk_size = m_config.text_size - offset;
            }

            m_data = util::open_file_partition(m_file, m_config.text_filename,
                                               chunk_size, offset);
            uint64_t begin = 0;
            uint64_t end = m_file.size();
            assert(end > 0);

            util::optimize_sequential_access(m_data, end);

            if (!file_begin) align_forward(begin);
            std::string boundary = m_boundary;
            m_boundary.clear();
            if (!is_aligned(end - 1)) align_backward(begin, --end);

            uint64_t n = end;
            assert(n != 0 and n <= mm_region_size);
            m_reader.init(m_data, boundary, begin, end, file_begin, file_end);
            m_reader.run();
            file_begin = false;

            uint64_t num_pages = util::ceil_div(n, page_size);
            assert(num_pages > 0);
            page_id += num_pages;
            m_file.close();
        }

        m_CPU_time += m_reader.CPU_time();
        m_I_time += m_reader.I_time();
        m_writer.terminate();
        m_reader.print_stats();
    }

    void print_stats() const {
        std::cout << "\"CPU\":" << m_CPU_time << ", ";
        std::cout << "\"I\":" << m_I_time << ", ";
        std::cout << "\"O\":" << m_writer.O_time() << ", ";
    }

private:
    bool is_aligned(uint64_t pos) const {
        return m_data[pos] == ' ' or m_data[pos] == '\n';
    }

    void align_forward(uint64_t& begin) {
        for (;; ++begin) {
            auto c = m_data[begin];
            if (c == ' ' or c == '\n') {
                ++begin;  // first char after a whitespace
                break;
            }
            m_boundary.push_back(c);
        }
    }

    void align_backward(uint64_t begin, uint64_t& end) {
        for (; begin != end; --end) {
            auto c = m_data[end];
            if (c == ' ' or c == '\n') {
                ++end;  // one-past
                std::reverse(m_boundary.begin(), m_boundary.end());
                break;
            }
            m_boundary.push_back(c);
        }
    }

    configuration const& m_config;
    boost::iostreams::mapped_file_source m_file;
    uint8_t const* m_data;

    std::string m_boundary;

    double m_CPU_time;
    double m_I_time;

    typedef counting_writer<BlockWriter, Comparator> counting_writer_type;
    typedef counting_reader<counting_writer_type> counting_reader_type;
    counting_writer_type m_writer;
    counting_reader_type m_reader;
};

}  // namespace tongrams
