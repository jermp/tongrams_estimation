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

template <typename BlockWriter>
struct counting {
    counting(configuration const& config, tmp::data& tmp_data, tmp::statistics&,
             statistics&)
        : m_config(config)
        , m_CPU_time(0.0)
        , m_I_time(0.0)
        , m_writer(config, tmp_data, constants::file_extension::counts)
        , m_reader(config, tmp_data, m_writer) {
        uint64_t hash_empty = hash_utils::hash_empty;
        tmp_data.vocab_builder.push_empty();
        tmp_data.word_ids[hash_empty] = 0;  // token '</>' gets word id 0
    }

    void run() {
        bool file_begin = true;
        bool file_end = false;
        static constexpr uint64_t mm_region_size = 1 * essentials::GiB;
        uint64_t blocks = util::ceil_div(m_config.text_size, mm_region_size);
        uint64_t page_size = sysconf(_SC_PAGESIZE);

        m_writer.start();

        for (uint64_t block = 0, begin = 0, end = 0,
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
            end = m_file.size();
            util::optimize_sequential_access(m_data, end);
            if (!is_aligned(end - 1)) align_backward(begin, --end);
            uint64_t n = end - begin;
            assert(n != 0 and n <= mm_region_size);
            m_reader.init(m_data, begin, end, file_begin, file_end);
            m_reader.run();
            file_begin = false;

            // now update end to a window back
            for (uint8_t i = 0; i < m_config.max_order - 1; ++i) {
                end -= 2;  // discards one-past-the-end and whitespace
                align_backward(begin, end);
                assert(is_aligned(end - 1));
            }

            uint64_t num_pages = util::ceil_div(n, page_size);
            assert(num_pages > 0);
            page_id += num_pages - 1;
            begin = offset + end;
            // now begin points to the beginning of 1 window back
            assert(begin >= page_id * page_size);
            begin -= page_id * page_size;
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

    void align_backward(uint64_t begin, uint64_t& end) {
        for (; begin != end; --end) {
            auto c = m_data[end];
            if (c == ' ' or c == '\n') {
                ++end;  // one-past
                break;
            }
        }
    }

    configuration const& m_config;
    boost::iostreams::mapped_file_source m_file;
    uint8_t const* m_data;
    double m_CPU_time;
    double m_I_time;

    typedef counting_writer<BlockWriter> counting_writer_type;
    typedef counting_reader<counting_writer_type> counting_reader_type;
    counting_writer_type m_writer;
    counting_reader_type m_reader;
};

}  // namespace tongrams
