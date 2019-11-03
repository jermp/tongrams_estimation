#pragma once

#include "util.hpp"

namespace tongrams {

struct lines_iterator {
    lines_iterator(const char* filename) : m_begin(0), m_end(0) {
        m_file.open(filename);
        util::check_file(m_file);
        m_data = (uint8_t const*)m_file.data();
        m_size = m_file.size() / sizeof(m_data[0]);
        util::optimize_sequential_access(m_data, m_size);
        this->operator++();  // seek first end of line
    }

    void operator++() {
        m_begin = m_end;
        for (; m_end != m_size; ++m_end) {
            if (m_data[m_end] == '\n') {
                ++m_end;  // one past the end
                break;
            }
        }
    }

    auto const& operator*() {
        m_cur_line.first = &m_data[m_begin];
        m_cur_line.second = &m_data[m_end];
        return m_cur_line;
    }

    bool eof() const {
        return m_end == m_size;
    }

private:
    boost::iostreams::mapped_file_source m_file;
    uint8_t const* m_data;
    size_t m_size;
    uint64_t m_begin, m_end;
    byte_range m_cur_line;
};

}  // namespace tongrams
