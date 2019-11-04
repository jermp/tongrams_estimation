#pragma once

#include "typedefs.hpp"

namespace tongrams {

template <typename Iterator>
struct parallel_lsd_radix_sorter {
    parallel_lsd_radix_sorter(
        uint32_t max_digit, uint32_t num_digits,
        uint32_t num_threads = std::thread::hardware_concurrency())
        : m_max_digit(max_digit)
        , m_num_digits(num_digits)
        , m_num_threads(num_threads) {}

    void sort(Iterator begin, Iterator end) const {
        uint32_t first_column_index = m_num_digits;
        for (uint32_t column_index = first_column_index;
             column_index - first_column_index < m_num_digits; ++column_index) {
            uint32_t k = column_index - 1;
            if (column_index > m_num_digits) {
                k -= first_column_index;
            }
            parallel_counting_sort(begin, end, k);
        }
    }

private:
    uint32_t m_max_digit;
    uint32_t m_num_digits;
    uint32_t m_num_threads;

    void parallel_counting_sort(Iterator begin, Iterator end,
                                uint32_t column_index) const {
        std::vector<std::vector<uint32_t>> counts(
            m_num_threads + 1, std::vector<uint32_t>(m_max_digit, 0));
        size_t n = end - begin;
        uint64_t batch_size = n / m_num_threads;
        if (!batch_size) {
            throw std::runtime_error("too many threads");
        }

        parallel_executor p(m_num_threads);
        task_region(*(p.executor), [&](task_region_handle& trh) {
            for (uint64_t i = 0; i < m_num_threads; ++i) {
                trh.run([&, i] {
                    auto b = begin + i * batch_size;
                    auto e = b + batch_size;
                    if (i == m_num_threads - 1) {
                        e = end;
                    }
                    std::for_each(b, e, [&](auto const& x) {
                        uint32_t id = x[column_index];
                        assert(id < m_max_digit);
                        ++counts[i + 1][id];
                    });
                });
            }
        });

        // prefix sum
        for (uint32_t j = 0, sum = 0; j < m_max_digit; ++j) {
            for (uint32_t i = 0; i < m_num_threads + 1; ++i) {
                uint32_t occ = counts[i][j];
                counts[i][j] = sum;
                sum += occ;
            }
        }

        // for (auto const& positions: counts) {
        //     for (auto pos: positions) {
        //         std::cerr << pos << " ";
        //     }
        //     std::cerr << std::endl;
        // }

        std::vector<ngram_pointer_type> tmp_index(n);
        task_region(*(p.executor), [&](task_region_handle& trh) {
            for (uint64_t i = 0; i < m_num_threads; ++i) {
                trh.run([&, i] {
                    auto b = begin + i * batch_size;
                    auto e = b + batch_size;
                    if (i == m_num_threads - 1) {
                        e = end;
                    }
                    auto& partition_counts = counts[i + 1];
                    std::for_each(b, e, [&](auto const& x) {
                        uint32_t id = x[column_index];
                        assert(id < m_max_digit);
                        tmp_index[partition_counts[id]++] = x;
                    });
                });
            }
        });

        task_region(*(p.executor), [&](task_region_handle& trh) {
            for (uint64_t i = 0; i < m_num_threads; ++i) {
                trh.run([&, i] {
                    auto b = tmp_index.begin() + i * batch_size;
                    auto output = begin + i * batch_size;
                    auto e = b + batch_size;
                    if (i == m_num_threads - 1) {
                        e = tmp_index.end();
                    }
                    std::for_each(b, e, [&](auto const& x) {
                        *output = x;
                        ++output;
                    });
                });
            }
        });
    }
};
}  // namespace tongrams
