#pragma once

#include "counting_common.hpp"
#include "configuration.hpp"
#include "tmp.hpp"
#include "sliding_window.hpp"

namespace tongrams {

template <typename Writer>
struct counting_reader {
    counting_reader(configuration const& config, tmp::data& tmp_data,
                    Writer& thread)
        : m_tmp_data(tmp_data)
        , m_window(config.max_order)
        , m_max_order(config.max_order)
        , m_writer(thread)
        , m_next_word_id(constants::empty_token_word_id + 1)
        , m_CPU_time(0.0) {
        m_window.fill(constants::empty_token_word_id);
        static constexpr double weight = 0.9;
        size_t bytes_per_ngram = sizeof_ngram(config.max_order) +
                                 sizeof(count_type) +  // payload
                                 sizeof(word_id*) +    // pointer
                                 sizeof(ngram_id);     // hashset
        m_num_ngrams_per_block = ((weight * config.RAM) /
                                  (2 * hash_utils::probing_space_multiplier)) /
                                 bytes_per_ngram;
    }

    void init(uint8_t const* data, std::string const& boundary,
              uint64_t partition_begin, uint64_t partition_end, bool file_begin,
              bool file_end) {
        auto s = clock_type::now();
        m_partition_end = partition_end;
        m_file_begin = file_begin;
        m_file_end = file_end;
        assert(partition_begin <= partition_end);
        m_counts.init(m_max_order, m_num_ngrams_per_block);
        if (file_begin) count();  // count empty window
        m_window.init({data + partition_begin, data + m_partition_end},
                      partition_begin);

        if (!boundary.empty()) {
            m_window.shift();
            stl_string_adaptor adaptor;
            byte_range range = adaptor(boundary);
            uint64_t hash = hash_utils::byte_range_hash64(range);
            auto id = find_or_insert(range, hash);
            m_window.eat(id);
            count();
        }

        auto e = clock_type::now();
        std::chrono::duration<double> diff = e - s;
        m_CPU_time += diff.count();
    }

    void print_stats() const {
        std::cerr << "\treader thread stats:\n";
        std::cerr << "\tCPU time: " << m_CPU_time << " [sec]\n";
        std::cerr << "\tI time: " << m_window.time() << " [sec]" << std::endl;
    }

    void run() {
        auto s = clock_type::now();
        while (advance()) count();

        // NOTE: if we are at the end of file,
        // add [m_max_order - 1] ngrams padded with empty tokens,
        // i.e., for max_order = 5 and m text words:
        // w_{m-3} w_{m-2} w_{m-1} w_m </>
        // w_{m-2} w_{m-1} w_m </> </>
        // w_{m-1} w_m </> </> </>
        // w_m </> </> </> </>
        if (m_file_end) {
            assert(m_max_order > 0);
            for (uint8_t i = 0; i != m_max_order - 1; ++i) {
                m_window.shift();
                m_window.eat(constants::empty_token_word_id);
                count();
            }
        }

        push_block();

        auto e = clock_type::now();
        std::chrono::duration<double> diff = e - s;
        m_CPU_time += diff.count();
        m_CPU_time -= I_time();
    }

    double CPU_time() const {
        return m_CPU_time;
    }

    double I_time() const {
        return m_window.time();
    }

private:
    tmp::data& m_tmp_data;
    sliding_window m_window;
    uint8_t m_max_order;
    Writer& m_writer;
    word_id m_next_word_id;
    double m_CPU_time;

    uint64_t m_partition_end;
    uint64_t m_num_ngrams_per_block;
    bool m_file_begin, m_file_end;
    counting_step::block_type m_counts;

    word_id find_or_insert(byte_range range, uint64_t hash) {
        word_id id = m_next_word_id;
        auto it = m_tmp_data.word_ids.find(hash);
        if (it == m_tmp_data.word_ids.end()) {
            m_tmp_data.word_ids[hash] = m_next_word_id;
            m_tmp_data.vocab_builder.push_back(range);
            ++m_next_word_id;
        } else {
            id = (*it).second;
        }
        assert(id < m_next_word_id);
        return id;
    }

    bool advance() {
        if (!m_window.advance()) return false;
        auto const& word = m_window.last();
        assert(word.hash != constants::invalid_hash);
        auto id = find_or_insert(word.range, word.hash);
        assert(id < m_next_word_id);
        m_window.eat(id);
        return true;
    }

    void count() {
        uint64_t hash =
            hash_utils::hash64(m_window.data(), sizeof_ngram(m_max_order));
        auto [found, at] = m_counts.find_or_insert(m_window.get(), hash);
        if (found) {
            auto count = ++m_counts[at];
            auto& max_count = m_counts.statistics().max_count;
            if (count > max_count) max_count = count;
        }
        if (m_counts.size() == m_num_ngrams_per_block) push_block();
    }

    void push_block() {
        while (m_writer.size() > 0)
            ;  // wait for flush
        counting_step::block_type tmp;
        tmp.init(m_max_order, m_num_ngrams_per_block);
        tmp.swap(m_counts);
        tmp.release_hash_index();
        m_writer.push(tmp);
    }
};

}  // namespace tongrams