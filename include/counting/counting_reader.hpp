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
        , m_next_word_id(1)  // 0 is reserved for token '</>'
        , m_CPU_time(0.0) {
        static constexpr double weight = 0.9;
        size_t bytes_per_ngram = ngram::size_of(config.max_order) +
                                 sizeof(count_type) +  // payload
                                 sizeof(word_id*) +    // pointer
                                 sizeof(ngram_id);     // hashset
        m_num_ngrams_per_block = ((weight * config.RAM) /
                                  (2 * hash_utils::probing_space_multiplier)) /
                                 bytes_per_ngram;
    }

    void init(uint8_t const* data, uint64_t partition_begin,
              uint64_t partition_end, bool file_begin, bool file_end) {
        auto s = clock_type::now();
        m_partition_end = partition_end;
        m_file_begin = file_begin;
        m_file_end = file_end;
        assert(partition_begin <= partition_end);
        m_counts.init(m_max_order, m_num_ngrams_per_block);
        m_window.init({data + partition_begin, data + m_partition_end},
                      partition_begin);
        m_window.fill(0);
        for (uint8_t n = 0; n < m_max_order - 1; ++n) {
            if (m_file_begin) {
                count();
            } else {
                advance();
            }
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

        while (m_window.end() < m_partition_end) {
            count();
        }

        // NOTE: if we are at the end of file,
        // add [m_max_order - 1] ngrams padded with empty tokens,
        // i.e., for max_order = 5 and m text words:
        // w_{m-3} w_{m-2} w_{m-1} w_m </>
        // w_{m-2} w_{m-1} w_m </> </>
        // w_{m-1} w_m </> </> </>
        // w_m </> </> </> </>
        if (m_file_end) {
            for (uint64_t i = 0; i < m_max_order - 2; ++i) {
                count();
            }
        }

        while (m_writer.size() > 0)
            ;  // wait for flush
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

    void advance() {
        m_window.advance();
        auto const& word = m_window.last();
        assert(word.hash != constants::invalid_hash);

        word_id id = m_next_word_id;
        auto it = m_tmp_data.word_ids.find(word.hash);
        if (it == m_tmp_data.word_ids.end()) {
            m_tmp_data.word_ids[word.hash] = m_next_word_id;
            ++m_next_word_id;
            m_tmp_data.vocab_builder.push_back(word.range);
        } else {
            id = (*it).second;
        }

        assert(id < m_next_word_id);
        m_window.eat(id);
    }

    void count() {
        advance();

        uint64_t hash = hash_utils::murmur_hash64(
            m_window.data(), ngram::size_of(m_max_order), 0);
        ngram_id at;
        bool found = m_counts.find_or_insert(m_window.get(), hash, at);
        if (found) {
            auto count = ++m_counts[at];
            auto& max_count = m_counts.statistics().max_count;
            if (count > max_count) max_count = count;
        }

        if (m_counts.size() == m_num_ngrams_per_block) {
            essentials::logger("waiting for flushing...");
            while (m_writer.size() > 0)
                ;  // wait for flush
            essentials::logger("done");
            push_block();
        }
    }

    void push_block() {
        counting_step::block_type tmp;
        tmp.init(m_max_order, m_num_ngrams_per_block);
        tmp.swap(m_counts);
        tmp.release_hash_index();
        m_writer.push(tmp);
    }
};

}  // namespace tongrams