#pragma once

#include "comparators.hpp"
#include "configuration.hpp"
#include "constants.hpp"
#include "iterators.hpp"
#include "ngram.hpp"
#include "sliding_window.hpp"
#include "statistics.hpp"
#include "util.hpp"
#include "util_types.hpp"

#include "../external/tongrams/include/utils/util.hpp"

#include "stream.hpp"

// #include "front_coding.hpp"
#include "front_coding2.hpp"
#include "ngrams_hash_block.hpp"

namespace tongrams {

template <typename BlockWriter>
struct counting {
    counting(configuration const& config, tmp::data& tmp_data, tmp::statistics&,
             statistics&)
        : m_config(config)
        , m_tmp_data(tmp_data)
        , m_stream_size(
              util::ceil_div(config.text_size, config.text_chunk_size))
        , m_text_chunk_size(config.text_chunk_size)
        , m_page(0)
        , m_begin(0)
        , m_end(0)
        , m_file_begin(true)
        , m_file_end(false)
        , m_next_word_id(0)
        , m_CPU_time(0.0)
        , m_I_time(0.0)
        , m_writer(config, tmp_data, constants::file_extension::counts) {
        // std::cerr << "config.text_size = " << config.text_size << std::endl;
        // std::cerr << "config.text_chunk_size = " << config.text_chunk_size <<
        // std::endl; std::cerr << "stream_size = " << m_stream_size <<
        // std::endl;

        // NOTE: useful for vocab_estimated_size
        // m_tmp_data.word_ids.resize();
        // m_tmp_data.vocab_builder.reserve();

        uint64_t hash_empty = hash_utils::hash_empty;
        // NOTE: token '</>' gets word id 0
        m_tmp_data.vocab_builder.push_empty();
        m_tmp_data.word_ids[hash_empty] = m_next_word_id;
        ++m_next_word_id;
    }

    typedef ngrams_hash_block<payload,
                              hash_utils::linear_prober  // linear_prober
                                                         // quadratic_prober
                                                         // double_hash_prober
                              >
        block_type;

    void run() {
        m_writer.start();

        for (size_t pos = 0; pos < m_stream_size; ++pos) {
            uint64_t chunk_size = m_text_chunk_size;
            size_t offset = m_page * m_config.page_size;

            if (offset + chunk_size > m_config.text_size) {
                m_file_end = true;
                chunk_size = m_config.text_size - offset;
            }

            assert(m_config.page_size);

            util::open_file_partition(m_file, m_config.text_filename,
                                      chunk_size, offset, &m_data);
            m_end = m_file.size();
            util::optimize_sequential_access(m_data, m_end);

            if (not is_aligned(m_end - 1)) {
                --m_end;
                align_backward(m_begin, m_end);
            }

            uint64_t n = m_end - m_begin;
            assert(n and n <= m_config.text_chunk_size);

            // n must contain at least m_config.max_order
            // non whitespace characters
            if (n < 2 * m_config.max_order - 1) {
                throw std::runtime_error("too many threads");
            }

            {
                reader w(m_data, m_begin, m_begin + n, m_file_begin, m_file_end,
                         m_next_word_id, m_config, m_tmp_data, m_writer);

                m_file_begin = false;

                // NOTE: this is useless: it should just be w.run()
                // parallel_executor p(1);
                // task_region(*(p.executor), [&](task_region_handle& trh) {
                //     trh.run([&w] {
                //         w.run();
                //     });
                // });

                w.run();

                m_CPU_time += w.CPU_time();
                m_I_time += w.I_time();
            }

            // now update m_end to a window back
            for (uint8_t i = 0; i < m_config.max_order - 1; ++i) {
                m_end -= 2;  // discards one-past-the-end and whitespace
                align_backward(m_begin, m_end);
                assert(is_aligned(m_end - 1));
            }

            size_t num_pages = util::ceil_div(n, m_config.page_size);
            assert(num_pages);
            m_page += num_pages - 1;
            m_begin =
                offset +
                m_end;  // now m_begin points to the beginning of 1 window back
            assert(m_begin >= m_page * m_config.page_size);
            m_begin -= m_page * m_config.page_size;

            m_file.close();
        }

        m_writer.terminate();
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

    struct writer {
        writer(configuration const& config, tmp::data& tmp_data,
               std::string const& file_extension)
            : m_tmp_data(tmp_data)
            , m_f_gen(config.tmp_dirname, "", file_extension)
            , m_O_time(0.0)
            , m_CPU_time(0.0)
            , m_num_flushes(0)
            , m_counts_comparator(config.max_order)
            , m_writer(config.max_order) {
            m_buffer.open();
        }

        ~writer() {
            if (not m_buffer.empty()) {
                std::cerr << "Error: some data still need to be written"
                          << std::endl;
                std::terminate();
            }
        }

        void start() {
            m_thread = std::thread(&writer::run, this);
        }

        void terminate() {
            m_buffer.lock();
            m_buffer.close();
            m_buffer.unlock();

            if (m_thread.joinable()) {
                m_thread.join();
            }

            // std::cerr << "writer thread terminating" << std::endl;
            assert(not m_buffer.active());
            while (not m_buffer.empty()) {
                flush();
            }

            std::cerr << "\twriter thread stats:\n";
            std::cerr << "\tflushed blocks: " << m_num_flushes << "\n";
            std::cerr << "\twrite time: " << m_O_time << "\n";
            // NOTE: this is overlapped with reader CPU time
            std::cerr << "\tCPU time: " << m_CPU_time << "\n";
        }

        void push(block_type& block) {
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
        semi_sync_queue<block_type> m_buffer;
        std::thread m_thread;
        filename_generator m_f_gen;
        double m_O_time;
        double m_CPU_time;
        uint64_t m_num_flushes;

        context_order_comparator_type m_counts_comparator;
        // context_order_comparator_SIMD<ngram_pointer<payload>>
        // m_counts_comparator;

        BlockWriter m_writer;

        bool m_compress_blocks;

        void run() {
            while (m_buffer.active()) {
                flush();
            }
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
            block.sort(m_counts_comparator);
            auto end = clock_type::now();
            std::chrono::duration<double> elapsed = end - start;
            m_CPU_time += elapsed.count();
            std::cerr << "sorting took: " << elapsed.count() << std::endl;

            start = clock_type::now();
            std::string filename = m_f_gen();
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
            m_f_gen.next();
        }
    };

    struct reader {
        reader(uint8_t const* data, uint64_t partition_begin,
               uint64_t partition_end, bool file_begin, bool file_end,
               word_id& next_word_id, configuration const& config,
               tmp::data& tmp_data, writer& thread)

            : m_partition_end(partition_end)
            , m_next_word_id(next_word_id)
            , m_file_begin(file_begin)
            , m_file_end(file_end)

            , m_config(config)
            , m_tmp_data(tmp_data)
            , m_window(config.max_order)

            , m_CPU_time(0.0)
            , m_writer(thread)

            , m_hash_size(0)
            , m_num_bytes(sizeof(word_id) * config.max_order) {
            auto s = clock_type::now();
            // std::cout << "initializing counter thread" << std::endl;
            assert(partition_begin <= partition_end);
            // std::cout << "m_file_begin = " << int(m_file_begin) << ";
            // m_file_end = " << int(m_file_end) << std::endl; std::cout <<
            // "partition_begin = " << partition_begin << "; partition_end = "
            // << partition_end << std::endl;

            m_hash_size =
                0.9 * m_config.RAM / 2 /
                (m_num_bytes + 8  // ngrams + payload
#ifdef LSD_RADIX_SORT
                 + 8  // pointers
#endif
                 + 4 * hash_utils::probing_space_multiplier  // hashset
                );

            // std::cout << "m_count_block_size = " << m_count_block_size <<
            // std::endl;
            m_counts.init(config.max_order, m_hash_size, payload(1));

            m_count_block_size = m_hash_size;

            byte_range text_piece(data + partition_begin,
                                  data + m_partition_end);
            m_window.init(text_piece, partition_begin);

            word_id invalid_word_id = 0;
            m_window.fill(invalid_word_id);

            // init of m_rolling_hasher
            // uint32_t hash_in = hash_utils::murmur_hash32(&invalid_word_id, 4,
            // 0); for (uint8_t n = 0; n < m_config.max_order; ++n) {
            //     m_rolling_hasher.eat(hash_in);
            // }

            // count(); // to push the first text word in the window, i.e.,
            // now window is [</> </> </> </> w1]
            // std::cout << "at the beginning, window is: ";
            // m_window.print();

            // NOTE: if we are at the beginning of file,
            // add [m_config.max_order - 1] ngrams padded with empty tokens,
            // i.e., for max_order = 5:
            // [</> </> </> </> w1]
            // [</> </> </> w1 w2]
            // [</> </> w1 w2 w3]
            // [</> w1 w2 w3 w4],
            // otherwise just advance window by [m_config.max_order - 1]
            for (uint8_t n = 0; n < m_config.max_order - 1; ++n) {
                if (m_file_begin) {
                    count();
                } else {
                    advance();
                }
            }

            // now window is [w1 w2 w3 w4 w5]
            // std::cout << "now window is: ";
            // m_window.print();

            auto e = clock_type::now();
            std::chrono::duration<double> diff = e - s;
            m_CPU_time += diff.count();
        }

        ~reader() {
            assert(m_counts.empty());
            // std::cout << "at the end, window is: ";
            // m_window.print();
            std::cerr << "\treader thread stats:\n";
            std::cerr << "\tCPU time: " << m_CPU_time << " [sec]\n";
            std::cerr << "\tI time: " << m_window.time() << " [sec]"
                      << std::endl;
        }

        void run() {
            auto s = clock_type::now();

            while (m_window.end() < m_partition_end) {
                count();
            }

            // std::cout << "here window is one-word past the end" << std::endl;
            // m_window.print();

            // NOTE: if we are at the end of file,
            // add [m_config.max_order - 1] ngrams padded with empty tokens,
            // i.e., for max_order = 5 and m text words: w_{m-3} w_{m-2} w_{m-1}
            // w_m </> w_{m-2} w_{m-1} w_m </> </> w_{m-1} w_m </> </> </> w_m
            // </> </> </> </>
            if (m_file_end) {
                for (uint64_t i = 0; i < m_config.max_order - 2; ++i) {
                    count();
                }
            }

            while (m_writer.size() > 0)
                ;  // wait for flush

            {
                block_type tmp;
                tmp.init(m_config.max_order, m_hash_size, payload(1));
                tmp.swap(m_counts);
                tmp.release_hash_index();
                m_writer.push(tmp);
            }

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
        uint64_t m_count_block_size;
        uint64_t m_partition_end;
        word_id& m_next_word_id;
        bool m_file_begin, m_file_end;

        configuration const& m_config;
        tmp::data& m_tmp_data;

        block_type m_counts;

        sliding_window m_window;

        double m_CPU_time;
        writer& m_writer;
        // util::filename_generator m_f_gen;
        // context_order_comparator<ngram_pointer<payload>> m_counts_comparator;

        uint64_t m_hash_size;
        size_t m_num_bytes;

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

            uint64_t fingerprint = hash_utils::murmur_hash64(
                &(m_window.get().data[0]), m_num_bytes, 0);

            ngram_id at;
            if (m_counts.find_or_insert(m_window.get(), fingerprint, at)) {
                auto ptr = m_counts[at];
                uint64_t count = ++(ptr.value(m_config.max_order)->value);
                uint64_t& max_count = m_counts.statistics().max_count;
                if (count > max_count) {
                    max_count = count;
                }
            }

            if (m_counts.size() == m_count_block_size) {
                while (m_writer.size() > 0)
                    ;  // wait for flush

                block_type tmp;
                tmp.init(m_config.max_order, m_hash_size, payload(1));
                tmp.swap(m_counts);
                tmp.release_hash_index();
                m_writer.push(tmp);
            }
        }
    };

    configuration const& m_config;
    tmp::data& m_tmp_data;

    uint64_t m_stream_size;
    uint64_t m_text_chunk_size;

    boost::iostreams::mapped_file_source m_file;
    uint8_t const* m_data;
    uint64_t m_page;  // keep the number of the disk page that contains the
                      // current file beginning
    uint64_t m_begin, m_end;
    bool m_file_begin, m_file_end;

    word_id m_next_word_id;

    double m_CPU_time;
    double m_I_time;
    writer m_writer;
};
}  // namespace tongrams

/*
    this version works with LSD_RADIX_SORT
*/
// #pragma once

// #include "../util.hpp"
// #include "../values.hpp"
// #include "../util_types.hpp"
// #include "../ngram.hpp"
// #include "../constants.hpp"
// #include "../configuration.hpp"
// #include "../sliding_window.hpp"
// #include "../comparators.hpp"
// #include "../iterators.hpp"
// #include "../open_addressing.hpp"
// #include "statistics.hpp"

// #include "../../utils/util.hpp"

// #include "stream.hpp"

// #include "front_coding.hpp"
// // #include "front_coding2.hpp"
// #include "ngrams_hash_block.hpp"

// namespace tongrams {

//     template<typename BlockWriter>
//     struct counting {
//         counting(configuration const& config,
//                  tmp::data& tmp_data,
//                  tmp::statistics& /*tmp_stats*/,
//                  statistics& /*stats*/)
//             : m_config(config)
//             , m_tmp_data(tmp_data)
//             , m_stream_size(util::ceil_div(config.text_size,
//             config.text_chunk_size)) ,
//             m_text_chunk_size(config.text_chunk_size) , m_page(0),
//             m_begin(0), m_end(0) , m_file_begin(true), m_file_end(false) ,
//             m_next_word_id(0)

//             , m_f_gen(config.tmp_dirname, "",
//             constants::file_extension::counts) , m_CPU_time(0.0),
//             m_I_time(0.0), m_O_time(0.0)
//             // , m_writer(config, tmp_data,
//             constants::file_extension::counts)
//             // , m_writer(config.max_order)
//             // , m_f_gen(config.tmp_dirname, "",
//             constants::file_extension::counts)
//         {
//             // std::cerr << "config.text_size = " << config.text_size <<
//             std::endl;
//             // std::cerr << "config.text_chunk_size = " <<
//             config.text_chunk_size << std::endl;
//             // std::cerr << "stream_size = " << m_stream_size << std::endl;

//             // NOTE: useful for vocab_estimated_size
//             // m_tmp_data.word_ids.resize();
//             // m_tmp_data.vocab_builder.reserve();

//             uint64_t hash_empty = hash_utils::hash_empty;
//             m_tmp_data.word_ids[hash_empty] = m_next_word_id;
//             ++m_next_word_id;
//         }

//         typedef ngrams_hash_block<
//                                   payload,
//                                   linear_prober // linear_prober
//                                   quadratic_prober double_hash_prober
//                                  > block_type;

//         void run() {

//             // m_writer.start();

//             for (size_t pos = 0; pos < m_stream_size; ++pos)
//             {
//                 uint64_t chunk_size = m_text_chunk_size;
//                 size_t offset = m_page * m_config.page_size;

//                 if (offset + chunk_size > m_config.text_size) {
//                     m_file_end = true;
//                     chunk_size = m_config.text_size - offset;
//                 }

//                 assert(m_config.page_size);

//                 util::open_file_partition(m_file, m_config.text_filename,
//                 chunk_size, offset, &m_data); m_end = m_file.size();
//                 util::optimize_sequential_access(m_data, m_end);

//                 if (not is_aligned(m_end - 1)) {
//                     --m_end;
//                     align_backward(m_begin, m_end);
//                 }

//                 uint64_t n = m_end - m_begin;
//                 assert(n and n <= m_config.text_chunk_size);

//                 // n must be at least [m_config.max_order]
//                 // non whitespace characters
//                 if (n < 2 * m_config.max_order - 1) {
//                     throw std::runtime_error("file too small for current
//                     order");
//                 }

//                 {
//                     reader w(
//                         m_data, m_begin, m_begin + n,
//                         m_file_begin, m_file_end,
//                         m_next_word_id,
//                         m_config, m_tmp_data, m_f_gen
//                     );

//                     m_file_begin = false;

//                     parallel_executor p(1);
//                     task_region(*(p.executor), [&](task_region_handle& trh) {
//                         trh.run([&w] {
//                             w.run();
//                         });
//                     });

//                     m_CPU_time += w.CPU_time();
//                     m_I_time += w.I_time();
//                     m_O_time += w.O_time();
//                 }

//                 // now update m_end to a window back
//                 for (uint8_t i = 0; i < m_config.max_order - 1; ++i) {
//                     m_end -= 2; // discards one-past-the-end and whitespace
//                     align_backward(m_begin, m_end);
//                     assert(is_aligned(m_end - 1));
//                 }

//                 size_t num_pages = util::ceil_div(n, m_config.page_size);
//                 assert(num_pages);
//                 m_page += num_pages - 1;
//                 m_begin = offset + m_end; // now m_begin points to the
//                 beginning of 1 window back assert(m_begin >= m_page *
//                 m_config.page_size); m_begin -= m_page * m_config.page_size;

//                 m_file.close();
//             }

//             m_tmp_data.vocab_builder.finalize(); // push empty token

//             // m_writer.terminate();
//         }

//         void print_stats() const {
//             // std::cout << "\"CPU\":" << m_CPU_time + m_writer.CPU_time() <<
//             ", ";
//             // std::cout << "\"I\":" << m_I_time << ", ";
//             // std::cout << "\"O\":" << m_writer.O_time() << ", ";
//             std::cout << "\"CPU\":" << m_CPU_time << ", ";
//             std::cout << "\"I\":" << m_I_time << ", ";
//             std::cout << "\"O\":" << m_O_time << ", ";
//         }

//     private:

//         bool is_aligned(uint64_t pos) const {
//             return m_data[pos] == ' '
//                 or m_data[pos] == '\n';
//         }

//         void align_backward(uint64_t begin, uint64_t& end) {
//             for (; begin != end; --end) {
//                 auto c = m_data[end];
//                 if (c == ' ' or c == '\n') {
//                     ++end; // one-past
//                     break;
//                 }
//             }
//         }

//         struct reader {
//             reader(uint8_t const* data,
//                    uint64_t partition_begin,
//                    uint64_t partition_end,
//                    bool file_begin,
//                    bool file_end,
//                    word_id& next_word_id,
//                    configuration const& config,
//                    tmp::data& tmp_data,
//                    util::filename_generator& f_gen
//                    )

//                 : m_partition_end(partition_end)
//                 , m_next_word_id(next_word_id)
//                 , m_file_begin(file_begin)
//                 , m_file_end(file_end)

//                 , m_config(config)
//                 , m_tmp_data(tmp_data)
//                 , m_window(config.max_order)

//                 , m_CPU_time(0.0)
//                 , m_SORT_time(0.0)
//                 , m_O_time(0.0)
//                 , m_lookup_time(0.0)
//                 , m_num_lookups(0)

//                 , m_writer(config.max_order)
//                 , m_f_gen(f_gen)

//                 , m_hash_size(0)
//                 , m_num_bytes(sizeof(word_id) * config.max_order)
//             {
//                 auto s = clock_type::now();
//                 // std::cout << "initializing counter thread" << std::endl;
//                 assert(partition_begin <= partition_end);
//                 // std::cout << "m_file_begin = " << int(m_file_begin) << ";
//                 m_file_end = " << int(m_file_end) << std::endl;
//                 // std::cout << "partition_begin = " << partition_begin << ";
//                 partition_end = " << partition_end << std::endl;

//                 m_hash_size = 0.9 * m_config.RAM / 2 / (
//                                                         m_num_bytes + 8 //
//                                                         ngrams + payload
//                                                         + 8             //
//                                                         pointers
//                                                         + 4 *
//                                                         hash_utils::constants::probing_space_multiplier
//                                                         // hashset
//                                                        );

//                 // std::cout << "m_count_block_size = " << m_count_block_size
//                 << std::endl; m_counts.init(config.max_order, m_hash_size,
//                 payload(1));

//                 m_count_block_size = m_hash_size;

//                 byte_range text_piece(data + partition_begin,
//                                       data + m_partition_end);
//                 m_window.init(text_piece, partition_begin);

//                 word_id invalid_word_id = 0;
//                 m_window.fill(invalid_word_id);

//                 // init of m_rolling_hasher
//                 // uint32_t hash_in =
//                 hash_utils::murmur_hash32(&invalid_word_id, 4, 0);
//                 // for (uint8_t n = 0; n < m_config.max_order; ++n) {
//                 //     m_rolling_hasher.eat(hash_in);
//                 // }

//                 // count(); // to push the first text word in the window,
//                 i.e.,
//                            // now window is [</> </> </> </> w1]
//                 // std::cout << "at the beginning, window is: ";
//                 // m_window.print();

//                 // NOTE: if we are at the beginning of file,
//                 // add [m_config.max_order - 1] ngrams padded with empty
//                 tokens, i.e.,
//                 // for max_order = 5:
//                 // [</> </> </> </> w1]
//                 // [</> </> </> w1 w2]
//                 // [</> </> w1 w2 w3]
//                 // [</> w1 w2 w3 w4],
//                 // otherwise just advance window by [m_config.max_order - 1]
//                 for (uint8_t n = 0; n < m_config.max_order - 1; ++n) {
//                     if (m_file_begin) {
//                         count();
//                     } else {
//                         advance();
//                     }
//                 }

//                 // now window is [w1 w2 w3 w4 w5]
//                 // std::cout << "now window is: ";
//                 // m_window.print();

//                 auto e = clock_type::now();
//                 std::chrono::duration<double> diff = e - s;
//                 m_CPU_time += diff.count();
//             }

//             ~reader() {
//                 util::wait(m_handle);
//                 assert(m_counts.empty());
//                 // std::cout << "at the end, window is: ";
//                 // m_window.print();
//                 std::cerr << "\treader thread stats:\n";
//                 std::cerr << "\ttotal CPU time: " << m_CPU_time << "
//                 [sec]\n"; std::cerr << "\tsorting time: " << m_SORT_time << "
//                 [sec]\n"; std::cerr << "\ttotal lookup time: " <<
//                 m_lookup_time << " [sec]\n"; std::cerr << "\tnum. lookups: "
//                 << m_num_lookups << "\n"; std::cerr << "\tavg. lookup time: "
//                 << m_lookup_time / m_num_lookups * 1000000 << " [musec]\n";
//                 std::cerr << "\tI time: " << m_window.time() << " [sec]" <<
//                 std::endl; std::cerr << "\tO time: " << m_O_time << " [sec]"
//                 << std::endl;
//             }

//             std::function<void(void)> flush = [&]() {
//                 m_tmp.statistics().max_word_id = m_tmp_data.word_ids.size();
//                 std::cerr << "sorting block" << std::endl;
//                 auto start = clock_type::now();
//                 m_tmp.sort(m_counts_comparator);
//                 auto end = clock_type::now();
//                 std::chrono::duration<double> elapsed = end - start;
//                 m_SORT_time += elapsed.count();
//                 std::cerr << "sorting took: " << elapsed.count() <<
//                 std::endl; start = clock_type::now();

//                 std::ofstream os(m_f_gen().c_str(), std::ofstream::binary);
//                 m_writer.write_block(os,
//                                      m_tmp.begin(), m_tmp.end(),
//                                      m_tmp.size(), m_tmp.statistics());
//                 os.close();
//                 end = clock_type::now();
//                 elapsed = end - start;
//                 m_O_time += elapsed.count();

//                 m_tmp.release();
//             };

//             void run() {

//                 auto s = clock_type::now();

//                 while (m_window.end() < m_partition_end) {
//                     count();
//                 }

//                 // std::cout << "here window is one-word past the end" <<
//                 std::endl;
//                 // m_window.print();

//                 // NOTE: if we are at the end of file,
//                 // add [m_config.max_order - 1] ngrams padded with empty
//                 tokens, i.e.,
//                 // for max_order = 5 and m text words:
//                 // w_{m-3} w_{m-2} w_{m-1} w_m </>
//                 // w_{m-2} w_{m-1} w_m </> </>
//                 // w_{m-1} w_m </> </> </>
//                 // w_m </> </> </> </>
//                 if (m_file_end) {
//                     for (uint64_t i = 0; i < m_config.max_order - 2; ++i) {
//                         count();
//                     }
//                 }

//                 {
//                     util::wait(m_handle);

//                     // block_type m_tmp;

//                     m_tmp.init(m_config.max_order, m_hash_size, payload(1));
//                     m_tmp.swap(m_counts);
//                     m_tmp.release_hash_index();
//                     // m_writer.push(tmp);

//                     m_handle = util::async_call(flush);
//                     m_f_gen.next();
//                 }

//                 auto e = clock_type::now();
//                 std::chrono::duration<double> diff = e - s;
//                 m_CPU_time += diff.count();
//                 m_CPU_time -= I_time();

//                 // NOTE: do not sum beacuse these are completely overlapped
//                 // m_CPU_time += m_SORT_time;
//             }

//             double CPU_time() const {
//                 return m_CPU_time;
//             }

//             double I_time() const {
//                 return m_window.time();
//             }

//             double O_time() const {
//                 return m_O_time;
//             }

//         private:
//             uint64_t m_count_block_size;
//             uint64_t m_partition_end;
//             word_id& m_next_word_id;
//             bool m_file_begin, m_file_end;

//             configuration const& m_config;
//             tmp::data& m_tmp_data;

//             block_type m_counts;
//             block_type m_tmp;

//             sliding_window m_window;

//             double m_CPU_time;
//             double m_SORT_time;
//             double m_O_time;
//             double m_lookup_time;
//             size_t m_num_lookups;

//             BlockWriter m_writer;
//             util::filename_generator& m_f_gen;
//             std::unique_ptr<std::thread> m_handle;

//             context_order_comparator<ngram_pointer<payload>>
//             m_counts_comparator;

//             uint64_t m_hash_size;
//             size_t m_num_bytes;

//             void advance() {
//                 m_window.advance();
//                 auto const& word = m_window.last();
//                 if (word.hash == constants::invalid_hash) {
//                     m_window.print();
//                 }
//                 assert(word.hash != constants::invalid_hash);
//                 word_id id = m_next_word_id;
//                 auto it = m_tmp_data.word_ids.find(word.hash);
//                 if (it == m_tmp_data.word_ids.end()) {
//                     m_tmp_data.word_ids[word.hash] = m_next_word_id;
//                     ++m_next_word_id;
//                     m_tmp_data.vocab_builder.push_back(word.range);
//                 } else {
//                     id = (*it).second;
//                 }
//                 assert(id < m_next_word_id);
//                 m_window.eat(id);
//             }

//             void count() {

//                 advance();
//                 auto const& ngram = m_window.get();

//                 uint64_t fingerprint =
//                 hash_utils::murmur_hash64(&(ngram.data[0]), m_num_bytes, 0);
//                 // ++m_num_lookups;
//                 // auto start = clock_type::now();
//                 ngram_id at;
//                 // ngram_pointer<payload> at;
//                 if (m_counts.find_or_insert(ngram, fingerprint, at)) {
//                     auto ptr = m_counts[at];
//                     uint64_t count =
//                     ++(ptr.value(m_config.max_order)->value);
//                     // uint64_t count =
//                     ++(at.value(m_config.max_order)->value); uint64_t&
//                     max_count = m_counts.statistics().max_count; if (count >
//                     max_count) {
//                         max_count = count;
//                     }
//                 }

//                 // auto end = clock_type::now();
//                 // std::chrono::duration<double> elapsed = end - start;
//                 // m_lookup_time += elapsed.count();

//                 if (m_counts.size() == m_count_block_size) {

//                     util::wait(m_handle);

//                     m_tmp.init(m_config.max_order, m_hash_size, payload(1));
//                     m_tmp.swap(m_counts);
//                     m_tmp.release_hash_index();
//                     // m_writer.push(tmp);

//                     m_handle = util::async_call(flush);
//                     m_f_gen.next();
//                 }
//             }
//         };

//         configuration const& m_config;
//         tmp::data& m_tmp_data;

//         uint64_t m_stream_size;
//         uint64_t m_text_chunk_size;

//         boost::iostreams::mapped_file_source m_file;
//         uint8_t const* m_data;
//         uint64_t m_page; // keep the number of the disk page that contains
//         the current file beginning uint64_t m_begin, m_end; bool
//         m_file_begin, m_file_end;

//         word_id m_next_word_id;

//         util::filename_generator m_f_gen;
//         double m_CPU_time;
//         double m_I_time;
//         double m_O_time;
//         // writer m_writer;
//         // util::filename_generator m_f_gen;
//     };
// }
