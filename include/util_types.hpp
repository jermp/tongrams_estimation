#pragma once

#include <deque>
#include <mutex>
#include <thread>
#include <chrono>
#include <vector>
#include <sparsehash/dense_hash_map>

#define BOOST_THREAD_VERSION 4
#define BOOST_THREAD_PROVIDES_EXECUTORS

#include <boost/config.hpp>
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/experimental/parallel/v2/task_region.hpp>

#include "../external/tongrams/include/utils/util_types.hpp"

namespace tongrams {

typedef uint32_t ngram_id;
typedef uint32_t word_id;
typedef uint32_t range_id;
typedef uint32_t occurrence;
typedef uint64_t count_type;
typedef uint64_t iterator;
typedef google::dense_hash_map<uint64_t, word_id> words_map;
typedef std::chrono::high_resolution_clock clock_type;

bool equal_to(word_id const* x, word_id const* y, size_t n) {
    return memcmp(x, y, n) == 0;
}

typedef boost::executors::basic_thread_pool executor_type;
typedef boost::experimental::parallel::v2::task_region_handle_gen<executor_type>
    task_region_handle;
using boost::experimental::parallel::v2::task_region;

struct parallel_executor {
    parallel_executor(
        size_t num_threads = std::thread::hardware_concurrency()) {
        executor.reset(new executor_type(num_threads));
    }
    std::unique_ptr<executor_type> executor;
};

template <typename Iterator>
struct iterator_range {
    iterator_range() {}
    iterator_range(Iterator b, Iterator e) : begin(b), end(e) {}
    Iterator begin, end;
};

template <typename T>
struct adaptor {
    byte_range operator()(T const& x) const {
        const uint8_t* buf = reinterpret_cast<const uint8_t*>(&x);
        return {buf, buf + sizeof(T)};
    }
};

struct filename_generator {
    filename_generator(std::string const& dir_name, std::string const& prefix,
                       std::string const& extension, int seed = -1)
        : m_seed(seed)
        , m_prefix(dir_name + "/.tmp." + prefix)
        , m_extension(extension) {
        next();
    }

    auto const& operator()() {
        return m_cur_filename;
    }

    auto const& prx() {
        return m_prefix;
    }

    auto const& ext() {
        return m_extension;
    }

    auto seed() const {
        return m_seed;
    }

    void next() {
        ++m_seed;
        m_cur_filename = prx() + std::to_string(m_seed) + "." + ext();
    }

private:
    int m_seed;
    std::string m_prefix;
    std::string m_extension;
    std::string m_cur_filename;
};

template <typename T>
struct semi_sync_queue {
    semi_sync_queue() {
        open();
    }

    void close() {
        m_open = false;
    }

    void open() {
        m_open = true;
    }

    void lock() {
        return m_mutex.lock();
    }

    void unlock() {
        return m_mutex.unlock();
    }

    void push(T& val) {
        m_buffer.push_back(std::move(val));
    }

    T& pick() {
        return m_buffer.front();
    }

    void pop() {
        m_buffer.pop_front();
    }

    bool active() const {
        return m_open;
    }

    bool empty() const {
        return m_buffer.empty();
    }

    size_t size() const {
        return m_buffer.size();
    }

    auto begin() {
        return m_buffer.begin();
    }

    auto end() {
        return m_buffer.end();
    }

    void release() {
        std::deque<T>().swap(m_buffer);
    }

private:
    std::mutex m_mutex;
    std::deque<T> m_buffer;
    bool m_open;
};

}  // namespace tongrams
