#pragma once

#include <deque>
#include <mutex>
#include <thread>

#define BOOST_THREAD_VERSION 4
#define BOOST_THREAD_PROVIDES_EXECUTORS

#include <boost/config.hpp>
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/experimental/parallel/v2/task_region.hpp>

#include "../external/tongrams/include/utils/util_types.hpp"

namespace tongrams {

struct equal_to {
    bool operator()(word_id const* x, word_id const* y, size_t n) {
        return memcmp(x, y, n) == 0;
        // const __m128i xx = _mm_loadu_si128(reinterpret_cast<const
        // __m128i*>(x)); const __m128i yy =
        // _mm_loadu_si128(reinterpret_cast<const __m128i*>(y)); const __m128i
        // result = _mm_cmpeq_epi32(xx, yy); const int mask =
        // _mm_movemask_epi8(result); if (mask == 0xffff) {
        //     return x[n-1] == y[n-1];
        // }
        // return false;
    }
};

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

template <typename UINT>
struct payload_container {
    payload_container() {}

    payload_container(UINT x) : value(x) {}

    static payload_container invalid() {
        return payload_container(0);
    }

    static payload_container combine_values(payload_container lt,
                                            payload_container rt) {
        return payload_container(lt.value + rt.value);
    }

    static size_t size_of() {
        return sizeof(value);
    }

    void print() const {
        std::cerr << "[" << value << "]";
    }

    UINT value;
};

typedef payload_container<uint64_t> payload;

struct filename_generator {
    filename_generator(std::string const& dir_name, std::string const& prefix,
                       std::string const& extension, int seed = -1)
        : m_seed(seed)
        , m_prefix(dir_name + "/.tmp." + prefix)
        , m_extension(extension) {
        next();
    }

    auto& operator()() {
        return m_cur_filename;
    }

    auto& prx() {
        return m_prefix;
    }

    auto& ext() {
        return m_extension;
    }

    auto seed() const {
        return m_seed;
    }

    void next() {
        ++m_seed;
        m_cur_filename = prx() + std::to_string(m_seed) + ext();
    }

    auto get_filename(int seed) {
        return prx() + std::to_string(seed) + ext();
    }

private:
    int m_seed;
    std::string m_prefix;
    std::string m_extension;
    std::string m_cur_filename;
};

struct directory {
    directory(std::string const& dir) : m_dir_path(dir.c_str()) {}

    typedef boost::filesystem::directory_iterator iterator;  // non-recursive

    struct directory_iterator {
        directory_iterator(iterator const& it) : m_it(it) {}

        void operator++() {
            ++m_it;
        }

        boost::filesystem::path operator*() {
            return m_it->path();
        }

        bool operator!=(directory_iterator const& rhs) const {
            return this->m_it != rhs.m_it;
        }

    private:
        iterator m_it;
    };

    directory_iterator begin() const {
        iterator it(m_dir_path);
        return directory_iterator(it);
    }

    directory_iterator end() const {
        iterator it;
        return directory_iterator(it);
    }

private:
    boost::filesystem::path m_dir_path;
};

template <typename T>
struct semi_sync_queue {
    semi_sync_queue() {
        open();
    }

    inline void close() {
        m_open = false;
    }

    inline void open() {
        m_open = true;
    }

    inline void lock() {
        return m_mutex.lock();
    }

    inline void unlock() {
        return m_mutex.unlock();
    }

    inline void push(T& t) {
        m_buffer.push_back(std::move(t));
    }

    template <typename Iterator>
    inline void bulk_push(Iterator begin, Iterator end) {
        lock();
        for (; begin != end; ++begin) {
            m_buffer.push_back(*begin);
        }
        unlock();
    }

    inline T& pick() {
        return m_buffer.front();
    }

    inline void pop() {
        m_buffer.pop_front();
    }

    inline bool active() const {
        return m_open;
    }

    inline bool empty() const {
        return m_buffer.empty();
    }

    inline size_t size() const {
        return m_buffer.size();
    }

    auto begin() {
        return m_buffer.begin();
    }

    auto end() {
        return m_buffer.end();
    }

    auto& mutex() {
        return m_mutex;
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
