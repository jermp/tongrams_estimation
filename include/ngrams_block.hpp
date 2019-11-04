#pragma once

#include "util_types.hpp"
#include "ngram.hpp"
#include "../external/tongrams/include/utils/util.hpp"
#include "comparators.hpp"

namespace tongrams {

template <typename Value>
struct ngram_pointer {
    inline word_id operator[](size_t i) const {
        return data[i];
    }

    inline Value* value(uint8_t ngram_order, uint64_t i = 0) const {
        return reinterpret_cast<Value*>(data + ngram_order) + i;
    }

    // check if the ngram data is equal to another
    // in the range data[begin, end)
    inline bool equal_to(ngram_pointer<Value> const& other, size_t begin,
                         size_t end) const {
        return memcmp(other.data + begin, this->data + begin,
                      (end - begin) * sizeof(word_id)) == 0;
    }

    void print(uint8_t ngram_order, uint64_t num_values) const {
        for (uint8_t i = 0; i < ngram_order; ++i) {
            std::cerr << data[i] << " ";
        }
        for (uint64_t i = 0; i < num_values; ++i) {
            (value(ngram_order, i))->print();
        }
        std::cerr << "\n";
    }

    word_id* data;
};

typedef ngram_pointer<count_type> ngram_pointer_type;
typedef context_order_comparator<ngram_pointer_type>
    context_order_comparator_type;

struct ngrams_block_statistics {
    word_id max_word_id;
    uint64_t max_count;
};

template <typename Value, typename Memory = bytes_block,
          typename Index = std::vector<ngram_pointer<Value>>>
struct ngrams_block {
    typedef typename Index::iterator iterator;

    struct allocator {
        allocator() : m_offset(0), m_alignment(0) {}

        allocator(uint8_t ngram_order) {
            init(ngram_order);
        }

        void init(uint8_t ngram_order) {
            m_offset = 0;
            m_alignment = ngram::size_of(ngram_order);
        }

        void resize(Memory& memory, uint64_t num_ngrams, uint64_t num_values) {
            memory.resize((m_alignment + Value::size_of() * num_values) *
                          num_ngrams);
        }

        template <typename Iterator>
        void construct(ngram_pointer<Value>& ptr, Iterator begin, Iterator end,
                       Value const& value,
                       uint64_t value_offset)  // offset at which we set value
        {
            uint64_t n = 0;
            for (; begin != end; ++n, ++begin) {
                ptr.data[n] = *begin;
            }
            *(ptr.value(n, value_offset)) = value;
        }

        template <typename Iterator>
        void construct(ngram_pointer<Value>& ptr, Iterator begin, size_t n,
                       Value const& value,
                       uint64_t value_offset)  // offset at which we set value
        {
            for (size_t i = 0; i < n; ++i, ++begin) {
                ptr.data[i] = *begin;
            }
            *(ptr.value(n, value_offset)) = value;
        }

        auto allocate(Memory& memory, uint64_t num_values) {
            assert(m_offset < memory.size());
            ngram_pointer<Value> ptr;
            ptr.data = reinterpret_cast<word_id*>(&memory[m_offset]);
            m_offset += m_alignment + Value::size_of() * num_values;
            return ptr;
        }

        // NOTE: pos is an index, not an offset
        auto allocate(Memory& memory, uint64_t num_values, uint64_t pos) {
            uint64_t offset =
                pos * (m_alignment + Value::size_of() * num_values);
            assert(offset < memory.size());
            ngram_pointer<Value> ptr;
            ptr.data = reinterpret_cast<word_id*>(&memory[offset]);
            return ptr;
        }

        uint64_t allocated() const {
            return m_offset;
        }

        uint8_t order() const {
            return m_alignment / sizeof(word_id);
        }

        void swap(allocator& other) {
            std::swap(m_offset, other.m_offset);
            std::swap(m_alignment, other.m_alignment);
        }

        void print_stats() const {
            std::cerr << "allocator stats:\n";
            std::cerr << "m_offset = " << m_offset << "\n";
            std::cerr << "m_alignment = " << m_alignment << "\n";
            std::cerr << "order() = " << int(order()) << "\n";
        }

    private:
        uint64_t m_offset;
        uint64_t m_alignment;
    };

    ngrams_block() {}

    ngrams_block(uint8_t ngram_order, uint64_t num_values) {
        init(ngram_order, num_values);
    }

    ngrams_block(ngrams_block&& rhs) {
        *this = std::move(rhs);
    }

    void init(uint8_t ngram_order, uint64_t num_values) {
        stats = {0, 0};
        m_memory.resize(0);
        m_allocator.init(ngram_order);
        m_index.resize(0);
        m_record_size = ngrams_block<Value, Memory, Index>::record_size(
            ngram_order, num_values);
    }

    inline ngrams_block& operator=(ngrams_block&& rhs) {
        if (this != &rhs) {
            swap(rhs);
        }
        return *this;
    };

    ngrams_block(ngrams_block const&) {
        assert(false);
    }

    ngrams_block& operator=(ngrams_block const&) {
        assert(false);
        return *this;
    };

    // TODO: change name in size_of
    static size_t record_size(uint8_t ngram_order, uint64_t num_values) {
        return ngram::size_of(ngram_order) + Value::size_of() * num_values;
    }

    void resize_memory(uint64_t num_ngrams, uint64_t num_values) {
        m_allocator.resize(m_memory, num_ngrams, num_values);
        m_record_size =
            ngrams_block<Value, Memory>::record_size(order(), num_values);
    }

    void reserve_index(uint64_t num_ngrams) {
        m_index.reserve(num_ngrams);
    }

    void resize_index(uint64_t num_ngrams) {
        m_index.resize(num_ngrams);
    }

    auto& index() {
        return m_index;
    }

    void release() {
        ngrams_block<Value, Memory>().swap(*this);
    }

    void materialize_index(uint64_t num_ngrams, uint64_t num_values) {
        std::cerr << "m_index size before materializing index: "
                  << m_index.size() << std::endl;
        m_index.clear();
        size_t n = m_memory.size() / m_record_size;
        std::cerr << "reserving index space for " << n << " ngrams"
                  << std::endl;
        m_index.reserve(n);
        assert(m_memory.size());
        for (uint64_t i = 0; i < num_ngrams; ++i) {
            auto ptr = m_allocator.allocate(m_memory, num_values, i);
            push_back(ptr);
        }
        assert(size() == num_ngrams);
    }

    void push_back(ngram_pointer<Value> const& ptr) {
        m_index.push_back(ptr);
    }

    template <typename Iterator>
    void push_back(
        Iterator begin, Iterator end, Value const& value,
        uint64_t num_values = 1)  // NOTE: specify num. of values
                                  // that we want to allocate; if > 1,
                                  // value is copied to the last allocated
    {
        assert(num_values);
        auto ptr = m_allocator.allocate(m_memory, num_values);
        m_allocator.construct(ptr, begin, end, value, num_values - 1);
        push_back(ptr);
    }

    template <typename Iterator>
    void set(uint64_t pos, Iterator begin, Iterator end, Value const& value,
             uint64_t num_values = 1) {
        assert(num_values > 0);
#ifdef LSD_RADIX_SORT
        assert(pos < size());
#endif
        auto ptr = m_allocator.allocate(m_memory, num_values, pos);
        m_allocator.construct(ptr, begin, end, value, num_values - 1);
#ifdef LSD_RADIX_SORT
        m_index[pos] = ptr;
#endif
    }

    template <typename Iterator>
    void set(uint64_t pos, Iterator begin, size_t n, Value const& value,
             uint64_t num_values = 1) {
        assert(num_values > 0);
        auto ptr = m_allocator.allocate(m_memory, num_values, pos);
        m_allocator.construct(ptr, begin, n, value, num_values - 1);
    }

    void print_allocator_stats() const {
        m_allocator.print_stats();
    }

    inline uint64_t allocated_bytes() const {
        return m_allocator.allocated();
    }

    inline uint64_t size() const {
        return m_index.size();
    }

    inline uint8_t order() const {
        return m_allocator.order();
    }

    uint8_t num_values() const {
        return (m_record_size - order() * sizeof(word_id)) / Value::size_of();
    }

    // random access with pointers
    inline auto& operator[](size_t i) {
        assert(i < size());
        return m_index[i];
    }

    // random access with indexes
    inline auto access(size_t i) {
        uint64_t offset = i * m_record_size;
        assert(offset < m_memory.size());
        ngram_pointer<Value> ptr;
        ptr.data = reinterpret_cast<word_id*>(&m_memory[offset]);
        return ptr;
    }

    inline iterator begin() {
        return m_index.begin();
    }

    inline iterator end() {
        return m_index.end();
    }

    inline auto& front() {
        return m_index.front();
    }

    inline auto& back() {
        return m_index.back();
    }

    uint64_t record_size() const {
        return m_record_size;
    }

    bool has_memory() const {
        return not m_memory.empty();
    }

    size_t num_bytes() const {
        return m_memory.size();
    }

    template <typename Comparator, typename Iterator>
    bool is_sorted(Iterator begin, Iterator end) {
        uint8_t N = order();
        Comparator comparator(N);
        auto it = begin;
        auto prev = *it;
        ++it;
        bool ret = true;
        // uint64_t sum = 0;
        for (size_t i = 1; it != end; ++i, ++it) {
            auto curr = *it;
            // curr.print(5,1);
            // sum += (curr.value(N))->value;
            // util::do_not_optimize_away(curr);
            int cmp = comparator.compare(prev, curr);

            if (cmp == 0) {
                std::cerr << "Error at " << i << "/" << size() << ":\n";
                prev.print(order(), 0);
                curr.print(order(), 0);
                std::cerr << "Repeated ngrams" << std::endl;
            }

            if (cmp > 0) {
                std::cerr << "Error at " << i << "/" << size() << ":\n";
                prev.print(order(), 0);
                curr.print(order(), 0);
                std::cerr << std::endl;
                // return false;
                ret = false;
            }
            prev = curr;
        }
        // std::cerr << sum << std::endl;
        return ret;
    }

    auto& memory() {
        return m_memory;
    }

    void steal(ngrams_block<Value, Memory, Index>& other) {
        m_memory.swap(other.m_memory);
        m_index.swap(other.m_index);
    }

    void swap(ngrams_block<Value, Memory, Index>& other) {
        steal(other);
        m_allocator.swap(other.m_allocator);
        std::swap(m_record_size, other.m_record_size);
        std::swap(stats.max_word_id, other.stats.max_word_id);
        std::swap(stats.max_count, other.stats.max_count);
    }

    ngrams_block_statistics stats;

protected:
    Memory m_memory;         // memory
    allocator m_allocator;   // allocator
    Index m_index;           // memory index
    uint64_t m_record_size;  // ngram bytes + values bytes
};

template <typename Value>
struct ngram_cache {
    ngram_cache() : m_empty(true) {}

    typedef ngram_pointer<Value> pointer;

    ngram_cache(uint8_t N, uint8_t M) {
        init(N, M);
    }

    void init(uint8_t N, uint8_t M) {
        m_data.resize(ngrams_block<Value>::record_size(N, M));
        m_empty = true;
    }

    pointer get() {
        pointer ptr;
        ptr.data = reinterpret_cast<word_id*>(m_data.data());
        return ptr;
    }

    void store(pointer const& ptr) {
        std::memcpy(m_data.data(), ptr.data, m_data.size());
        m_empty = false;
    }

    bool empty() const {
        return m_empty;
    }

    void swap(ngram_cache<Value>& other) {
        m_data.swap(other.m_data);
        std::swap(m_empty, other.m_empty);
    }

private:
    bytes_block m_data;
    bool m_empty;
};

}  // namespace tongrams
