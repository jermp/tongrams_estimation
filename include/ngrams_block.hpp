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

    inline Value* value(uint8_t order) const {
        return reinterpret_cast<Value*>(data + order);
    }

    inline bool equal_to(ngram_pointer<Value> const& other, size_t begin,
                         size_t end) const {
        return memcmp(other.data + begin, this->data + begin,
                      (end - begin) * sizeof(word_id)) == 0;
    }

    void print(uint8_t order) const {
        for (uint8_t i = 0; i < order; ++i) {
            std::cerr << data[i] << " ";
        }
        value(order)->print();
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

template <typename Value, typename Index = std::vector<ngram_pointer<Value>>>
struct ngrams_block {
    typedef typename Index::iterator iterator;

    struct allocator {
        allocator() : m_offset(0), m_alignment(0) {}

        allocator(uint8_t order) {
            init(order);
        }

        void init(uint8_t order) {
            m_offset = 0;
            m_alignment = ngram::size_of(order);
        }

        void resize(std::vector<uint8_t>& memory, uint64_t num_ngrams) {
            memory.resize((m_alignment + Value::size_of()) * num_ngrams);
        }

        template <typename Iterator>
        void construct(ngram_pointer<Value>& ptr, Iterator begin, Iterator end,
                       Value value) {
            uint64_t n = 0;
            for (; begin != end; ++n, ++begin) {
                ptr.data[n] = *begin;
            }
            *(ptr.value(n)) = value;
        }

        auto allocate(std::vector<uint8_t>& memory) {
            assert(m_offset < memory.size());
            ngram_pointer<Value> ptr;
            ptr.data = reinterpret_cast<word_id*>(&memory[m_offset]);
            m_offset += m_alignment + Value::size_of();
            return ptr;
        }

        // NOTE: pos is an index, not an offset
        auto allocate(std::vector<uint8_t>& memory, uint64_t pos) {
            uint64_t offset = pos * (m_alignment + Value::size_of());
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

    ngrams_block(uint8_t order) {
        init(order);
    }

    ngrams_block(ngrams_block&& rhs) {
        *this = std::move(rhs);
    }

    void init(uint8_t order) {
        stats = {0, 0};
        m_memory.resize(0);
        m_allocator.init(order);
        m_index.resize(0);
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
    inline static size_t record_size(uint8_t order) {
        return ngram::size_of(order) + Value::size_of();
    }

    inline uint64_t record_size() const {
        return record_size(order());
    }

    void resize_memory(uint64_t num_ngrams) {
        m_allocator.resize(m_memory, num_ngrams);
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
        ngrams_block<Value>().swap(*this);
    }

    void push_back(ngram_pointer<Value> const& ptr) {
        m_index.push_back(ptr);
    }

    template <typename Iterator>
    void push_back(Iterator begin, Iterator end, Value value) {
        auto ptr = m_allocator.allocate(m_memory);
        m_allocator.construct(ptr, begin, end, value);
        push_back(ptr);
    }

    template <typename Iterator>
    void set(uint64_t pos, Iterator begin, Iterator end, Value const& value) {
#ifdef LSD_RADIX_SORT
        assert(pos < size());
#endif
        auto ptr = m_allocator.allocate(m_memory, pos);
        m_allocator.construct(ptr, begin, end, value);
#ifdef LSD_RADIX_SORT
        m_index[pos] = ptr;
#endif
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

    // random access with pointers
    inline auto& operator[](size_t i) {
        assert(i < size());
        return m_index[i];
    }

    // random access with indexes
    inline auto access(size_t i) {
        uint64_t offset = i * record_size();
        assert(offset < m_memory.size());
        ngram_pointer<Value> ptr;
        ptr.data = reinterpret_cast<word_id*>(m_memory.data() + offset);
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
        for (size_t i = 1; it != end; ++i, ++it) {
            auto curr = *it;
            int cmp = comparator.compare(prev, curr);
            if (cmp == 0) {
                std::cerr << "Error at " << i << "/" << size() << ":\n";
                prev.print(N);
                curr.print(N);
                std::cerr << "Repeated ngrams" << std::endl;
            }
            if (cmp > 0) {
                std::cerr << "Error at " << i << "/" << size() << ":\n";
                prev.print(N);
                curr.print(N);
                std::cerr << std::endl;
                ret = false;
            }
            prev = curr;
        }
        return ret;
    }

    auto& memory() {
        return m_memory;
    }

    void steal(ngrams_block<Value, Index>& other) {
        m_memory.swap(other.m_memory);
        m_index.swap(other.m_index);
    }

    void swap(ngrams_block<Value, Index>& other) {
        steal(other);
        m_allocator.swap(other.m_allocator);
        std::swap(stats.max_word_id, other.stats.max_word_id);
        std::swap(stats.max_count, other.stats.max_count);
    }

    ngrams_block_statistics stats;

protected:
    std::vector<uint8_t> m_memory;  // memory
    allocator m_allocator;          // allocator
    Index m_index;                  // memory index
};

template <typename Value>
struct ngram_cache {
    ngram_cache() : m_empty(true) {}

    typedef ngram_pointer<Value> pointer;

    ngram_cache(uint8_t order) {
        init(order);
    }

    void init(uint8_t order) {
        m_data.resize(ngrams_block<Value>::record_size(order));
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
    std::vector<uint8_t> m_data;
    bool m_empty;
};

}  // namespace tongrams
