#pragma once

#include "util_types.hpp"

#include <vector>
#include <algorithm>

namespace tongrams {

template <typename Iterator>
struct cursor {
    cursor(Iterator begin, Iterator end, uint64_t i)
        : range(begin, end), index(i) {}

    iterator_range<Iterator> range;
    uint64_t index;
};

template <typename Comparator>
struct cursor_comparator {
    cursor_comparator() {}
    cursor_comparator(uint8_t ngram_order) : m_comparator(ngram_order) {}

    template <typename T>
    bool operator()(cursor<T>& l, cursor<T>& r) {
        return m_comparator.compare(l.range.begin.operator*(),
                                    r.range.begin.operator*()) >= 0;
    }

private:
    Comparator m_comparator;
};

template <typename T, typename Comparator>
struct min_heap {
    min_heap(Comparator comparator) : m_comparator(comparator) {}

    void push(T const& t) {
        m_q.push_back(t);
        std::push_heap(m_q.begin(), m_q.end(), m_comparator);
    }

    T& top() {
        return m_q.front();
    }

    void pop() {
        std::pop_heap(m_q.begin(), m_q.end(), m_comparator);
        m_q.pop_back();
    }

    void heapify() {
        sink(0);
    }

    void clear() {
        m_q.clear();
    }

    bool empty() const {
        return m_q.empty();
    }

    inline uint64_t size() const {
        return m_q.size();
    }

    void print() const {
        for (auto x : m_q) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }

private:
    std::vector<T> m_q;
    Comparator m_comparator;

    void sink(uint64_t pos) {
        assert(pos <= size());
        while (2 * pos + 1 < size()) {
            uint64_t i = 2 * pos + 1;
            if (i + 1 < size() and m_comparator(m_q[i], m_q[i + 1])) ++i;
            if (!m_comparator(m_q[pos], m_q[i])) break;
            std::swap(m_q[pos], m_q[i]);
            pos = i;
        }
    }
};

}  // namespace tongrams
