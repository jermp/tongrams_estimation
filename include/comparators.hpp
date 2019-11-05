#pragma once

namespace tongrams {

template <typename T>
int compare_i(T const& x, T const& y, int i) {
    if (x[i] != y[i]) {
        return x[i] < y[i] ? -1 : 1;
    }
    return 0;
}

template <typename T>
struct context_order_comparator {
    context_order_comparator() {}

    void init(uint8_t N) {
        m_N = N;
    }

    context_order_comparator(uint8_t N) {
        init(N);
    }

    int order() const {
        return m_N;
    }

    void swap(context_order_comparator& other) {
        std::swap(m_N, other.m_N);
    }

    bool operator()(T const& x, T const& y) const {
        return compare(x, y) < 0;
    }

    inline int begin() const {
        return m_N - 2;
    }

    inline int end() const {  // last valid index, not one-past the end
        return m_N - 1;
    }

    inline void next(int& i) const {
        if (i == 0) {
            i = end();
        } else {
            --i;
        }
    }

    inline void advance(int& i, int n) const {
        assert(n <= m_N);
        i -= n;  // i -= n % m_N to fall back
        if (i < 0) {
            i += m_N;
        }
    }

    int lcp(T const& x, T const& y) const {
        int l = 0;  // length of lcp(x,y)
        for (int i = begin(); i != end(); next(i)) {
            int cmp = compare_i(x, y, i);
            if (cmp != 0) return l;
            ++l;
        }
        return l;
    }

    int compare(T const& x, T const& y) const {
        for (int i = int(begin()); i != -1; --i) {
            int cmp = compare_i(x, y, i);
            if (cmp != 0) return cmp;
        }
        return compare_i(x, y, begin() + 1);
    }

    bool equals(T const& x, T const& y) const {
        return compare(x, y) == 0;
    }

private:
    int m_N;
};

}  // namespace tongrams
