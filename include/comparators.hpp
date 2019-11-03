#pragma once

namespace tongrams {

    template<typename T>
    int compare_i(T const& x, T const& y, int i) {
        if (x[i] != y[i]) {
            return x[i] < y[i] ? -1 : 1;
        }
        return 0;
    }

    template<typename T>
    struct context_order_comparator_SIMD {
        context_order_comparator_SIMD(uint8_t N) {
            (void)N;
        }

        bool operator()(T const& x, T const& y) const {
            const __m128i xx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&(x.data[0])));
            const __m128i yy = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&(y.data[0])));
            const __m128i result = _mm_cmpeq_epi32(xx, yy);

            // const int mask = _mm_movemask_epi8(result);
            // return mask == 0xffff;

            // const __m128i result = _mm_cmplt_epi32(xx, yy);
            // std::cout << result << std::endl;
            const int mask = ~(_mm_movemask_epi8(result) << 16);
            // std::cout << "mask = " << mask << std::endl;
            if (mask == 0xffff) {
                return x.data[4] < y.data[4];
            }

            int i = 3 - (__builtin_clz(mask)) / 4;
            return x.data[i] < y.data[i];
        }
    };

    template<typename T>
    struct suffix_order_comparator {
        suffix_order_comparator()
        {}

        void init(uint8_t N) {
            m_N = N;
        }

        suffix_order_comparator(uint8_t N) {
            init(N);
        }

        int order() const {
            return m_N;
        }

        void swap(suffix_order_comparator& other) {
            std::swap(m_N, other.m_N);
        }

        bool operator()(T const& x, T const& y) const {
            return compare(x, y) < 0;
        }

        inline int begin() const {
            return m_N - 1;
        }

        inline int end() const { // last valid index, not one-past the end
            return 0;
        }

        inline void next(int& i) const {
            --i;
        }

        inline void advance(int& i, int n) const {
            i -= n;
        }

        // returns the length of lcp(x,y)
        int lcp(T const& x, T const& y) const {
            int l = 0;
            for (int i = begin(); i != end(); next(i)) {
                int cmp = compare_i(x, y, i);
                if (cmp != 0) return l;
                ++l;
            }
            return l;
        }

        // TODO: rewrite in terms on begin and end
        int compare(T const& x, T const& y) const {
            for (int i = int(begin()); i != -1; --i) {
                int cmp = compare_i(x, y, i);
                if (cmp != 0) return cmp;
            }
            return 0;
        }

        bool equals(T const& x, T const& y) const {
            return compare(x, y) == 0;
        }

    private:
        int m_N;
    };

    template<typename T>
    struct prefix_order_comparator {
        prefix_order_comparator()
        {}

        void init(uint8_t N) {
            m_N = N;
        }

        prefix_order_comparator(uint8_t N) {
            init(N);
        }

        int order() const {
            return m_N;
        }

        void swap(prefix_order_comparator& other) {
            std::swap(m_N, other.m_N);
        }

        bool operator()(T const& x, T const& y) const {
            return compare(x, y) < 0;
        }

        inline int begin() const {
            return 0;
        }

        inline int end() const { // last valid index, not one-past the end
            return m_N - 1;
        }

        inline void next(int& i) const {
            ++i;
        }

        inline void advance(int& i, int n) const {
            i += n;
        }

        // returns the length of lcp(x,y)
        int lcp(T const& x, T const& y) const {
            for (int i = begin(); i != end(); next(i)) {
                int cmp = compare_i(x, y, i);
                if (cmp != 0) return i;
            }
            return m_N;
        }

        // TODO: rewrite in terms on begin and end
        int compare(T const& x, T const& y) const {
            for (int i = begin(); i < m_N; ++i) {
                int cmp = compare_i(x, y, i);
                if (cmp != 0) return cmp;
            }
            return 0;
        }

        bool equals(T const& x, T const& y) const {
            return compare(x, y) == 0;
        }

    private:
        int m_N;
    };

    template<typename T>
    struct context_order_comparator {
        context_order_comparator()
        {}

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

        inline int end() const { // last valid index, not one-past the end
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
            i -= n; // i -= n % m_N to fall back
            if (i < 0) {
                i += m_N;
            }
        }

        // returns the length of lcp(x,y)
        int lcp(T const& x, T const& y) const {
            int l = 0;
            for (int i = begin(); i != end(); next(i)) {
                int cmp = compare_i(x, y, i);
                if (cmp != 0) return l;
                ++l;
            }
            return l;
        }

        // TODO: rewrite in terms on begin and end
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
}
