Tongrams Estimation
--------

Modified [Kneser-Ney](https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing) language model estimation powered by [tongrams](https://github.com/jermp/tongrams).

This C++ library implements the 1-Sort algorithm described in the paper
[*Handling Massive N-Gram Datasets Efficiently*](http://pages.di.unipi.it/pibiri/papers/TOIS19.pdf) by Giulio Ermanno Pibiri and Rossano Venturini, published in ACM TOIS, 2019 [1].

### Compiling the code

	git clone --recursive https://github.com/jermp/tongrams_estimation.git
	mkdir -p build; cd build
	cmake ..
	make -j

### Sample usage

    ./estimate ../test_data/1Billion.1M 5 --tmp tmp_dir --ram 0.25 --out out.bin
    ./external/tongrams/score out.bin ../test_data/1Billion.1M

    ./count ../test_data/1Billion.1M 3 --tmp tmp_dir --ram 0.25 --out 3-grams

### External dependencies

1. [boost](https://www.boost.org/)
2. [sparsehash](https://github.com/sparsehash/sparsehash)
3. [tongrams](https://github.com/jermp/tongrams)

### Bibliography

[1] Pibiri, Giulio Ermanno, and Rossano Venturini. "Handling Massive N-Gram Datasets Efficiently." ACM Transactions on Information Systems (TOIS) 37.2 (2019): 1-41.