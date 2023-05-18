Tongrams Estimation
===================

Modified [Kneser-Ney](https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing) language model estimation powered by [Tongrams](https://github.com/jermp/tongrams).

This C++ library implements the 1-Sort algorithm described in the paper
[*Handling Massive N-Gram Datasets Efficiently*](http://pages.di.unipi.it/pibiri/papers/TOIS19.pdf) by Giulio Ermanno Pibiri and Rossano Venturini, published in ACM TOIS, 2019 [1].

### Compiling the code

	git clone --recursive https://github.com/jermp/tongrams_estimation.git
	mkdir -p build; cd build
	cmake ..
	make -j

### Sample usage

After installation of dependencies and compilation of the code, you can use
the sample text (first 1M lines from the 1Billion corpus; see the paper for dataset
information) in the directory
`test_data`. The text is gzipped, so it must be first uncompressed.

	cd build
	gunzip ../test_data/1Billion.1M.gz

##### 1. Estimation

Then you can estimate a Kneser-Ney language model of order 5 (using 25% of RAM and whose index is serialized to the file `index.bin`) as follows.

    ./estimate ../test_data/1Billion.1M 5 --tmp tmp_dir --ram 0.25 --out index.bin

##### 2. Computing Perplexity

With the index built and serialized to `index.bin` you can compute
the perplexity score with:

    ./external/tongrams/score index.bin ../test_data/1Billion.1M

##### 3. Counting N-Grams

You can also extract n-gram counts. An example follows below, for 3-grams.

    ./count ../test_data/1Billion.1M 3 --tmp tmp_dir --ram 0.25 --out 3-grams

The output file `3-grams` will list all extracted 3-grams sorted lexicographically
in the following standard format:

	<total_number_of_rows>
	<gram1> <TAB> <count1>
	<gram2> <TAB> <count2>
	<gram3> <TAB> <count3>
	...

where each `<gram>` is a sequence of words separated by a whitespace character.

### Dependencies

1. [boost](https://www.boost.org/)
2. [sparsehash](https://github.com/sparsehash/sparsehash)

### Bibliography

[1] Pibiri, Giulio Ermanno, and Rossano Venturini. "Handling Massive N-Gram Datasets Efficiently." ACM Transactions on Information Systems (TOIS) 37.2 (2019): 1-41.
