Tongrams Estimation
--------

### Compiling the code

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