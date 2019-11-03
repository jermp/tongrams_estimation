#include <iostream>
#include <fstream>

#include "util.hpp"
#include "util_types.hpp"
#include "stream2.hpp"

namespace tongrams {

namespace debug {

template <typename StreamGenerator, typename Comparator>
void check_sorted_files(configuration const& config) {
    directory tmp_dir(config.tmp_dirname);
    for (auto const& path : tmp_dir) {
        if (not is_directory(path) and is_regular_file(path) and
            (path.extension() == constants::file_extension::counts or
             path.extension() == constants::file_extension::merged)) {
            std::string filename(path.string());
            std::cerr << "DEBUG: checking file '" << filename << "'..."
                      << std::endl;
            StreamGenerator gen(config.max_order, 1);
            gen.open(filename);
            size_t record_size =
                ngrams_block<payload>::record_size(config.max_order, 1);
            size_t num_bytes = config.RAM / record_size * record_size;
            double time = 0.0;
            while (not gen.eos()) {
                gen.sync_fetch_next_blocks(1, num_bytes);
                auto* block = gen.get();
                auto start = clock_type::now();
                block->materialize_index(1);
                if (not block->template is_sorted<Comparator>(block->begin(),
                                                              block->end())) {
                    std::cerr << "NOT sorted" << std::endl;
                    std::abort();
                } else {
                    std::cerr << "OK" << std::endl;
                }
                auto end = clock_type::now();
                std::chrono::duration<double> elapsed = end - start;
                time += elapsed.count();
                gen.processed(block);
                gen.release_processed_blocks();
            }

            std::cerr << "\tscanning time: " << time << " [sec]" << std::endl;
        }
    }
}

}  // namespace debug
}  // namespace tongrams
