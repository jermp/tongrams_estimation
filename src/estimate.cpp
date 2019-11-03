#include <chrono>
#include <iostream>

#include "configuration.hpp"
#include "estimation.hpp"

int main(int argc, char** argv) {
    using namespace tongrams;
    configuration config;

    if (argc < 3) {
        std::cerr << "Usage " << argv[0] << ":\n"
                  << "\t text_filename"
                  << "\n \t order\n"
                  << "\t[--out output_filename]\n"
                  << "\t[--tmp tmp_dir]\n"
                  << "\t[--ram GB]\n"
                  << "\t[--thr num_threads]\n"
                  << "\t[--compress_blocks]" << std::endl;
        return 1;
    }

    config.page_size = sysconf(_SC_PAGESIZE);
    size_t available_ram = config.page_size * sysconf(_SC_PHYS_PAGES);
    config.RAM = available_ram;

    config.text_filename = std::string(argv[1]);
    if (!util::exists(config.text_filename.c_str())) {
        std::cerr << "Error: corpus file does not exist" << std::endl;
        return 1;
    }

    config.text_size = util::file_size(config.text_filename.c_str());
    std::cerr << "reading from " << config.text_filename << " "
              << config.text_size << " bytes" << std::endl;
    config.max_order = std::stoull(argv[2]);
    building_util::check_order(config.max_order);

    for (int i = 3; i < argc; ++i) {
        if (argv[i] == std::string("--ram")) {
            uint64_t ram =
                static_cast<uint64_t>(std::stod(argv[++i]) * essentials::GiB);
            if (ram > config.RAM) {
                std::cerr << "Warning: this machine has "
                          << config.RAM / essentials::GiB << " GiB of RAM."
                          << std::endl;
            } else {
                config.RAM = ram;
            }
        } else if (argv[i] == std::string("--out")) {
            config.output_filename = std::string(argv[++i]);
        } else if (argv[i] == std::string("--tmp")) {
            config.tmp_dirname = std::string(argv[++i]);
        } else if (argv[i] == std::string("--thr")) {
            config.num_threads = std::stoull(argv[++i]);
            if (config.num_threads == 0) {
                std::cerr << "number of threads must be > 0" << std::endl;
                return 1;
            }
        } else if (argv[i] == std::string("--compress_blocks")) {
            config.compress_blocks = true;
        } else {
            std::cerr << "unknown option: '" << argv[i] << "'" << std::endl;
            return 1;
        }
    }

    // TODO: put as configurable parameter
    // config.probs_quantization_bits = 13;
    // config.backoffs_quantization_bits = 13;

    config.text_chunk_size = config.RAM;
    if (config.text_chunk_size % config.page_size) {
        std::cerr
            << "text_chunk_size must be a multiple of the operating system "
               "granularity: "
            << config.page_size << " bytes" << std::endl;
        return 1;
    }

    config.vocab_tmp_subdirname = config.tmp_dirname + "/vocab";
    bool ok = essentials::create_directory(config.tmp_dirname) and
              essentials::create_directory(config.vocab_tmp_subdirname);

    if (not ok) return 1;

    std::cerr << "estimating with " << config.RAM << "/" << available_ram
              << " bytes of RAM"
              << " (" << config.RAM * 100.0 / available_ram << "\%)\n";

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    estimation e(config);
    e.run();
    e.print_stats();

    return 0;
}
