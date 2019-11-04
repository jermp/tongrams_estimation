#pragma once

#include "ngrams_hash_block.hpp"
#include "hash_utils.hpp"

namespace tongrams {
namespace counting_step {

typedef ngrams_hash_block<payload, hash_utils::linear_prober> block_type;

}  // namespace counting_step
}  // namespace tongrams