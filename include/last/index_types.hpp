#pragma once

#include "../external/tongrams/include/lm_types.hpp"

namespace tongrams {

typedef trie_prob_lm<double_valued_mpht64,           // vocabulary
                     identity_mapper,                // mapper
                     quantized_sequence_collection,  // values
                     compact_vector,                 // ranks
                     pef::uniform_pef_sequence,      // word ids
                     ef_sequence                     // pointers
                     >
    reversed_trie_index;

}  // namespace tongrams
