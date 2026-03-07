#pragma once

#include "params.h"

namespace sm90::decode::sparse_bf16 {

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_bf16_sparse_kernel(const SparseAttnDecodeParams &params);

}
