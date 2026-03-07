#pragma once

#include "params.h"

namespace smxx::decode::sparse_bf16 {

void run_flash_splitkv_mla_bf16_sparse_kernel(const SparseAttnDecodeParams &params);

}
