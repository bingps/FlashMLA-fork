#include "../splitkv_mla.cuh"

namespace sm90::decode::sparse_bf16 {

template void run_flash_splitkv_mla_bf16_sparse_kernel<ModelType::MODEL1, 128>(const SparseAttnDecodeParams &params);

}
