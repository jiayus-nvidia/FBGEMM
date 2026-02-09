/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
};
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

// Some auxiliary funtions for Philox RNG
struct ull2 {
    unsigned long long x;
    unsigned long long y;
};

__forceinline__ __device__ uint2 mulhilo32(const unsigned int a, const unsigned int b) {
    uint2 *res;
    unsigned long long tmp;
    asm ("mul.wide.u32 %0, %1, %2;\n\t"
          : "=l"(tmp)
          : "r"(a), "r"(b));
    res = (uint2*)(&tmp);
    return *res;
}

__forceinline__ __device__ uint4 philox_single_round(const uint4 ctr, const uint2 key) {
    constexpr unsigned long kPhiloxSA = 0xD2511F53;
    constexpr unsigned long kPhiloxSB = 0xCD9E8D57;
    uint2 res0 = mulhilo32(kPhiloxSA, ctr.x);
    uint2 res1 = mulhilo32(kPhiloxSB, ctr.z);
    uint4 ret = {res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};
    return ret;
}

__forceinline__ __device__ uint4 philox(unsigned long long seed,
                               unsigned long long subsequence,
                               unsigned long long offset) {
    constexpr unsigned long kPhilox10A = 0x9E3779B9;
    constexpr unsigned long kPhilox10B = 0xBB67AE85;
    uint2 key = reinterpret_cast<uint2&>(seed);
    uint4 counter;
    ull2 *tmp = reinterpret_cast<ull2*>(&counter);
    tmp->x = offset;
    tmp->y = subsequence;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        counter = philox_single_round(counter, key);
        key.x += (kPhilox10A);
        key.y += (kPhilox10B);
    }
    uint4 output = philox_single_round(counter, key);
    return output;
}
////////////////////////////////////////////////////////////////////////////////////////////////////



struct Dropout {

    const unsigned long long seed, offset;
    const uint8_t p_dropout_in_uint8_t;

    __forceinline__ __device__ Dropout(const unsigned long long seed, const unsigned long long offset,
                              const uint8_t p_dropout_in_uint8_t,
                              const int bid, const int hid, const int tid, const int nheads)
            : seed(seed)
            , offset(offset + (bid * nheads + hid) * 32 + tid % 32)
            , p_dropout_in_uint8_t(p_dropout_in_uint8_t) {
    }
    
    template <bool encode_dropout_in_sign_bit=false, bool drop_value_neg_inf=false, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_dropout(Tensor<Engine, Layout> &tensor_,
                                         int block_row_start, int block_col_start, int block_row_stride) {
        // convert shape from (4, MMA_M, MMA_N) to (8, MMA_M, MMA_N / 2)
        Tensor tensor = make_tensor(tensor_.data(), convert_layout_acc_dropout(tensor_.layout()));
        using T = typename Engine::value_type;
        auto encode_dropout = [](bool keep, T val) {
            if constexpr (drop_value_neg_inf) {
                return keep ? val : T(-INFINITY);
            } else if constexpr (!encode_dropout_in_sign_bit) {
                return keep ? val : T(0);
            } else {
                T abs_val = val < T(0) ? -val : val;
                return keep ? abs_val : -abs_val;
            }
        };
        static_assert(decltype(size<2>(tensor))::value % 2 == 0);
        const uint16_t p_dropout_8bit_in_uint16_t = uint16_t(p_dropout_in_uint8_t);
        const uint32_t p_dropout_8bit_in_uint32_t = (uint32_t(p_dropout_8bit_in_uint16_t) << 16) | uint32_t(p_dropout_8bit_in_uint16_t);
        auto tidx = threadIdx.x;
        // if (cute::thread0()) { printf("threshold2 = 0x%x\n", p_dropout_8bit_in_uint32_t); }
        #pragma unroll
        for (int m = 0; m < size<1>(tensor); ++m, block_row_start += block_row_stride) {
            uint2 rowcol = make_uint2(block_row_start, block_col_start);
            #pragma unroll
            for (int n = 0; n < size<2>(tensor) / 2; ++n, ++rowcol.y) {
                // if (cute::thread(32, 0)) { printf("m = %d, n = %d, row = %d, col = %d\n", m, n, int(rowcol.x), int(rowcol.y));}
                uint4 random_uint4 = philox(seed, reinterpret_cast<unsigned long long&>(rowcol), offset);
                // if (cute::thread0()) { printf("philox = %u, %d, %d, %d\n", random_uint4.x, random_uint4.y, random_uint4.z, random_uint4.w);}
                uint8_t (&rnd_8)[16] = reinterpret_cast<uint8_t (&)[16]>(random_uint4);
                // Special implementation for 16-bit types: we duplicate the threshold to the
                // low and high 16 bits of a 32-bit value, then use the f16x2 comparison instruction
                // to get a mask. The low 16 bits of the mask will be either 0xffff or 0x0000,
                // and the high 16 bits will be either 0xffff or 0x0000, depending on whether
                // the random value is less than the threshold.
                // We then do a bit-wise AND between the mask and the original value (in 32-bit).
                // We're exploiting the fact that floating point comparison is equivalent to integer
                // comparison, since we're comparing unsigned integers whose top 8-bits are zero.
                if (!encode_dropout_in_sign_bit
                    && (std::is_same<T, cutlass::half_t>::value || std::is_same<T, cutlass::bfloat16_t>::value)) {
                    uint16_t rnd_16[16];
                    #pragma unroll
                    for (int i = 0; i < 16; i++) { rnd_16[i] = uint16_t(rnd_8[i]); }
                    uint32_t (&rnd_32)[8] = reinterpret_cast<uint32_t (&)[8]>(rnd_16);
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        Tensor tensor_uint32 = recast<uint32_t>(tensor(_, m, n * 2 + j));
                        // if (cute::thread0()) { printf("random = 0x%x, 0x%x, 0x%x, 0x%x\n", rnd_32[j * 4 + 0], rnd_32[j * 4 + 1], rnd_32[j * 4 + 2], rnd_32[j * 4 + 3]); }
                        // if (cute::thread0()) { printf("tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3)); }
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            uint32_t mask;
                            asm volatile("set.le.u32.f16x2 %0, %1, %2;\n" : "=r"(mask) : "r"(rnd_32[j * 4 + i]), "r"(p_dropout_8bit_in_uint32_t));
                            tensor_uint32(i) &= mask;
                        }
                        // if (tidx == 4 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("ref tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x, %llu, %llu, %d, %d\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3), seed, offset, block_row_start, block_col_start); }
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        #pragma unroll
                        for (int i = 0; i < 8; i++) {
                            tensor(i, m, n * 2 + j) = encode_dropout(rnd_8[j * 8 + i] <= p_dropout_in_uint8_t, tensor(i, m, n * 2 + j));
                        }
                        Tensor tensor_uint32 = recast<uint32_t>(tensor(_, m, n * 2 + j));
                        // if (tidx == 4 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("kernel tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x, %llu, %llu, %d, %d\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3), seed, offset, block_row_start, block_col_start);}
                    }
                }
                // // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
                // //     printf("n = %d, ph  Philox: %u, %u, %u, %u\n", n, rnd_8.x, rnd_8.y, rnd_8.z, rnd_8.w);
                // // }
            }
        }
    }

    // using global coordinates 
    template <typename TensorAcc, typename CoordTensor, typename T_drop>
    __forceinline__ __device__ void apply_dropout_global(
        TensorAcc &tensor,
        const CoordTensor &coords,
        int m_block, int n_block, int kBlockM, int kBlockN,
        T_drop drop_value) {
        using T = typename TensorAcc::value_type;
        #pragma unroll
        for (int i = 0; i < size<0>(tensor); ++i) {
            #pragma unroll
            for (int m = 0; m < size<1>(tensor); ++m) {
                #pragma unroll
                for (int n = 0; n < size<2>(tensor); ++n) {
                    int global_row = m_block * kBlockM + get<0>(coords(i, m, n));
                    int global_col = n_block * kBlockN + get<1>(coords(i, m, n));
                    uint2 pos = make_uint2(global_row, global_col);
                    uint4 random = philox(seed,
                        reinterpret_cast<unsigned long long&>(pos), offset);
                    uint8_t rnd = reinterpret_cast<uint8_t*>(&random)[0];
                    if (rnd > p_dropout_in_uint8_t) {
                        tensor(i, m, n) = T(drop_value);
                    }
                }
            }
        }
    }

};

} // namespace FLASH_NAMESPACE
