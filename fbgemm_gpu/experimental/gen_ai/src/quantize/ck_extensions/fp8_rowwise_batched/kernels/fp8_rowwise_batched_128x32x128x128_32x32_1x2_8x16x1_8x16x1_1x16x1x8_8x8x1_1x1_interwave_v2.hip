/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_batched_common.h"

at::Tensor
fp8_rowwise_batched_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  // A kernel that seems to work well on mid sized tensors.

  // Check if this input needs to be padded.
  int M = XQ.size(1);
  int N = WQ.size(1);
  int K = WQ.size(2);
  bool pad = (K % 128 != 0);

  // Dispatch based on whether padding is needed or not.
  if (pad) {
    using DeviceGemmInstance = DeviceGemmHelper<
        128,
        32,
        128,
        128,
        32,
        32,
        1,
        2,
        S<8, 16, 1>,
        S<8, 16, 1>,
        S<1, 16, 1, 8>,
        S<8, 8, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Interwave,
        ck::BlockGemmPipelineVersion::v2,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
    // Run kernel instance.
    return f8f8bf16_rowwise_batched_impl<DeviceGemmInstance>(
        XQ, WQ, x_scale, w_scale, Y);
  } else {
    using DeviceGemmInstance = DeviceGemmHelper<
        128,
        32,
        128,
        128,
        32,
        32,
        1,
        2,
        S<8, 16, 1>,
        S<8, 16, 1>,
        S<1, 16, 1, 8>,
        S<8, 8, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Interwave,
        ck::BlockGemmPipelineVersion::v2,
        ck::tensor_operation::device::GemmSpecialization::Default>;
    // Run kernel instance.
    return f8f8bf16_rowwise_batched_impl<DeviceGemmInstance>(
        XQ, WQ, x_scale, w_scale, Y);
  }
}
