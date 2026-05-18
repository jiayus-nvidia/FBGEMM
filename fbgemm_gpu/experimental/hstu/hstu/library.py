#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.

# pyre-strict

import logging
import os

no_fbgemm_gpu: bool = False
try:
    import fbgemm_gpu  # noqa: F401
except ImportError:
    no_fbgemm_gpu: bool = True

import torch

try:
    # pyre-ignore[21]
    # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
    from fbgemm_gpu import open_source

except Exception:
    open_source: bool = False

if (
    torch.cuda.is_available()
    and torch.version.cuda is not None
    and torch.version.cuda >= "12.4"
):
    # Always load the bundled hstu ops .so so torch.export's register_fake hooks
    # in hstu/hstu_ops_gpu.py can find their target ops (fbgemm::hstu_varlen_fwd_{80,90}),
    # even on Blackwell where the actual sm100 path is served via hstu.hstu_blackwell.
    if open_source or no_fbgemm_gpu:
        torch.ops.load_library(
            os.path.join(os.path.dirname(__file__), "fbgemm_gpu_experimental_hstu.so")
        )
        torch.classes.load_library(
            os.path.join(os.path.dirname(__file__), "fbgemm_gpu_experimental_hstu.so")
        )
    else:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_gpu")

        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu/experimental/hstu/src:hstu_ops_gpu_sm80"
        )

        if torch.cuda.get_device_capability() >= (9, 0):
            torch.ops.load_library(
                "//deeplearning/fbgemm/fbgemm_gpu/experimental/hstu/src:hstu_ops_gpu_sm90"
            )

        if torch.cuda.get_device_capability() >= (12, 0):
            torch.ops.load_library(
                "//deeplearning/fbgemm/fbgemm_gpu/experimental/hstu/src:hstu_ops_gpu_sm120"
            )

else:
    logging.warning("CUDA is not available for FBGEMM HSTU")
