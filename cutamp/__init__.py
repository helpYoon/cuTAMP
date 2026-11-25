# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch

from cutamp._curobo_internals import install_curobo_patches

install_curobo_patches()

# A stepped plan's put-back solves batched 18 by 18 systems inside a recorded GPU graph (cutamp/held.py). By default
# this torch (2.9.1) hands batched LU factorizations above 16 by 16 to MAGMA, which a recording cannot hold; with this
# preference they take cuBLAS's batched routine, as the smaller ones already do. Set here, for the whole process and
# whether or not graphs are on, so that every run solves with the same routine.
torch.backends.cuda.preferred_linalg_library("cusolver")
