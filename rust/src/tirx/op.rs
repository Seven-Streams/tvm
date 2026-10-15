/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

//! Handles to the `tirx` operators the passes use, as C++ has `tirx::alloc_tensor_op()` and
//! the like. An operator of a nested namespace is named by its path: `tirx.cuda.decl_tmem` is
//! `CUDA_DECL_TMEM`.

use crate::ir::OpHandle;

/// `pub static NAME: OpHandle = OpHandle::registered("ns.name");` for each operator.
macro_rules! registered {
    ($($handle:ident = $name:literal,)*) => {
        $(
            #[doc = concat!("`", $name, "`.")]
            pub static $handle: OpHandle = OpHandle::registered($name);
        )*
    };
}

registered! {
    ABI_FIELD_GET = "tirx.abi_field_get",
    ABI_FIELD_SET = "tirx.abi_field_set",
    ADDRESS_OF = "tirx.address_of",
    ALLOC_TENSOR = "tirx.alloc_tensor",
    ALLOC_WORKSPACE = "tirx.alloc_workspace",
    ASSUME_ALIGNED = "tirx.assume_aligned",
    CALL_CPACKED = "tirx.call_cpacked",
    CALL_CPACKED_LOWERED = "tirx.call_cpacked_lowered",
    CALL_EXTERN = "tirx.call_extern",
    CALL_FFI_KERNEL = "tirx.call_ffi_kernel",
    CALL_LLVM_PURE_INTRIN = "tirx.call_llvm_pure_intrin",
    CALL_PACKED = "tirx.call_packed",
    CALL_PACKED_LOWERED = "tirx.call_packed_lowered",
    COMPUTE_SCOPE = "tirx.compute_scope",
    CUDA_BLOCK_DIM = "tirx.cuda.block_dim",
    CUDA_BLOCK_IDX = "tirx.cuda.block_idx",
    CUDA_DECL_TMEM = "tirx.cuda.decl_tmem",
    CUDA_ELECT_SYNC = "tirx.cuda.elect_sync",
    CUDA_GRID_DIM = "tirx.cuda.grid_dim",
    CUDA_MMA_FILL = "tirx.cuda.mma_fill",
    CUDA_MMA_STORE = "tirx.cuda.mma_store",
    CUDA_MOV_SREG = "tirx.cuda.mov_sreg",
    CUDA_TENSORMAP_ENCODE_TILED = "tirx.cuda.tensormap_encode_tiled",
    CUDA_THREAD_IDX = "tirx.cuda.thread_idx",
    DECL_TENSOR = "tirx.decl_tensor",
    DEVICE_CONTEXT = "tirx.device_context",
    DEVICE_ENTRY = "tirx.device_entry",
    DEVICE_SCOPE = "tirx.device_scope",
    FREE_WORKSPACE = "tirx.free_workspace",
    GET_ACTIVE_LANE_MASK = "tirx.get_active_lane_mask",
    GPU_ACTIVE_THREAD_SELECTOR = "tirx.gpu_active_thread_selector",
    GPU_STORAGE_SYNC = "tirx.gpu_storage_sync",
    GPU_THREAD_ALLREDUCE = "tirx.gpu_thread_allreduce",
    GPU_THREAD_FILTER = "tirx.gpu_thread_filter",
    GPU_WARP_ACTIVEMASK = "tirx.gpu_warp_activemask",
    GPU_WARP_SHUFFLE = "tirx.gpu_warp_shuffle",
    GPU_WARP_SHUFFLE_DOWN = "tirx.gpu_warp_shuffle_down",
    HANDLE_ADD_BYTE_OFFSET = "tirx.handle_add_byte_offset",
    ISNULLPTR = "tirx.isnullptr",
    KERNEL_REPLACE_POINT = "tirx.kernel_replace_point",
    LAUNCH_THREAD = "tirx.launch_thread",
    MASKED_LOAD = "tirx.masked_load",
    MASKED_STORE = "tirx.masked_store",
    OPENCL_ND_MEM_ALLOC_WITH_SCOPE = "tirx.opencl.nd_mem_alloc_with_scope",
    OPENCL_TEXTURE2D_LOAD = "tirx.opencl.texture2d_load",
    OPENCL_TEXTURE2D_STORE = "tirx.opencl.texture2d_store",
    PARALLEL_LAUNCH = "tirx.parallel_launch",
    PTR_BYTE_OFFSET = "tirx.ptr_byte_offset",
    PTX_LEGACY_LDMATRIX = "tirx.ptx_legacy.ldmatrix",
    PTX_LEGACY_MMA = "tirx.ptx_legacy.mma",
    REINTERPRET = "tirx.reinterpret",
    STACK_ALLOCA = "tirx.stack_alloca",
    STACK_MAKE_DLTENSOR = "tirx.stack_make_dltensor",
    STACK_MAKE_SHAPE = "tirx.stack_make_shape",
    TENSOR_DATA_PTR = "tirx.tensor_data_ptr",
    THROW_LAST_ERROR = "tirx.throw_last_error",
}
