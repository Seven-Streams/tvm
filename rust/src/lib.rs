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

//! Library facade over the stubgen-generated TIRx/IR bindings so downstream
//! crates (e.g. cpusim-rs) can `use tvm_tirx_bindings::generated::...`.
//! `src/generated/` is the `tvm-ffi-stubgen --target rust` output with a
//! single hand patch (see `ffi_compat`); see README.md for the regeneration
//! recipe.

pub mod analysis;
pub mod ffi_compat;
pub mod generated;
