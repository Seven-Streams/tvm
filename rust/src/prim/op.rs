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

//! Handles to the `prim` operators the passes use, as C++ has `prim::if_then_else_op()` and
//! the like.

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
    CEIL = "prim.ceil",
    CLZ = "prim.clz",
    FMA = "prim.fma",
    IF_THEN_ELSE = "prim.if_then_else",
    LIKELY = "prim.likely",
    LOG2 = "prim.log2",
    POW = "prim.pow",
    VSCALE = "prim.vscale",
}
