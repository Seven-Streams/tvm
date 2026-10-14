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

//! Smoke tests for the stubgen-generated `tirx` bindings against `libtvm_compiler`.

use std::cell::Cell;
use std::rc::Rc;

use tvm::ir::{
    AttrsObj, BaseFunc, BaseFuncObj, DictAttrs, Evaluate, Expr, GlobalInfo, GlobalVar, IRModule,
    IntImm, Location, PrimType, SeqStmt, Stmt, Type, TypeObj, Var,
};
use tvm::tirx::cuda::*;
use tvm::tirx::tensor::*;
use tvm::tirx::transform::*;
use tvm::tirx::{self, FunctionPass};
use tvm::transform::{PassInfo, PassInfoObj, PassObj};
use tvm::tvm_ffi::object::ObjectRef;
use tvm::tvm_ffi::{
    Any, AnyView, Array, BigInt, DLDataType, DLDataTypeExt, FieldGetter, Function as FfiFunction,
    Map, ObjectCore, ObjectRefCast, Optional, RValueRef, Result, String as FfiString,
};

mod common;
use common::{assert_type_contracts, global, load_tvm_compiler, runtime_type_info, structural_equal};

#[test]
fn generated_types_match_the_registry() {
    load_tvm_compiler();
    type Object = tvm::tvm_ffi::Object;
    assert_type_contracts!(true;
        tirx::CallFFIKernelAttrObj: AttrsObj, tirx::ExecScopeObj: Object,
        tirx::FunctionObj: BaseFuncObj, tirx::IndexMapObj: Object, tirx::IterObj: Object,
        tirx::TensorMapTypeObj: TypeObj, tirx::TensorTypeObj: TypeObj,
        TCGen05InstrDescriptorAttrsObj: AttrsObj, TCGen05InstrDescriptorBlockScaledAttrsObj: AttrsObj,
        TensorMapEncodeTiledAttrObj: AttrsObj,
        ScopeAttrsObj: AttrsObj, TrnAttrsObj: AttrsObj,
        RemoveNoOpConfigObj: Object, UnrollLoopConfigObj: Object,
        PassInfoObj: Object,
    );
    // Opaque bindings embed only their parent; the native object is larger.
    assert_type_contracts!(false;
        tirx::DispatchContextObj: Object, tirx::FunctionPassObj: PassObj, tirx::LayoutObj: Object,
        tirx::ComposeLayoutObj: tirx::LayoutObj, tirx::TileLayoutObj: tirx::LayoutObj,
        PassObj: Object,
    );
    // `tirx.Axis` derives from the builtin `ffi.Enum`, whose stand-in is private to the module.
    assert_eq!(tirx::AxisObj::TYPE_KEY, "tirx.Axis");
    assert_eq!(runtime_type_info::<tirx::AxisObj>().type_depth, tirx::AxisObj::TYPE_DEPTH);
}

/// Read a reflected field through the C ABI getter, bypassing the Rust layout.
fn getter<N: ObjectCore>(object: &N, field: &str) -> Any {
    FieldGetter::new(N::type_index(), field).unwrap().get_any(object).unwrap()
}

fn type_key<N: ObjectCore>(_: &N) -> &'static str {
    N::TYPE_KEY
}

fn same_value(lhs: AnyView, rhs: AnyView) -> bool {
    let equal = global("ffi.StructuralEqual")
        .call_packed(&[lhs, rhs, AnyView::from(&false), AnyView::from(&false)])
        .unwrap();
    bool::try_from(equal).unwrap()
}

/// Each field read directly agrees with its C ABI getter and, where given, holds the value.
///
/// After `optional;` the fields are the crate's `Optional<T>` cells, read out as `Option<T>`.
macro_rules! assert_fields {
    ($object:expr; $($field:ident $(= $expected:expr)?),* $(,)?) => {{
        let object = &*$object;
        $(assert_fields!(@check object, $field, object.$field $(, $expected)?);)*
    }};
    ($object:expr; optional; $($field:ident = $expected:expr),* $(,)?) => {{
        let object = &*$object;
        $(assert_fields!(@check object, $field, object.$field.get(), $expected);)*
    }};
    (@check $object:ident, $field:ident, $direct:expr $(, $expected:expr)?) => {{
        let native = getter($object, stringify!($field));
        let name = format!("{}.{}", type_key($object), stringify!($field));
        let direct = &$direct;
        assert!(
            same_value(AnyView::from(direct), AnyView::from(&native)),
            "{name}: direct read differs from the getter"
        );
        $(assert!(
            same_value(AnyView::from(&$expected), AnyView::from(&native)),
            "{name}: unexpected value"
        );)?
    }};
}

fn s(text: &str) -> FfiString {
    FfiString::from(text)
}

fn strings(items: &[&str]) -> Array<FfiString> {
    Array::new(items.iter().map(|item| s(item)).collect())
}

#[test]
fn rust_allocated_data_objects_are_read_by_native_getters() {
    load_tvm_compiler();

    let scope = tirx::ExecScope::new(s("cuda"), s("kernel"));
    assert_fields!(scope; backend = s("cuda"), name = s("kernel"));

    let kernel_attrs: Map<FfiString, i64> = [(s("k"), 7i64)].into_iter().collect();
    let attr = tirx::CallFFIKernelAttr::new(
        strings(&["a", "b"]),
        strings(&["c"]),
        3,
        kernel_attrs.clone(),
    );
    assert_fields!(attr;
        launch_params = strings(&["a", "b"]), launch_fields = strings(&["c"]),
        num_kernel_args = 3i64, kernel_attrs = kernel_attrs,
    );

    let descriptor = TCGen05InstrDescriptorAttrs::new(
        s("f32"),
        s("f16"),
        s("bf16"),
        1,
        2,
        3,
        true,
        false,
        4,
        false,
        true,
        true,
        false,
    );
    assert_fields!(descriptor;
        d_dtype = s("f32"), a_dtype = s("f16"), b_dtype = s("bf16"), M = 1i64, N = 2i64, K = 3i64,
        trans_a = true, trans_b = false, n_cta_groups = 4i64, neg_a = false, neg_b = true,
        sat_d = true, is_sparse = false,
    );

    let block_scaled = TCGen05InstrDescriptorBlockScaledAttrs::new(
        s("f32"),
        s("e4m3"),
        s("e5m2"),
        s("ue8"),
        s("ue4"),
        1,
        2,
        3,
        false,
        true,
        4,
        true,
        false,
        true,
    );
    assert_fields!(block_scaled;
        d_dtype = s("f32"), a_dtype = s("e4m3"), b_dtype = s("e5m2"), sfa_dtype = s("ue8"),
        sfb_dtype = s("ue4"), M = 1i64, N = 2i64, K = 3i64, trans_a = false, trans_b = true,
        n_cta_groups = 4i64, neg_a = true, neg_b = false, is_sparse = true,
    );

    let uint8 = DLDataType::try_from_str("uint8").unwrap();
    let tiled = TensorMapEncodeTiledAttr::new(uint8, 1, 2, 3, 4, 5, 6);
    assert_fields!(tiled;
        descriptor_dtype = uint8, rank = 1i64, interleave = 2i64, swizzle = 3i64,
        l2_promotion = 4i64, oob_fill = 5i64, force_cu_dtype = 6i64,
    );

    let scope_attrs = ScopeAttrs::new(s("kernel"));
    assert_fields!(scope_attrs; scope = s("kernel"));

    let axes: Array<i64> = Array::new(vec![0, 2]);
    let trn = TrnAttrs::new(
        s("thread"),
        Optional::some(8i64),
        s("op"),
        s("add"),
        s("x"),
        s("y"),
        axes.clone(),
        true,
        false,
    );
    assert_fields!(trn;
        scope = s("thread"), opcode = s("op"), reduce_op = s("add"), op0 = s("x"), op1 = s("y"),
        axes = axes, negate = true, reverse1 = false,
    );
    assert_fields!(trn; optional; max_inst_size = Some(8i64));

    let remove_no_op = RemoveNoOpConfig::new(5, true);
    assert_fields!(remove_no_op; max_simplification_steps = 5i64, ignore_profiler_call = true);

    let unroll = UnrollLoopConfig::new(1, 2, 3, 4, 5);
    assert_fields!(unroll;
        auto_max_step = 1i64, auto_max_depth = 2i64, auto_max_extent = 3i64,
        explicit_unroll = 4i64, unroll_local_access = 5i64,
    );

    let info = PassInfo::new(2, s("rust.pass"));
    assert_fields!(info; opt_level = 2i64, name = s("rust.pass"));
}

/// The pieces the IR-bearing tests share.
struct Ir {
    loc: Location,
    int32: PrimType,
    x: Var,
    y: Var,
}

fn ir() -> Ir {
    let loc: Location = global("ir.UnknownLoc").call_packed(&[]).unwrap().try_into().unwrap();
    let int32: PrimType = global("ir.PrimType")
        .call_tuple((DLDataType::try_from_str("int32").unwrap(),))
        .unwrap()
        .try_into()
        .unwrap();
    let var = |name: &str| -> Var {
        global("ir.Var")
            .call_tuple((s(name), int32.clone(), loc.clone()))
            .unwrap()
            .try_into()
            .unwrap()
    };
    let (x, y) = (var("x"), var("y"));
    Ir { loc, int32, x, y }
}

fn int_imm(ir: &Ir, value: i64) -> IntImm {
    global("ir.IntImm")
        .call_tuple((ir.int32.dtype, BigInt::from(value), ir.loc.clone()))
        .unwrap()
        .try_into()
        .unwrap()
}

/// `{ evaluate(value) }`, with the statements allocated in Rust.
fn evaluate_body(ir: &Ir, value: i64) -> SeqStmt {
    let stmt = Evaluate::new(ir.loc.clone(), int_imm(ir, value).into());
    SeqStmt::new(ir.loc.clone(), Array::<Stmt>::new(vec![stmt.into()]))
}

fn function(ir: &Ir, body: &SeqStmt) -> tirx::Function {
    let params = Array::new(vec![ir.x.clone(), ir.y.clone()]);
    let attrs = DictAttrs::new(Map::<FfiString, Any>::new());
    global("tirx.Function")
        .call_tuple((params, body.clone(), Type::from(ir.int32.clone()), attrs, ir.loc.clone()))
        .unwrap()
        .try_into()
        .unwrap()
}

#[test]
fn native_ir_objects_agree_with_getters_and_rust_allocation() {
    load_tvm_compiler();
    let ir = ir();

    // tirx.Function: built natively around a Rust-allocated body.
    let body = evaluate_body(&ir, 41);
    let func = function(&ir, &body);
    let params = Array::new(vec![ir.x.clone(), ir.y.clone()]);
    assert_fields!(func;
        params = params, ret_type = Type::from(ir.int32.clone()), body = Some(body.clone()),
    );
    let as_base: BaseFunc = func.clone().into();
    assert_fields!(as_base; attrs);
    let rust = tirx::Function::new(
        func.loc.clone(),
        func.ty.clone(),
        func.attrs.clone(),
        func.params.clone(),
        func.ret_type.clone(),
        func.body.clone(),
    );
    assert!(structural_equal(&rust, &func));
    let other = tirx::Function::new(
        func.loc.clone(),
        func.ty.clone(),
        func.attrs.clone(),
        func.params.clone(),
        func.ret_type.clone(),
        Some(evaluate_body(&ir, 7)),
    );
    assert!(!structural_equal(&other, &func));

    // tirx.TensorType
    let shape = Array::<Expr>::new(vec![int_imm(&ir, 4).into(), int_imm(&ir, 8).into()]);
    let strides = Array::<Expr>::new(vec![]);
    let tensor: tirx::TensorType = global("tirx.TensorType")
        .call_packed(&[
            AnyView::from(&shape),
            AnyView::from(&ir.int32),
            AnyView::from(&s("global")),
            AnyView::from(&strides),
            AnyView::from(&0i64),
            AnyView::from(&64i64),
            AnyView::from(&1i64),
            AnyView::new(),
            AnyView::from(&ir.loc),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    assert_fields!(tensor;
        dtype = ir.int32, scope, shape = shape, strides = strides, elem_offset,
        data_alignment = 64i64, offset_factor = 1i64, layout = None::<tirx::Layout>,
    );
    let rust = tirx::TensorType::new(
        tensor.loc.clone(),
        tensor.dtype.clone(),
        tensor.scope.clone(),
        tensor.shape.clone(),
        tensor.strides.clone(),
        tensor.elem_offset.clone(),
        tensor.data_alignment,
        tensor.offset_factor,
        tensor.layout.clone(),
    );
    assert!(structural_equal(&rust, &tensor));
    let as_type: Type = tensor.clone().into();
    assert!(as_type.clone().try_cast::<tirx::TensorType>().is_ok());
    assert!(as_type.try_cast::<tirx::TensorMapType>().is_err());

    let map_type: tirx::TensorMapType =
        global("tirx.TensorMapType").call_tuple((ir.loc.clone(),)).unwrap().try_into().unwrap();
    assert!(structural_equal(&tirx::TensorMapType::new(ir.loc.clone()), &map_type));

    // tirx.Axis is a registry enum: one instance per name.
    let axis = |name: &str| -> tirx::Axis {
        global("tirx.AxisGet").call_tuple((s(name),)).unwrap().try_into().unwrap()
    };
    let memory = axis("m");
    assert!(std::ptr::eq::<tirx::AxisObj>(&*memory, &*axis("m")));

    // tirx.Iter
    let (extent, stride): (Expr, Expr) = (int_imm(&ir, 4).into(), int_imm(&ir, 1).into());
    let iter: tirx::Iter = global("tirx.Iter")
        .call_tuple((extent.clone(), stride.clone(), memory.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    assert_fields!(iter; extent = extent, stride = stride, axis = memory);
    let rust = tirx::Iter::new(extent.clone(), stride.clone(), memory.clone());
    assert!(structural_equal(&rust, &iter));

    // tirx.TileLayout is opaque: its fields come through the getters.
    let layout: tirx::TileLayout = global("tirx.TileLayout")
        .call_tuple((
            Array::new(vec![iter.clone()]),
            Array::<tirx::Iter>::new(vec![]),
            Map::<tirx::Axis, Expr>::new(),
        ))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(layout.shard().unwrap().len(), 1);
    assert!(structural_equal(&layout.shard().unwrap().get(0).unwrap(), &iter));
    assert_eq!(layout.replica().unwrap().len(), 0);
    assert_eq!(layout.offset().unwrap().len(), 0);
    let as_layout: tirx::Layout = layout.clone().into();
    assert!(as_layout.clone().try_cast::<tirx::TileLayout>().is_ok());
    assert!(as_layout.try_cast::<tirx::ComposeLayout>().is_err());

    // tirx.IndexMap
    let initial = Array::new(vec![ir.x.clone(), ir.y.clone()]);
    let swapped = Array::<Expr>::new(vec![ir.y.clone().into(), ir.x.clone().into()]);
    let index_map: tirx::IndexMap = global("tirx.IndexMap")
        .call_packed(&[AnyView::from(&initial), AnyView::from(&swapped), AnyView::new()])
        .unwrap()
        .try_into()
        .unwrap();
    assert_fields!(index_map;
        initial_indices = initial, final_indices = swapped, inverse_index_map = None::<ObjectRef>,
    );
}

#[test]
fn a_rust_closure_runs_as_a_function_pass() {
    load_tvm_compiler();
    let ir = ir();

    // The pass: rebuild the function with `evaluate(n)` turned into `evaluate(n + 1)`.
    let calls = Rc::new(Cell::new(0));
    let seen = calls.clone();
    let pass_func = FfiFunction::from_packed(move |args: &[AnyView]| -> Result<Any> {
        let func: tirx::Function = RValueRef::<tirx::Function>::try_from(args[0])?.into_inner();
        seen.set(seen.get() + 1);
        let body = func.body.clone().expect("the function has a body");
        let stmt: Evaluate = body.seq.get(0)?.try_cast()?;
        let value: IntImm = stmt.value.clone().try_cast()?;
        let bumped = IntImm::new(
            value.loc.clone(),
            value.ty.clone(),
            BigInt::from(value.value.to_i64().expect("a small constant") + 1),
        );
        let stmt = Evaluate::new(stmt.loc.clone(), bumped.into());
        let body = SeqStmt::new(body.loc.clone(), Array::<Stmt>::new(vec![stmt.into()]));
        let rewritten = tirx::Function::new(
            func.loc.clone(),
            func.ty.clone(),
            func.attrs.clone(),
            func.params.clone(),
            func.ret_type.clone(),
            Some(body),
        );
        Ok(Any::from(rewritten))
    });

    // A Rust-allocated PassInfo names the pass; the native factory wraps the closure.
    let pass: FunctionPass = global("tirx.transform.CreateFunctionPass")
        .call_tuple((pass_func, PassInfo::new(0, s("rust.BumpEvaluate"))))
        .unwrap()
        .try_into()
        .unwrap();
    let info = pass.pass_info().unwrap();
    assert_eq!(info.name.as_str(), "rust.BumpEvaluate");
    assert_eq!(info.opt_level, 0);

    let main: GlobalVar =
        global("ir.GlobalVar").call_tuple((s("main"),)).unwrap().try_into().unwrap();
    let functions: Map<GlobalVar, BaseFunc> =
        [(main.clone(), BaseFunc::from(function(&ir, &evaluate_body(&ir, 41))))]
            .into_iter()
            .collect();
    let module: IRModule = global("ir.IRModule")
        .call_packed(&[
            AnyView::from(&functions),
            AnyView::new(),
            AnyView::from(&Map::<FfiString, Array<GlobalInfo>>::new()),
        ])
        .unwrap()
        .try_into()
        .unwrap();

    let lowered: IRModule =
        global("transform.RunPass").call_tuple((pass, module)).unwrap().try_into().unwrap();
    assert_eq!(calls.get(), 1);
    let result: tirx::Function =
        lowered.functions.get(&main).unwrap().expect("main survives").try_cast().unwrap();
    assert!(structural_equal(&result, &function(&ir, &evaluate_body(&ir, 42))));
    assert!(!structural_equal(&result, &function(&ir, &evaluate_body(&ir, 41))));
}
