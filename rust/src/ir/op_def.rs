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

//! Registering operators from Rust: C++ `OpDef` and its signature descriptors, and the handles
//! that keep operators in `static`s.
//!
//! Every hook an operator gets here is native, as the C++ ones are: a Rust function compiled into
//! the C ABI adapter that C++ `NativeFunctionView` calls through a function pointer, without
//! packing arguments. A signature generates its validator from its argument types, as C++
//! `OpDef::signature` does from `sig::arg<T>`; each signature is its own compiled function.

use std::any::TypeId;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex};
use tvm_ffi::{
    Any, AnyView, Array, Error, ObjectArc, ObjectRefCore, Result, String, INTERNAL_ERROR,
    TYPE_ERROR, VALUE_ERROR,
};

use super::{Attrs, CallObj, Expr, Op, OpObj, RegionStmtObj, SeqCell, Type, Var};
use crate::ops::type_key;
use sig::{ArgType, Checks};

/// C++ `OpDef`: register an operator from Rust, or attach more to a registered one.
///
/// Each step updates the registry at once, as in C++. After a failed step the rest are skipped,
/// and [`OpDef::op`] returns the error. The registry does not synchronize registration with
/// readers, so register an operator before building IR with it. An operator and its hooks live
/// for the rest of the process.
///
/// ```ignore
/// use tvm::ir::sig::IntExpr;
///
/// fn first_arg_type(call: &CallObj) -> Result<Type> {
///     Ok(call.args.get(0)?.ty.clone())
/// }
///
/// let op = OpDef::new("my_ext.clamp", "Clamp an index to a bound.")
///     .signature(Signature::new().arg::<IntExpr>("index", "").arg::<IntExpr>("bound", ""))
///     .set_attr("TCallEffectKind", 0i64)
///     .set_infer_type(first_arg_type)
///     .op()?;
/// ```
pub struct OpDef(Result<Op>);

impl OpDef {
    /// Register a new operator (`ir.RegisterOp`); a name already registered is an error.
    pub fn new(name: &str, doc: &str) -> Self {
        let registered = tvm_ffi::cached_global_func!("ir.RegisterOp")
            .call_tuple((String::from(name), String::from(doc)));
        Self(registered.and_then(|_| Op::get(name)))
    }

    /// Attach more to the registered operator `name`.
    pub fn extend(name: &str) -> Self {
        Self(Op::get(name))
    }

    /// The operator, or the error of the step that failed.
    pub fn op(self) -> Result<Op> {
        self.0
    }

    /// C++ `OpDef::signature`: replace the argument descriptions and the attrs type, and the
    /// validator generated from them, which checks a Call's argument counts, argument types and
    /// attrs. A custom validator keeps precedence. Without [`Signature::call_attrs`] the
    /// operator has no attrs type, and `Call.attrs` is unconstrained.
    pub fn signature<L: Checks>(self, signature: Signature<L>) -> Self {
        let Signature {
            arg_names, arg_docs, ty_arg_names, ty_arg_docs, var_args, var_ty_args, ..
        } = signature;
        let lists =
            [arg_names, arg_docs, ty_arg_names, ty_arg_docs, var_args, var_ty_args].map(Array::new);
        let args: Vec<AnyView> = lists.iter().map(AnyView::from).collect();
        let def = self.invoke("_set_signature", &args).then(|op| {
            if !op.validator_is_custom() {
                write_validator(op, validate_signature::<L>, false);
            }
            Ok(())
        });
        match L::ATTRS_TYPE_KEY {
            Some(key) => def.attrs_type_key(key),
            None => def,
        }
    }

    /// Append a value argument's description, without changing what the validator checks.
    pub fn add_arg(self, name: &str, doc: &str) -> Self {
        let (name, doc) = (String::from(name), String::from(doc));
        self.invoke("add_arg", &[AnyView::from(&name), AnyView::from(&doc)])
    }

    /// Append a type argument's description, without changing what the validator checks.
    pub fn add_ty_arg(self, name: &str, doc: &str) -> Self {
        let (name, doc) = (String::from(name), String::from(doc));
        self.invoke("add_ty_arg", &[AnyView::from(&name), AnyView::from(&doc)])
    }

    /// The type of `Call.attrs`, by its registered type key.
    pub fn attrs_type_key(self, key: &str) -> Self {
        let key = String::from(key);
        self.invoke("set_attrs_type_key", &[AnyView::from(&key)])
    }

    /// Register attribute `key`; a value already registered for this operator is an error.
    pub fn set_attr(self, key: &str, value: impl Into<Any>) -> Self {
        self.attr(key, value.into(), false)
    }

    /// Register attribute `key`, replacing a value already registered for this operator.
    pub fn replace_attr(self, key: &str, value: impl Into<Any>) -> Self {
        self.attr(key, value.into(), true)
    }

    /// Remove this operator's value of attribute `key`.
    pub fn reset_attr(self, key: &str) -> Self {
        let key = String::from(key);
        self.invoke("reset_attr", &[AnyView::from(&key)])
    }

    /// `FInferType`: the result type of a Call that does not give one, which the `Call`
    /// constructor and `Call::ReinferType` ask for; `TFixedReturnType` takes precedence. `infer`
    /// is a function or a closure that captures nothing.
    pub fn set_infer_type<F>(self, infer: F) -> Self
    where
        F: Fn(&CallObj) -> Result<Type> + 'static,
    {
        let _ = infer;
        let hook: Hook = infer_type_abi::<F>;
        self.set_attr("FInferType", hook as *mut c_void)
    }

    /// A custom validator, run by `Call::Validate`, which reports an invalid Call as an error. It
    /// replaces the one a signature generated, and a later signature keeps it; a second custom
    /// validator is an error. `validate` is a function or a closure that captures nothing.
    pub fn set_validator<F>(self, validate: F) -> Self
    where
        F: Fn(&CallObj) -> Result<()> + 'static,
    {
        let _ = validate;
        self.then(|op| {
            if op.validator_is_custom() {
                let message = format!("Validator of {} is already registered", op.name.as_str());
                return Err(Error::new(VALUE_ERROR, &message, ""));
            }
            write_validator(op, validate_abi::<F>, true);
            Ok(())
        })
    }

    /// `FRegionGetBodyParams`, which makes the operator a region operation: the variables a
    /// region binds in its body, from its Call (the operation, operands and attributes), which
    /// the `RegionStmt` constructor checks the body parameters against. A region that binds none
    /// returns none. `body_params` is a function or a closure that captures nothing.
    pub fn set_region_body_params<F>(self, body_params: F) -> Self
    where
        F: Fn(&CallObj) -> Result<Array<Var>> + 'static,
    {
        let _ = body_params;
        let hook: Hook = region_body_params_abi::<F>;
        self.set_attr("FRegionGetBodyParams", hook as *mut c_void)
    }

    /// `FRegionValidate`: a check of a complete region statement, run by the `RegionStmt`
    /// constructor after the body parameters. `validate` is a function or a closure that
    /// captures nothing.
    pub fn set_region_validate<F>(self, validate: F) -> Self
    where
        F: Fn(&RegionStmtObj) -> Result<()> + 'static,
    {
        let _ = validate;
        let hook: RegionHook = region_validate_abi::<F>;
        self.set_attr("FRegionValidate", hook as *mut c_void)
    }

    fn attr(self, key: &str, value: Any, replace: bool) -> Self {
        let key = String::from(key);
        self.invoke(
            "_set_attr",
            &[AnyView::from(&key), AnyView::from(&value), AnyView::from(&replace)],
        )
    }

    /// Call the `ir.Op` method `method` on the operator with `args`.
    fn invoke(self, method: &str, args: &[AnyView]) -> Self {
        self.then(|op| {
            let packed: Vec<AnyView> =
                std::iter::once(AnyView::from(op)).chain(args.iter().copied()).collect();
            tvm_ffi::Function::from_type_key_method("ir.Op", method)?.call_packed(&packed)?;
            Ok(())
        })
    }

    fn then(self, step: impl FnOnce(&Op) -> Result<()>) -> Self {
        Self(self.0.and_then(|op| step(&op).map(|_| op)))
    }
}

/// A handle to an operator, kept in a `static` as C++ keeps one in `const Op& xxx_op()`: either
/// an operator registered elsewhere (C++ or Python), looked up by name, or one Rust defines,
/// registered once, by [`register_ops`] or on first use. The operator is fetched once per thread;
/// [`OpHandle::get`] returns it and [`OpHandle::is`] compares with it.
///
/// ```ignore
/// static LAUNCH_THREAD: OpHandle = OpHandle::registered("tirx.launch_thread");
/// static CLAMP: OpHandle = OpHandle::define("my_ext.clamp", "Clamp an index to a bound.", |def| {
///     def.signature(Signature::new().arg::<IntExpr>("index", "").arg::<IntExpr>("bound", ""))
///         .set_attr("TCallEffectKind", 0i64)
///         .set_infer_type(first_arg_type)
/// });
///
/// if LAUNCH_THREAD.is(&region.op) { ... }
/// let clamped = Call::new(ty, CLAMP.get()?, [index, bound])?;
/// ```
pub struct OpHandle {
    name: &'static str,
    definition: Option<Definition>,
    /// The outcome of the definition, which runs once: the error's text if it failed.
    defined: OnceLock<std::result::Result<(), std::string::String>>,
    /// This handle's slot in each thread's cache of fetched operators.
    slot: OnceLock<usize>,
}

/// The documentation and the definition of an operator Rust defines.
type Definition = (&'static str, fn(OpDef) -> OpDef);

/// The number of slots handed out to handles.
static SLOTS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// The operators this thread fetched, by handle slot.
    static FETCHED: RefCell<Vec<Option<Op>>> = const { RefCell::new(Vec::new()) };
}

impl OpHandle {
    /// The operator `name`, registered elsewhere.
    pub const fn registered(name: &'static str) -> Self {
        Self { name, definition: None, defined: OnceLock::new(), slot: OnceLock::new() }
    }

    /// The operator `name`, defined here: `define` adds to `OpDef::new(name, doc)`. An operator
    /// of that name registered elsewhere is an error, as Rust does not take operators over.
    pub const fn define(name: &'static str, doc: &'static str, define: fn(OpDef) -> OpDef) -> Self {
        let definition = Some((doc, define));
        Self { name, definition, defined: OnceLock::new(), slot: OnceLock::new() }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Register the operator if Rust defines it and it is not registered yet. The caller that
    /// runs a failing definition gets its error; later ones get an error that names it.
    pub fn register(&self) -> Result<()> {
        let Some((doc, define)) = self.definition else { return Ok(()) };
        let mut failure = None;
        let outcome = self.defined.get_or_init(|| match define(OpDef::new(self.name, doc)).op() {
            Ok(_) => Ok(()),
            Err(error) => {
                let text = format!("{}: {}", error.kind(), error.message());
                failure = Some(error);
                Err(text)
            }
        });
        match (failure, outcome) {
            (Some(error), _) => Err(error),
            (None, Ok(())) => Ok(()),
            (None, Err(text)) => {
                let message = format!("Operator {} failed to register: {text}", self.name);
                Err(Error::new(INTERNAL_ERROR, &message, ""))
            }
        }
    }

    /// The operator, registered first if Rust defines it.
    pub fn get(&self) -> Result<Op> {
        self.with(Op::clone)
    }

    /// Whether `value`, such as a Call's `op`, is this operator (C++ `value.same_as(op)`). No
    /// value is an operator that cannot be fetched, such as one this build does not register.
    pub fn is(&self, value: &impl ObjectRefCore) -> bool {
        self.with(|op| value.same_as(op)).unwrap_or(false)
    }

    /// `use_op` applied to the operator, fetched into this thread's slot the first time.
    fn with<R>(&self, use_op: impl FnOnce(&Op) -> R) -> Result<R> {
        self.register()?;
        let slot = *self.slot.get_or_init(|| SLOTS.fetch_add(1, Ordering::Relaxed));
        FETCHED.with(|fetched| {
            if let Some(Some(op)) = fetched.borrow().get(slot) {
                return Ok(use_op(op));
            }
            let op = Op::get(self.name)?;
            let mut fetched = fetched.borrow_mut();
            if fetched.len() <= slot {
                fetched.resize(slot + 1, None);
            }
            Ok(use_op(fetched[slot].insert(op)))
        })
    }
}

/// Register the operators of `ops` that Rust defines and are not registered yet, in order; the
/// first failure is the error. Registering again does nothing, so a library can call it from
/// every entry point.
pub fn register_ops(ops: &[&OpHandle]) -> Result<()> {
    ops.iter().try_for_each(|op| op.register())
}

thread_local! {
    static OPS: RefCell<HashMap<&'static str, Op>> = RefCell::new(HashMap::new());
}

/// `Op::Get(name)`, fetched once per thread, as C++ keeps an operator in a `static`.
pub(crate) fn cached_op(name: &'static str) -> Result<Op> {
    if let Some(op) = OPS.with(|ops| ops.borrow().get(name).cloned()) {
        return Ok(op);
    }
    let op = Op::get(name)?;
    OPS.with(|ops| ops.borrow_mut().insert(name, op.clone()));
    Ok(op)
}

/// The arguments an operator takes, for [`OpDef::signature`]: the fixed value and type arguments
/// in order and an optional variadic tail of each, with their documentation and the types the
/// validator checks (C++ `sig::arg<T>` and friends). `L` lists the checks, newest first.
pub struct Signature<L = ()> {
    arg_names: Vec<String>,
    arg_docs: Vec<String>,
    ty_arg_names: Vec<String>,
    ty_arg_docs: Vec<String>,
    /// Empty, or the tail's name and documentation.
    var_args: Vec<String>,
    var_ty_args: Vec<String>,
    checks: PhantomData<L>,
}

impl Signature {
    pub fn new() -> Self {
        Self {
            arg_names: Vec::new(),
            arg_docs: Vec::new(),
            ty_arg_names: Vec::new(),
            ty_arg_docs: Vec::new(),
            var_args: Vec::new(),
            var_ty_args: Vec::new(),
            checks: PhantomData,
        }
    }
}

impl Default for Signature {
    fn default() -> Self {
        Self::new()
    }
}

impl<L> Signature<L> {
    /// A required value argument of type `T` (C++ `sig::arg<T>`); [`Expr`] accepts any.
    pub fn arg<T: ArgType>(mut self, name: &str, doc: &str) -> Signature<(sig::Arg<T>, L)> {
        self.arg_names.push(String::from(name));
        self.arg_docs.push(String::from(doc));
        self.retype()
    }

    /// Zero or more further value arguments of type `T` (C++ `sig::var_args<T>`).
    pub fn var_args<T: ArgType>(
        mut self,
        name: &str,
        doc: &str,
    ) -> Signature<(sig::VarArgs<T>, L)> {
        self.var_args = vec![String::from(name), String::from(doc)];
        self.retype()
    }

    /// A required type argument of type `T` (C++ `sig::ty_arg<T>`); [`Type`] accepts any.
    pub fn ty_arg<T: ArgType>(mut self, name: &str, doc: &str) -> Signature<(sig::TyArg<T>, L)> {
        self.ty_arg_names.push(String::from(name));
        self.ty_arg_docs.push(String::from(doc));
        self.retype()
    }

    /// Zero or more further type arguments of type `T` (C++ `sig::var_ty_args<T>`).
    pub fn var_ty_args<T: ArgType>(
        mut self,
        name: &str,
        doc: &str,
    ) -> Signature<(sig::VarTyArgs<T>, L)> {
        self.var_ty_args = vec![String::from(name), String::from(doc)];
        self.retype()
    }

    /// `Call.attrs` holds a `T`, which is also the operator's attrs type (C++
    /// `sig::call_attrs<T>`).
    pub fn call_attrs<T>(self) -> Signature<(sig::CallAttrs<T>, L)>
    where
        T: ObjectRefCore + Into<Attrs> + 'static,
    {
        self.retype()
    }

    fn retype<M>(self) -> Signature<M> {
        Signature {
            arg_names: self.arg_names,
            arg_docs: self.arg_docs,
            ty_arg_names: self.ty_arg_names,
            ty_arg_docs: self.ty_arg_docs,
            var_args: self.var_args,
            var_ty_args: self.var_ty_args,
            checks: PhantomData,
        }
    }
}

/// The C ABI of a native operator hook, `NativeFunctionView<R(const CallNode*)>`: it returns
/// the result, or the error, in an owned `TVMFFIAny`.
type Hook = extern "C" fn(*const CallObj) -> TVMFFIAny;

/// The C ABI of a native hook on a region statement, `NativeFunctionView<R(const
/// RegionStmtNode*)>`.
type RegionHook = extern "C" fn(*const RegionStmtObj) -> TVMFFIAny;

extern "C" fn validate_abi<F: Fn(&CallObj) -> Result<()>>(call: *const CallObj) -> TVMFFIAny {
    call_hook(call, zero_sized::<F>())
}

extern "C" fn infer_type_abi<F: Fn(&CallObj) -> Result<Type>>(call: *const CallObj) -> TVMFFIAny {
    call_hook(call, zero_sized::<F>())
}

extern "C" fn validate_signature<L: Checks>(call: *const CallObj) -> TVMFFIAny {
    call_hook(call, sig::validate::<L>)
}

extern "C" fn region_body_params_abi<F: Fn(&CallObj) -> Result<Array<Var>>>(
    call: *const CallObj,
) -> TVMFFIAny {
    call_hook(call, zero_sized::<F>())
}

extern "C" fn region_validate_abi<F: Fn(&RegionStmtObj) -> Result<()>>(
    region: *const RegionStmtObj,
) -> TVMFFIAny {
    call_hook(region, zero_sized::<F>())
}

/// The only value of `F`, a function or a closure that captures nothing.
fn zero_sized<F>() -> F {
    const {
        assert!(
            std::mem::size_of::<F>() == 0,
            "a native operator hook is a function or a closure that captures nothing"
        )
    };
    // SAFETY: a zero-sized type has no bytes to initialize.
    unsafe { std::mem::zeroed() }
}

/// Run `hook` on the node C++ passes (the Call or region being typed or checked) and return its
/// result as the hook ABI does. A panic, which must not unwind into C++, becomes an error.
fn call_hook<N, R: Into<Any>>(node: *const N, hook: impl Fn(&N) -> Result<R>) -> TVMFFIAny {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: C++ passes the node being typed or checked, alive for the call.
        match unsafe { node.as_ref() } {
            Some(node) => hook(node).map(Into::into),
            None => Err(Error::new(TYPE_ERROR, "Invalid Op Call", "")),
        }
    }));
    let value = match result {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => Any::from(error),
        Err(payload) => Any::from(tvm_ffi::function_internal::panic_to_error(payload)),
    };
    // SAFETY: the C++ caller takes ownership of the value.
    unsafe { Any::into_raw_ffi_any(value) }
}

/// Store `hook` as `op`'s validator, as C++ `OpDef::set_validator` and `OpDef::signature` do.
/// The validator is a member of the operator node, not an attribute column.
fn write_validator(op: &Op, hook: Hook, custom: bool) {
    // SAFETY: C++ `OpDef` writes these members of the registry's node in place too. No reference
    // into the node is live, and registration is not synchronized with readers (see `OpDef`).
    unsafe {
        let node = ObjectArc::as_raw(&op.base) as *mut OpObj;
        let previous = std::ptr::replace(
            std::ptr::addr_of_mut!((*node).validator),
            Any::from(hook as *mut c_void),
        );
        std::ptr::addr_of_mut!((*node).validator_is_custom).write(custom);
        drop(previous);
    }
}

/// The elements of an `ffi.Array`, read in place.
fn cells<T: tvm_ffi::type_traits::ContainerElement + Clone>(array: &Array<T>) -> &[TVMFFIAny] {
    // SAFETY: an `ffi.Array` is the object header followed by `TVMFFISeqCell`, and `array` keeps
    // its `size` elements alive.
    unsafe {
        let cell = &*(ObjectArc::as_raw(tvm_ffi::ObjectRefCore::data(array)) as *const SeqCell);
        if cell.size == 0 {
            return &[];
        }
        std::slice::from_raw_parts(cell.data, cell.size as usize)
    }
}

/// Signature descriptors: the argument types a generated validator checks.
///
/// A node type, such as [`StringImm`](super::StringImm) or [`Var`](super::Var), accepts an
/// instance of it, as C++ `TypeTraits<T>::CheckAnyStrict` does. [`PrimExpr`], [`IntExpr`],
/// [`PrimVar`] and [`TensorVar`] are the C++ view types, which also check the expression's type.
/// Implement [`ArgType`] for another check.
pub mod sig {
    use super::*;
    use crate::ir::{ExprObj, PrimTypeObj, VarObj};
    use crate::tirx::TensorTypeObj;
    use tvm_ffi::{AnyCompatible, DLDataTypeExt, ObjectCore, ObjectRefCore};

    /// What a signature checks an argument against (C++ `TypeTraits<T>`).
    pub trait ArgType: 'static {
        /// Whether `value`, which is not None, is accepted (C++ `CheckAnyStrict`).
        fn check(value: &TVMFFIAny) -> bool;
        /// The expected type, for an error (C++ `TypeStr`).
        fn type_str() -> std::string::String;
        /// The rejected value, for an error (C++ `GetMismatchTypeInfo`): its type key.
        fn mismatch(value: &TVMFFIAny) -> std::string::String {
            type_key(value.type_index)
        }
    }

    impl<T: ObjectRefCore + AnyCompatible + 'static> ArgType for T {
        fn check(value: &TVMFFIAny) -> bool {
            // SAFETY: `value` is a live FFI value.
            unsafe { T::check_any_strict(value) }
        }

        fn type_str() -> std::string::String {
            T::type_str()
        }
    }

    /// C++ `PrimExpr`: an expression whose type is a `PrimType`.
    pub struct PrimExpr;

    impl ArgType for PrimExpr {
        fn check(value: &TVMFFIAny) -> bool {
            prim_dtype(value).is_some()
        }

        fn type_str() -> std::string::String {
            "ir.PrimExpr".into()
        }

        fn mismatch(value: &TVMFFIAny) -> std::string::String {
            typed_expr_mismatch(value)
        }
    }

    /// C++ `IntExpr`: a `PrimExpr` of a scalar signed or unsigned integer type.
    pub struct IntExpr;

    impl ArgType for IntExpr {
        fn check(value: &TVMFFIAny) -> bool {
            const DL_INT: u8 = 0;
            const DL_UINT: u8 = 1;
            prim_dtype(value)
                .is_some_and(|dtype| dtype.lanes == 1 && matches!(dtype.code, DL_INT | DL_UINT))
        }

        fn type_str() -> std::string::String {
            "ir.IntExpr".into()
        }

        fn mismatch(value: &TVMFFIAny) -> std::string::String {
            typed_expr_mismatch(value)
        }
    }

    /// C++ `PrimVar`: a `Var`, not a subclass, whose type is a `PrimType`.
    pub struct PrimVar;

    impl ArgType for PrimVar {
        fn check(value: &TVMFFIAny) -> bool {
            var_of_type::<PrimTypeObj>(value)
        }

        fn type_str() -> std::string::String {
            VarObj::TYPE_KEY.into()
        }
    }

    /// C++ `tirx::TensorVar`: a `Var`, not a subclass, whose type is a tirx `TensorType`.
    pub struct TensorVar;

    impl ArgType for TensorVar {
        fn check(value: &TVMFFIAny) -> bool {
            var_of_type::<TensorTypeObj>(value)
        }

        fn type_str() -> std::string::String {
            VarObj::TYPE_KEY.into()
        }
    }

    /// Whether `value` is a `Var`, not a subclass, whose type is an `N`.
    fn var_of_type<N: ObjectCore>(value: &TVMFFIAny) -> bool {
        value.type_index == VarObj::type_index()
            && expr(value).is_some_and(|var| var.ty.as_node::<N>().is_some())
    }

    /// The expression `value` holds, if it holds one.
    fn expr(value: &TVMFFIAny) -> Option<&ExprObj> {
        let is_object = value.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32;
        // SAFETY: an object value of an `ir.Expr` type points to an `ExprObj`.
        unsafe {
            (is_object && Expr::check_any_strict(value))
                .then(|| &*(value.data_union.v_obj as *const ExprObj))
        }
    }

    fn prim_dtype(value: &TVMFFIAny) -> Option<tvm_ffi::DLDataType> {
        Some(expr(value)?.ty.as_node::<PrimTypeObj>()?.dtype)
    }

    /// C++ `TypeTraits<TypedExpr<T>>::GetMismatchTypeInfo`: an expression with its type.
    fn typed_expr_mismatch(value: &TVMFFIAny) -> std::string::String {
        let Some(expr) = expr(value) else { return type_key(value.type_index) };
        let ty = match expr.ty.as_node::<PrimTypeObj>() {
            Some(prim) => prim.dtype.to_string().as_str().to_string(),
            None => type_key(AnyView::from(&expr.ty).type_index()),
        };
        format!("{}[ty={ty}]", type_key(value.type_index))
    }

    /// A required value argument of type `T`.
    pub struct Arg<T>(PhantomData<T>);
    /// The variadic value tail, of type `T`.
    pub struct VarArgs<T>(PhantomData<T>);
    /// A required type argument of type `T`.
    pub struct TyArg<T>(PhantomData<T>);
    /// The variadic type-argument tail, of type `T`.
    pub struct VarTyArgs<T>(PhantomData<T>);
    /// `Call.attrs`, of type `T`.
    pub struct CallAttrs<T>(PhantomData<T>);

    /// The checks of a signature, as a list of descriptors with the newest first.
    #[doc(hidden)]
    pub trait Checks: 'static {
        const ARGS: usize;
        const TY_ARGS: usize;
        const VAR_ARGS: usize;
        const VAR_TY_ARGS: usize;
        const CALL_ATTRS: usize;
        /// The type key of the attrs type, if the signature names one.
        const ATTRS_TYPE_KEY: Option<&'static str>;

        /// Check the arguments of `call` these descriptors describe, in declaration order.
        /// `args` and `ty_args` count the signature's fixed arguments, where its tails start.
        fn check(call: &CallObj, op: &OpObj, args: usize, ty_args: usize) -> Result<()>;
    }

    impl Checks for () {
        const ARGS: usize = 0;
        const TY_ARGS: usize = 0;
        const VAR_ARGS: usize = 0;
        const VAR_TY_ARGS: usize = 0;
        const CALL_ATTRS: usize = 0;
        const ATTRS_TYPE_KEY: Option<&'static str> = None;

        fn check(_call: &CallObj, _op: &OpObj, _args: usize, _ty_args: usize) -> Result<()> {
            Ok(())
        }
    }

    impl<T: ArgType, L: Checks> Checks for (Arg<T>, L) {
        const ARGS: usize = L::ARGS + 1;
        const TY_ARGS: usize = L::TY_ARGS;
        const VAR_ARGS: usize = L::VAR_ARGS;
        const VAR_TY_ARGS: usize = L::VAR_TY_ARGS;
        const CALL_ATTRS: usize = L::CALL_ATTRS;
        const ATTRS_TYPE_KEY: Option<&'static str> = L::ATTRS_TYPE_KEY;

        fn check(call: &CallObj, op: &OpObj, args: usize, ty_args: usize) -> Result<()> {
            L::check(call, op, args, ty_args)?;
            check_value::<T, Expr>(op, false, cells(&call.args), L::ARGS)
        }
    }

    impl<T: ArgType, L: Checks> Checks for (VarArgs<T>, L) {
        const ARGS: usize = L::ARGS;
        const TY_ARGS: usize = L::TY_ARGS;
        const VAR_ARGS: usize = L::VAR_ARGS + 1;
        const VAR_TY_ARGS: usize = L::VAR_TY_ARGS;
        const CALL_ATTRS: usize = L::CALL_ATTRS;
        const ATTRS_TYPE_KEY: Option<&'static str> = L::ATTRS_TYPE_KEY;

        fn check(call: &CallObj, op: &OpObj, args: usize, ty_args: usize) -> Result<()> {
            L::check(call, op, args, ty_args)?;
            let values = cells(&call.args);
            (args..values.len()).try_for_each(|i| check_value::<T, Expr>(op, false, values, i))
        }
    }

    impl<T: ArgType, L: Checks> Checks for (TyArg<T>, L) {
        const ARGS: usize = L::ARGS;
        const TY_ARGS: usize = L::TY_ARGS + 1;
        const VAR_ARGS: usize = L::VAR_ARGS;
        const VAR_TY_ARGS: usize = L::VAR_TY_ARGS;
        const CALL_ATTRS: usize = L::CALL_ATTRS;
        const ATTRS_TYPE_KEY: Option<&'static str> = L::ATTRS_TYPE_KEY;

        fn check(call: &CallObj, op: &OpObj, args: usize, ty_args: usize) -> Result<()> {
            L::check(call, op, args, ty_args)?;
            check_value::<T, Type>(op, true, cells(&call.ty_args), L::TY_ARGS)
        }
    }

    impl<T: ArgType, L: Checks> Checks for (VarTyArgs<T>, L) {
        const ARGS: usize = L::ARGS;
        const TY_ARGS: usize = L::TY_ARGS;
        const VAR_ARGS: usize = L::VAR_ARGS;
        const VAR_TY_ARGS: usize = L::VAR_TY_ARGS + 1;
        const CALL_ATTRS: usize = L::CALL_ATTRS;
        const ATTRS_TYPE_KEY: Option<&'static str> = L::ATTRS_TYPE_KEY;

        fn check(call: &CallObj, op: &OpObj, args: usize, ty_args: usize) -> Result<()> {
            L::check(call, op, args, ty_args)?;
            let values = cells(&call.ty_args);
            (ty_args..values.len()).try_for_each(|i| check_value::<T, Type>(op, true, values, i))
        }
    }

    impl<T: ObjectRefCore + 'static, L: Checks> Checks for (CallAttrs<T>, L) {
        const ARGS: usize = L::ARGS;
        const TY_ARGS: usize = L::TY_ARGS;
        const VAR_ARGS: usize = L::VAR_ARGS;
        const VAR_TY_ARGS: usize = L::VAR_TY_ARGS;
        const CALL_ATTRS: usize = L::CALL_ATTRS + 1;
        const ATTRS_TYPE_KEY: Option<&'static str> = Some(T::ContainerType::TYPE_KEY);

        fn check(call: &CallObj, op: &OpObj, args: usize, ty_args: usize) -> Result<()> {
            L::check(call, op, args, ty_args)?;
            let actual = match &call.attrs {
                Some(attrs) if attrs.as_node::<T::ContainerType>().is_some() => return Ok(()),
                Some(attrs) => format!("`{}`", type_key(AnyView::from(attrs).type_index())),
                None => "None".into(),
            };
            let message = format!(
                "Op `{}`: Call.attrs expected `{}`, got {actual}",
                op.name.as_str(),
                T::ContainerType::TYPE_KEY,
            );
            Err(Error::new(TYPE_ERROR, &message, ""))
        }
    }

    /// C++ `ValidateSignature<Specs...>`: the argument counts, then each argument's type and the
    /// attrs, in declaration order.
    pub(super) fn validate<L: Checks>(call: &CallObj) -> Result<()> {
        const {
            assert!(
                L::VAR_ARGS <= 1 && L::VAR_TY_ARGS <= 1 && L::CALL_ATTRS <= 1,
                "a signature has at most one variadic tail of each kind and one call_attrs"
            )
        };
        let Some(op) = call.op.as_node::<OpObj>() else {
            return Err(Error::new(TYPE_ERROR, "Invalid Op Call", ""));
        };
        let count = |label: &str, expected: usize, tail: bool, unit: &str, got: usize| {
            let short = if tail { got < expected } else { got != expected };
            if !short {
                return Ok(());
            }
            let at_least = if tail { "at least " } else { "" };
            let plural = if expected == 1 { "" } else { "s" };
            let message = format!(
                "Op `{}`: {label} expected {at_least}{expected} {unit}{plural}, got {got}",
                op.name.as_str()
            );
            Err(Error::new(TYPE_ERROR, &message, ""))
        };
        count("Call.args", L::ARGS, L::VAR_ARGS == 1, "argument", call.args.len())?;
        count(
            "Call.ty_args",
            L::TY_ARGS,
            L::VAR_TY_ARGS == 1,
            "type argument",
            call.ty_args.len(),
        )?;
        L::check(call, op, L::ARGS, L::TY_ARGS)
    }

    /// Check argument `index` against `T`. `Any` is the category's base type, which C++ does
    /// not check.
    fn check_value<T: ArgType, Any: 'static>(
        op: &OpObj,
        type_arg: bool,
        values: &[TVMFFIAny],
        index: usize,
    ) -> Result<()> {
        if TypeId::of::<T>() == TypeId::of::<Any>() {
            return Ok(());
        }
        let value = &values[index];
        if value.type_index != TVMFFITypeIndex::kTVMFFINone as i32 && T::check(value) {
            return Ok(());
        }
        let (fixed, tail) = match type_arg {
            false => (&op.args_info, &op.var_args_info),
            true => (&op.ty_args_info, &op.var_ty_args_info),
        };
        let name = match fixed.get(index) {
            Ok(info) => Some(info.name.clone()),
            Err(_) => tail.as_ref().map(|info| info.name.clone()),
        };
        let name = match name {
            Some(name) if !name.as_str().is_empty() => format!(" (`{}`)", name.as_str()),
            _ => std::string::String::new(),
        };
        let message = format!(
            "Op `{}`: `{}[{index}]`{name} expected `{}`, got `{}`.",
            op.name.as_str(),
            if type_arg { "Call.ty_args" } else { "Call.args" },
            T::type_str(),
            T::mismatch(value),
        );
        Err(Error::new(TYPE_ERROR, &message, ""))
    }
}
