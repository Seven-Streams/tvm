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

//! The CUDA execution-context analysis `TileDispatch` runs inside a kernel
//! (`src/backend/cuda/tile/{exec_scope,filter_canonical,exec_context,exec_analysis}.cc`), and the
//! lowering of CUDA index calls (`LowerIndices`).
//!
//! The analysis tracks the active thread set `A` of each program point as boxes on the axes
//! `laneid`, `warpid` and `cta_id` (or the factored CTA axes `bx`/`by`/`bz`, `cbx`/`cby`/`cbz`),
//! narrowed by the `if` predicates on CUDA index calls it can read, and splits it into the
//! `(inter, intra)` maps a tile dispatcher receives. C++ keeps `A` in a `TileLayout`, as storage
//! only; here it is the ordered axis list that layout holds.

use tvm_ffi::{
    bail, AnyCompatible, Array, Map, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, String,
    WalkOrder, WalkResult, INTERNAL_ERROR, VALUE_ERROR,
};

use crate::ir::{Bind, Call, CallObj, Expr, IntImmObj, OpHandle, OpObj, SeqStmt, Stmt, StringImm, Var};
use crate::ops::{int32, int32_dtype, int_imm, prim_type, Ex, PrimOps};
use crate::passes::analyzer::is_const_int;
use crate::prim::{
    AndObj, BitwiseAndObj, BitwiseNot, BitwiseNotObj, BitwiseOrObj, BitwiseXorObj, CastObj, EQObj,
    FloorDiv, FloorMod, FloorModObj, GEObj, GTObj, LEObj, LShiftObj, LTObj, ModObj, RShiftObj,
};
use crate::sym::{detect_linear_equation, Analyzer};
use crate::tirx::op::{
    CUDA_BLOCK_DIM, CUDA_BLOCK_IDX, CUDA_ELECT_SYNC, CUDA_GRID_DIM, CUDA_MOV_SREG, CUDA_THREAD_IDX,
    GPU_ACTIVE_THREAD_SELECTOR, GPU_THREAD_FILTER, GPU_WARP_SHUFFLE,
};
use crate::tirx::ExecScope;

const WARP_SIZE: i64 = 32;
const WG_SIZE: i64 = 4;
/// `sym::ConstIntBound::kPosInf` and `kNegInf`.
const POS_INF: i64 = i64::MAX;
const NEG_INF: i64 = -i64::MAX;

fn int64(value: i64) -> Result<Expr> {
    let dtype = tvm_ffi::DLDataType { code: 0, bits: 64, lanes: 1 };
    Ok(int_imm(dtype, value as i128)?.into())
}

/// An `IntImm` whose value fits `int64_t`.
fn as_i64(expr: &Expr) -> Option<i64> {
    expr.as_node::<IntImmObj>().and_then(|imm| imm.value.to_i64())
}

/// `analyzer->CanProveEqual(expr, 0)` with a fresh analyzer.
fn is_zero(expr: &Expr) -> Result<bool> {
    if let Some(imm) = expr.as_node::<IntImmObj>() {
        return Ok(imm.value == tvm_ffi::BigInt::from(0i64));
    }
    Analyzer::new()?.can_prove_equal(expr, &int32(0)?.into())
}

/// C++ `FloorDivInt`, with its unsigned wrap-around for a negative dividend.
fn floor_div(a: i64, b: i64) -> Result<i64> {
    check_cmp!(b, >, 0, INTERNAL_ERROR, "b > 0");
    if a >= 0 {
        return Ok(a / b);
    }
    let magnitude = (a.wrapping_neg() as u64).wrapping_add(b as u64 - 1) / b as u64;
    Ok((magnitude as i64).wrapping_neg())
}

fn ceil_div(a: i64, b: i64) -> Result<i64> {
    Ok(floor_div(a.wrapping_neg(), b)?.wrapping_neg())
}

fn normalize_mod(value: i64, modulus: i64) -> i64 {
    let ret = value % modulus;
    if ret < 0 {
        ret + modulus
    } else {
        ret
    }
}

fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        return (a, 1, 0);
    }
    let (g, x1, y1) = extended_gcd(b, a % b);
    (g, y1, x1 - (a / b) * y1)
}

fn modular_inverse(value: i64, modulus: i64) -> Result<i64> {
    let (g, x, _) = extended_gcd(normalize_mod(value, modulus), modulus);
    check_cmp!(g, ==, 1, INTERNAL_ERROR, "g == 1");
    Ok(normalize_mod(x, modulus))
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

// ---------------------------------------------------------------------------------------------
// Scopes (`exec_scope.h`)

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ScopeKind {
    Cluster,
    Cta,
    Warpgroup,
    Warp,
    Thread,
}

/// C++ `StringToScopeKind`, which aborts on an unknown name.
fn scope_kind(name: &str) -> Result<ScopeKind> {
    Ok(match name {
        "cluster" => ScopeKind::Cluster,
        "cta" => ScopeKind::Cta,
        "warpgroup" => ScopeKind::Warpgroup,
        "warp" => ScopeKind::Warp,
        "thread" => ScopeKind::Thread,
        _ => bail!(INTERNAL_ERROR, "Unknown scope kind name: {name}"),
    })
}

/// The `(parent, current)` scope a CUDA index call binds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ScopeBinding {
    KernelCluster,
    KernelCta,
    ClusterCta,
    CtaWarpgroup,
    CtaWarp,
    WarpgroupWarp,
    WarpThread,
    CtaThread,
    WarpgroupThread,
    ClusterCtaPair,
}

// ---------------------------------------------------------------------------------------------
// Active thread set (`exec_context.{h,cc}`)

/// The active slice `offset + stride * [0, extent)` of one axis.
#[derive(Clone)]
struct AxisRange {
    extent: Expr,
    offset: Expr,
    stride: Expr,
}

fn make_range(extent: i64, offset: i64) -> Result<AxisRange> {
    Ok(AxisRange { extent: int64(extent)?, offset: int64(offset)?, stride: int64(1)? })
}

impl AxisRange {
    fn ints(&self) -> Option<(i64, i64, i64)> {
        Some((as_i64(&self.offset)?, as_i64(&self.extent)?, as_i64(&self.stride)?))
    }

    /// The part in `[lo, hi)`, or `None` if empty or not structural.
    fn intersect(&self, lo: i64, hi: i64) -> Result<Option<AxisRange>> {
        let Some((offset, extent, stride)) = self.ints() else { return Ok(None) };
        if stride <= 0 {
            return Ok(None);
        }
        let i_lo = ceil_div(lo.wrapping_sub(offset), stride)?.max(0);
        let i_hi = extent.min(floor_div(hi.wrapping_sub(1).wrapping_sub(offset), stride)? + 1);
        if i_hi <= i_lo {
            return Ok(None);
        }
        Ok(Some(AxisRange {
            extent: int64(i_hi - i_lo)?,
            offset: int64(offset + stride * i_lo)?,
            stride: int64(stride)?,
        }))
    }

    /// The values with `axis % modulus == residue`, or `None` if empty or not structural.
    fn modulo(&self, modulus: i64, residue: i64) -> Result<Option<AxisRange>> {
        if modulus <= 0 {
            return Ok(None);
        }
        let Some((offset, extent, stride)) = self.ints() else { return Ok(None) };
        if stride <= 0 {
            return Ok(None);
        }
        let residue = normalize_mod(residue, modulus);
        let rhs = normalize_mod(residue - offset, modulus);
        let g = gcd(stride.abs(), modulus.abs());
        if rhs % g != 0 {
            return Ok(None);
        }
        let (reduced_stride, reduced_rhs, reduced_modulus) = (stride / g, rhs / g, modulus / g);
        let i0 = normalize_mod(
            reduced_rhs * modular_inverse(reduced_stride, reduced_modulus)?,
            reduced_modulus,
        );
        if i0 >= extent {
            return Ok(None);
        }
        Ok(Some(AxisRange {
            extent: int64((extent - 1 - i0) / reduced_modulus + 1)?,
            offset: int64(offset + stride * i0)?,
            stride: int64(stride * reduced_modulus)?,
        }))
    }
}

/// The active thread set: its axes in order, an offset that is provably zero stored as `0`.
#[derive(Clone)]
struct ActiveSet {
    axes: Vec<(std::string::String, AxisRange)>,
}

impl ActiveSet {
    /// C++ `MakeActiveSet`.
    fn new(axes: Vec<(std::string::String, AxisRange)>) -> Result<Self> {
        let mut normalized = Vec::with_capacity(axes.len());
        for (name, mut range) in axes {
            if is_zero(&range.offset)? {
                range.offset = int64(0)?;
            }
            normalized.push((name, range));
        }
        Ok(Self { axes: normalized })
    }

    fn get(&self, axis: &str) -> Option<AxisRange> {
        self.axes.iter().find(|(name, _)| name == axis).map(|(_, range)| range.clone())
    }

    fn with_axis(&self, axis: &str, range: AxisRange) -> Result<ActiveSet> {
        let mut axes = self.axes.clone();
        let Some(entry) = axes.iter_mut().find(|(name, _)| name == axis) else {
            bail!(INTERNAL_ERROR, "Internal Error: unknown active-set axis {axis}");
        };
        entry.1 = range;
        ActiveSet::new(axes)
    }

    /// C++ `InitialActiveSet`.
    fn initial(
        lane: i64,
        warp: i64,
        cta: i64,
        cta_axes: &[(std::string::String, i64)],
    ) -> Result<Self> {
        let mut axes = vec![
            ("laneid".to_string(), make_range(lane, 0)?),
            ("warpid".to_string(), make_range(warp, 0)?),
        ];
        if cta_axes.is_empty() {
            axes.push(("cta_id".to_string(), make_range(cta, 0)?));
        } else {
            for (axis, extent) in cta_axes {
                axes.push((axis.clone(), make_range(*extent, 0)?));
            }
        }
        ActiveSet::new(axes)
    }

    fn narrow_axis(&self, axis: &str, lo: i64, hi: i64) -> Result<Option<ActiveSet>> {
        let Some(current) = self.get(axis) else { return Ok(None) };
        match current.intersect(lo, hi)? {
            Some(narrowed) => Ok(Some(self.with_axis(axis, narrowed)?)),
            None => Ok(None),
        }
    }

    fn modulo_axis(&self, axis: &str, modulus: i64, residue: i64) -> Result<Option<ActiveSet>> {
        let Some(current) = self.get(axis) else { return Ok(None) };
        match current.modulo(modulus, residue)? {
            Some(narrowed) => Ok(Some(self.with_axis(axis, narrowed)?)),
            None => Ok(None),
        }
    }
}

/// One side of a split as a dispatcher receives it: axis -> `[extent, offset(, stride)]`.
pub(crate) type SplitMap = Map<String, Array<Expr>>;

/// One side of a split, keyed by axis name.
type SplitSide = Vec<(std::string::String, AxisRange)>;

fn set_side(side: &mut SplitSide, axis: &str, range: AxisRange) {
    match side.iter_mut().find(|(name, _)| name == axis) {
        Some(entry) => entry.1 = range,
        None => side.push((axis.to_string(), range)),
    }
}

/// The CTA axes of `A`: `cta_id`, or every axis but `laneid` and `warpid`.
fn add_cta_axes(active: &ActiveSet, side: &mut SplitSide) {
    if let Some(cta_id) = active.get("cta_id") {
        set_side(side, "cta_id", cta_id);
        return;
    }
    for (axis, range) in &active.axes {
        if axis != "laneid" && axis != "warpid" {
            set_side(side, axis, range.clone());
        }
    }
}

/// C++ `FactorWarpid`: `warpid` as `(wid_in_wg, wgid)`.
fn factor_warpid(warpid: &AxisRange) -> Result<Option<(AxisRange, AxisRange)>> {
    let Some((offset, extent, stride)) = warpid.ints() else { return Ok(None) };
    if stride != 1 {
        return Ok(None);
    }
    let (wid_off, wgid_off) = (offset % WG_SIZE, offset / WG_SIZE);
    if wid_off == 0 && extent % WG_SIZE == 0 {
        return Ok(Some((make_range(WG_SIZE, 0)?, make_range(extent / WG_SIZE, wgid_off)?)));
    }
    if extent <= WG_SIZE - wid_off {
        return Ok(Some((make_range(extent, wid_off)?, make_range(1, wgid_off)?)));
    }
    Ok(None)
}

fn same_int_range(lhs: &AxisRange, rhs: &AxisRange) -> bool {
    matches!((lhs.ints(), rhs.ints()), (Some(a), Some(b)) if a == b)
}

/// C++ `NarrowFlatProductRange`: a flat range `[lo, hi)` of `major * 32 + lane`.
fn narrow_flat_product(
    major: &AxisRange,
    lane: &AxisRange,
    lo: i64,
    hi: i64,
) -> Result<Option<(AxisRange, AxisRange)>> {
    let (Some((major_off, major_ext, major_stride)), Some((lane_off, lane_ext, lane_stride))) =
        (major.ints(), lane.ints())
    else {
        return Ok(None);
    };
    if major_ext <= 0 || lane_ext <= 0 || major_stride <= 0 || lane_stride <= 0 {
        return Ok(None);
    }
    let active_min = major_off.wrapping_mul(WARP_SIZE).wrapping_add(lane_off);
    let active_max = (major_off + major_stride * (major_ext - 1))
        .wrapping_mul(WARP_SIZE)
        .wrapping_add(lane_off + lane_stride * (lane_ext - 1))
        .wrapping_add(1);
    if lo <= active_min && active_max <= hi {
        return Ok(Some((major.clone(), lane.clone())));
    }
    if major_stride != 1 || lane_stride != 1 {
        return Ok(None);
    }
    let lane_hi = lane_off + lane_ext;
    let major_hi = major_off + major_ext;
    let hit_lo = major_off.max(floor_div(lo.wrapping_sub(lane_hi), WARP_SIZE)?.wrapping_add(1));
    let hit_hi = major_hi.min(ceil_div(hi.wrapping_sub(lane_off), WARP_SIZE)?);
    if hit_hi <= hit_lo {
        return Ok(None);
    }
    if hit_hi == hit_lo + 1 {
        let m = hit_lo;
        let new_lane_lo = lane_off.max(lo.wrapping_sub(m.wrapping_mul(WARP_SIZE)));
        let new_lane_hi = lane_hi.min(hi.wrapping_sub(m.wrapping_mul(WARP_SIZE)));
        if new_lane_hi <= new_lane_lo {
            return Ok(None);
        }
        return Ok(Some((make_range(1, m)?, make_range(new_lane_hi - new_lane_lo, new_lane_lo)?)));
    }
    if lo <= hit_lo.wrapping_mul(WARP_SIZE).wrapping_add(lane_off)
        && (hit_hi - 1).wrapping_mul(WARP_SIZE).wrapping_add(lane_hi) <= hi
    {
        return Ok(Some((make_range(hit_hi - hit_lo, hit_lo)?, lane.clone())));
    }
    Ok(None)
}

fn narrow_flat_cta_thread(active: &ActiveSet, lo: i64, hi: i64) -> Result<Option<ActiveSet>> {
    let (Some(lane), Some(warpid)) = (active.get("laneid"), active.get("warpid")) else {
        return Ok(None);
    };
    let Some((new_warpid, new_lane)) = narrow_flat_product(&warpid, &lane, lo, hi)? else {
        return Ok(None);
    };
    Ok(Some(active.with_axis("laneid", new_lane)?.with_axis("warpid", new_warpid)?))
}

fn narrow_flat_warpgroup_thread(active: &ActiveSet, lo: i64, hi: i64) -> Result<Option<ActiveSet>> {
    let (Some(lane), Some(warpid)) = (active.get("laneid"), active.get("warpid")) else {
        return Ok(None);
    };
    let Some((wid_in_wg, wgid)) = factor_warpid(&warpid)? else { return Ok(None) };
    let Some((new_wid_in_wg, new_lane)) = narrow_flat_product(&wid_in_wg, &lane, lo, hi)? else {
        return Ok(None);
    };
    let (Some(wgid_ext), Some(wgid_off)) = (as_i64(&wgid.extent), as_i64(&wgid.offset)) else {
        return Ok(None);
    };
    if wgid_ext != 1 {
        if same_int_range(&new_lane, &lane) && same_int_range(&new_wid_in_wg, &wid_in_wg) {
            return Ok(Some(active.clone()));
        }
        return Ok(None);
    }
    let (Some(wid_ext), Some(wid_off)) =
        (as_i64(&new_wid_in_wg.extent), as_i64(&new_wid_in_wg.offset))
    else {
        return Ok(None);
    };
    let warpid = make_range(wid_ext, wgid_off * WG_SIZE + wid_off)?;
    Ok(Some(active.with_axis("laneid", new_lane)?.with_axis("warpid", warpid)?))
}

/// C++ `FilterNarrow`: `A` narrowed to `[lo, hi)` on the axis `binding` reads.
fn filter_narrow(
    active: &ActiveSet,
    binding: ScopeBinding,
    lo: i64,
    hi: i64,
) -> Result<Option<ActiveSet>> {
    if lo >= hi {
        return Ok(None);
    }
    match binding {
        ScopeBinding::WarpThread => active.narrow_axis("laneid", lo, hi),
        ScopeBinding::CtaWarp => active.narrow_axis("warpid", lo, hi),
        ScopeBinding::KernelCta | ScopeBinding::ClusterCta => active.narrow_axis("cta_id", lo, hi),
        ScopeBinding::CtaWarpgroup => {
            let Some(warpid) = active.get("warpid") else { return Ok(None) };
            let (Some(offset), Some(extent)) = (as_i64(&warpid.offset), as_i64(&warpid.extent))
            else {
                return Ok(None);
            };
            if offset % WG_SIZE != 0 || extent % WG_SIZE != 0 {
                return Ok(None);
            }
            let outer = make_range(extent / WG_SIZE, offset / WG_SIZE)?;
            let Some(new_outer) = outer.intersect(lo, hi)? else { return Ok(None) };
            let (Some(outer_ext), Some(outer_off)) =
                (as_i64(&new_outer.extent), as_i64(&new_outer.offset))
            else {
                bail!(INTERNAL_ERROR, "Check failed: TryAsInt64(new_outer.extent)");
            };
            Ok(Some(
                active
                    .with_axis("warpid", make_range(outer_ext * WG_SIZE, outer_off * WG_SIZE)?)?,
            ))
        }
        ScopeBinding::WarpgroupWarp => {
            let Some(warpid) = active.get("warpid") else { return Ok(None) };
            let (Some(offset), Some(extent)) = (as_i64(&warpid.offset), as_i64(&warpid.extent))
            else {
                return Ok(None);
            };
            let inner_off = offset % WG_SIZE;
            if extent > WG_SIZE - inner_off {
                return Ok(None);
            }
            let inner = make_range(extent, inner_off)?;
            let Some(new_inner) = inner.intersect(lo, hi)? else { return Ok(None) };
            let (Some(inner_ext), Some(new_off)) =
                (as_i64(&new_inner.extent), as_i64(&new_inner.offset))
            else {
                bail!(INTERNAL_ERROR, "Check failed: TryAsInt64(new_inner.extent)");
            };
            let outer_base = (offset / WG_SIZE) * WG_SIZE;
            Ok(Some(active.with_axis("warpid", make_range(inner_ext, outer_base + new_off)?)?))
        }
        ScopeBinding::KernelCluster | ScopeBinding::ClusterCtaPair => Ok(None),
        ScopeBinding::CtaThread => narrow_flat_cta_thread(active, lo, hi),
        ScopeBinding::WarpgroupThread => narrow_flat_warpgroup_thread(active, lo, hi),
    }
}

/// C++ `ScopeSwitch`: `A` factored into `(inter, intra)` for `kind`, or `None`.
fn scope_switch(active: &ActiveSet, kind: ScopeKind) -> Result<Option<(SplitSide, SplitSide)>> {
    let (Some(laneid), Some(warpid)) = (active.get("laneid"), active.get("warpid")) else {
        bail!(INTERNAL_ERROR, "Check failed: A.GetAxis(\"laneid\", &laneid)");
    };
    let (mut inter, mut intra) = (SplitSide::new(), SplitSide::new());
    match kind {
        ScopeKind::Thread => {
            set_side(&mut inter, "laneid", laneid);
            set_side(&mut inter, "warpid", warpid);
            add_cta_axes(active, &mut inter);
        }
        ScopeKind::Warp => {
            set_side(&mut intra, "laneid", laneid);
            set_side(&mut inter, "warpid", warpid);
            add_cta_axes(active, &mut inter);
        }
        ScopeKind::Cta => {
            set_side(&mut intra, "laneid", laneid);
            set_side(&mut intra, "warpid", warpid);
            add_cta_axes(active, &mut inter);
        }
        ScopeKind::Cluster => {
            set_side(&mut intra, "laneid", laneid);
            set_side(&mut intra, "warpid", warpid);
            add_cta_axes(active, &mut intra);
        }
        ScopeKind::Warpgroup => {
            let Some((wid_in_wg, wgid)) = factor_warpid(&warpid)? else { return Ok(None) };
            set_side(&mut intra, "laneid", laneid);
            set_side(&mut intra, "wid_in_wg", wid_in_wg);
            set_side(&mut inter, "wgid", wgid);
            add_cta_axes(active, &mut inter);
        }
    }
    Ok(Some((inter, intra)))
}

/// C++ `EncodeSplitSide`: axis -> `[extent, offset]`, or `[extent, offset, stride]` if strided.
fn encode_side(side: &SplitSide) -> Result<SplitMap> {
    let mut out = Vec::with_capacity(side.len());
    for (axis, range) in side {
        let unit = is_zero(&(&range.stride - int64(1)?).expr()?)?;
        let mut fields = vec![range.extent.clone(), range.offset.clone()];
        if !unit {
            fields.push(range.stride.clone());
        }
        out.push((String::from(axis.as_str()), Array::new(fields)));
    }
    Ok(out.into_iter().collect())
}

/// The analysis state at one program point.
#[derive(Clone)]
struct ExecContext {
    active: ActiveSet,
    kind: ScopeKind,
}

impl ExecContext {
    /// `A` changed to `active`, keeping the scope kind, if it still splits.
    fn with_active(&self, active: Option<ActiveSet>) -> Result<Option<ExecContext>> {
        let Some(active) = active else { return Ok(None) };
        if scope_switch(&active, self.kind)?.is_none() {
            return Ok(None);
        }
        Ok(Some(ExecContext { active, kind: self.kind }))
    }

    fn with_filter(&self, binding: ScopeBinding, lo: i64, hi: i64) -> Result<Option<ExecContext>> {
        self.with_active(filter_narrow(&self.active, binding, lo, hi)?)
    }

    fn with_selector(&self, binding: ScopeBinding, selector: Expr) -> Result<Option<ExecContext>> {
        if binding != ScopeBinding::WarpThread {
            return Ok(None);
        }
        let range = AxisRange { extent: int64(1)?, offset: selector, stride: int64(1)? };
        self.with_active(Some(self.active.with_axis("laneid", range)?))
    }

    fn with_cta_axis_filter(&self, axis: &str, lo: i64, hi: i64) -> Result<Option<ExecContext>> {
        if lo >= hi {
            return Ok(None);
        }
        self.with_active(self.active.narrow_axis(axis, lo, hi)?)
    }

    fn with_cta_axis_modulo(
        &self,
        axis: &str,
        modulus: i64,
        residue: i64,
    ) -> Result<Option<ExecContext>> {
        self.with_active(self.active.modulo_axis(axis, modulus, residue)?)
    }
}

// ---------------------------------------------------------------------------------------------
// Canonical thread-filter predicates (`filter_canonical.cc`)

enum FilterAtom {
    /// `var` in `[lo, hi)`.
    Range { var: Var, lo: i64, hi: i64 },
    /// A bare `tirx.cuda.elect_sync()` call.
    ElectSync(Expr),
}

fn strip_cast(expr: &Expr) -> Expr {
    let mut current = expr.clone();
    while let Some(cast) = current.as_node::<CastObj>() {
        current = cast.value.clone();
    }
    current
}

/// The conjuncts of `pred` under `&&` and bitwise `&`, casts stripped.
fn flatten_conjuncts_stripped(pred: &Expr, out: &mut Vec<Expr>) {
    let stripped = strip_cast(pred);
    if let Some(node) = stripped.as_node::<AndObj>() {
        flatten_conjuncts_stripped(&node.a, out);
        flatten_conjuncts_stripped(&node.b, out);
    } else if let Some(node) = stripped.as_node::<BitwiseAndObj>() {
        flatten_conjuncts_stripped(&node.a, out);
        flatten_conjuncts_stripped(&node.b, out);
    } else {
        out.push(stripped);
    }
}

#[derive(Clone, Copy)]
enum CmpOp {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

fn reflect(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Eq => CmpOp::Eq,
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Ge => CmpOp::Le,
    }
}

fn op_to_range(op: CmpOp, c: i64) -> (i64, i64) {
    match op {
        CmpOp::Eq => (c, c.wrapping_add(1)),
        CmpOp::Lt => (NEG_INF, c),
        CmpOp::Le => (NEG_INF, c.wrapping_add(1)),
        CmpOp::Gt => (c.wrapping_add(1), POS_INF),
        CmpOp::Ge => (c, POS_INF),
    }
}

/// The operands of a comparison.
fn comparison(expr: &Expr) -> Option<(CmpOp, &Expr, &Expr)> {
    if let Some(node) = expr.as_node::<EQObj>() {
        return Some((CmpOp::Eq, &node.a, &node.b));
    }
    if let Some(node) = expr.as_node::<LTObj>() {
        return Some((CmpOp::Lt, &node.a, &node.b));
    }
    if let Some(node) = expr.as_node::<LEObj>() {
        return Some((CmpOp::Le, &node.a, &node.b));
    }
    if let Some(node) = expr.as_node::<GTObj>() {
        return Some((CmpOp::Gt, &node.a, &node.b));
    }
    if let Some(node) = expr.as_node::<GEObj>() {
        return Some((CmpOp::Ge, &node.a, &node.b));
    }
    None
}

fn parse_compare_atom(
    expr: &Expr,
    is_scope_id: &mut dyn FnMut(&Var) -> bool,
) -> Option<FilterAtom> {
    let (op, lhs, rhs) = comparison(expr)?;
    let as_var = |e: &Expr| e.clone().try_cast::<Var>().ok();
    let (var, imm, mirrored) = match (as_var(lhs), rhs.as_node::<IntImmObj>()) {
        (Some(var), Some(imm)) => (var, imm, false),
        _ => match (as_var(rhs), lhs.as_node::<IntImmObj>()) {
            (Some(var), Some(imm)) => (var, imm, true),
            _ => return None,
        },
    };
    if !is_scope_id(&var) {
        return None;
    }
    let normalized = if mirrored { reflect(op) } else { op };
    let (lo, hi) = op_to_range(normalized, imm.value.to_i64()?);
    Some(FilterAtom::Range { var, lo, hi })
}

/// C++ `TryClassifyCanonical`: `cond` as a conjunction of canonical atoms, or `None`.
fn classify_canonical(
    cond: &Expr,
    is_scope_id: &mut dyn FnMut(&Var) -> bool,
) -> Option<Vec<FilterAtom>> {
    let mut terms = Vec::new();
    flatten_conjuncts_stripped(cond, &mut terms);
    let mut atoms = Vec::with_capacity(terms.len());
    for term in terms {
        if term.as_node::<CallObj>().is_some_and(|call| CUDA_ELECT_SYNC.is(&call.op)) {
            atoms.push(FilterAtom::ElectSync(term));
            continue;
        }
        atoms.push(parse_compare_atom(&term, is_scope_id)?);
    }
    if atoms.is_empty() {
        return None;
    }
    Some(atoms)
}

// ---------------------------------------------------------------------------------------------
// The analysis (`exec_analysis.cc`)

/// The launch parameters of a kernel: tag (`threadIdx.x`, ...) -> (variable, extent).
pub(crate) type LaunchParams = Vec<(std::string::String, Var, Expr)>;

fn launch_param<'a>(
    params: &'a LaunchParams,
    tag: &str,
) -> Option<&'a (std::string::String, Var, Expr)> {
    params.iter().find(|(name, _, _)| name == tag)
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ScopeIdTarget {
    binding: ScopeBinding,
    dim: i32,
    ndim: i32,
}

impl ScopeIdTarget {
    fn flat(binding: ScopeBinding) -> Self {
        ScopeIdTarget { binding, dim: 0, ndim: 1 }
    }
}

struct ScopeIdRange {
    target: ScopeIdTarget,
    lo: i64,
    hi: i64,
}

/// Does `expr` contain a call to `op`?
/// C++ `TVM_FFI_ICHECK_EQ(call->args.size(), 2)` of a `tirx.gpu_thread_filter` call.
fn check_thread_filter_arity(call: &CallObj) -> Result<()> {
    let count = call.args.len();
    check_cmp!(
        count,
        ==,
        2,
        INTERNAL_ERROR,
        "call->args.size() == 2",
        "TIRxError: tirx.gpu_thread_filter expects (var, cond); got {} args",
        count
    );
    Ok(())
}

fn contains_call(expr: &Expr, op: &OpHandle) -> Result<bool> {
    let found = tvm_ffi::structural_walk(
        expr,
        |call: Call| {
            if op.is(&call.op) {
                WalkResult::Interrupt
            } else {
                WalkResult::Advance
            }
        },
        WalkOrder::PreOrder,
    )?;
    Ok(found.is_some())
}

/// C++ `backend::cuda::ExecAnalysis`.
pub(crate) struct ExecAnalysis {
    variable_cluster: bool,
    cluster_bounds: Vec<(std::string::String, Expr)>,
    /// The variables bound to a CUDA index, in binding order.
    scope_aliases: Vec<(Var, ScopeIdTarget)>,
    analyzer: Analyzer,
    stack: Vec<ExecContext>,
    launch_params: LaunchParams,
    cluster_cta_axis_extents: Vec<(std::string::String, i64)>,
}

impl ExecAnalysis {
    pub(crate) fn new(
        launch_params: &LaunchParams,
        cluster_bounds: &[(std::string::String, Expr)],
        variable_cluster: bool,
    ) -> Result<Self> {
        let mut analysis = ExecAnalysis {
            variable_cluster,
            cluster_bounds: cluster_bounds.to_vec(),
            scope_aliases: Vec::new(),
            analyzer: Analyzer::new()?,
            stack: Vec::new(),
            launch_params: launch_params.clone(),
            cluster_cta_axis_extents: Vec::new(),
        };
        analysis.push_kernel_entry()?;
        Ok(analysis)
    }

    fn cluster_bound(&self, tag: &str) -> Option<&Expr> {
        self.cluster_bounds.iter().find(|(name, _)| name == tag).map(|(_, extent)| extent)
    }

    /// C++ `PushKernelEntryCtx`. C++ also logs a warning when the thread extents are unknown.
    fn push_kernel_entry(&mut self) -> Result<bool> {
        let (params, bounds) = (self.launch_params.clone(), self.cluster_bounds.clone());
        let bound =
            |tag: &str| bounds.iter().find(|(name, _)| name == tag).map(|(_, extent)| extent);
        let product = |keys: &[&str]| -> i64 {
            let mut n: i64 = 1;
            for key in keys {
                let Some((_, _, extent)) = launch_param(&params, key) else { continue };
                let Some(imm) = extent.as_node::<IntImmObj>() else { return 0 };
                match imm.value.to_i64().and_then(|value| n.checked_mul(value)) {
                    Some(value) => n = value,
                    None => return 0,
                }
            }
            n
        };
        let collect = |keys: &[(&str, &str)]| -> Vec<(std::string::String, i64)> {
            let mut out = Vec::new();
            for (thread_key, axis) in keys {
                let Some((_, _, extent)) = launch_param(&params, thread_key) else {
                    continue;
                };
                let extent = bound(thread_key).unwrap_or(extent);
                match as_i64(extent) {
                    Some(value) => out.push((axis.to_string(), value)),
                    None => return Vec::new(),
                }
            }
            out
        };
        let threads = product(&["threadIdx.x", "threadIdx.y", "threadIdx.z"]);
        if threads <= 0 {
            return Ok(false);
        }
        let (lanes, warps) = (threads.min(32), (threads / 32).max(1));
        let mut cluster_axes = collect(&[
            ("clusterCtaIdx.x", "cbx"),
            ("clusterCtaIdx.y", "cby"),
            ("clusterCtaIdx.z", "cbz"),
        ]);
        if !self.cluster_bounds.is_empty() && cluster_axes.is_empty() {
            return Ok(false);
        }
        while cluster_axes.len() > 1 && cluster_axes.last().is_some_and(|axis| axis.1 == 1) {
            cluster_axes.pop();
        }
        self.cluster_cta_axis_extents = cluster_axes.clone();
        let mut cta_axes = cluster_axes;
        if cta_axes.is_empty() {
            cta_axes = collect(&[("blockIdx.x", "bx"), ("blockIdx.y", "by"), ("blockIdx.z", "bz")]);
            self.cluster_cta_axis_extents.clear();
        }
        while cta_axes.len() > 1 && cta_axes.last().is_some_and(|axis| axis.1 == 1) {
            cta_axes.pop();
        }
        let mut ctas: i64 = 1;
        for (_, extent) in &cta_axes {
            if *extent <= 0 || *extent > i64::MAX / ctas {
                return Ok(false);
            }
            ctas *= extent;
        }
        if cta_axes.len() <= 1 {
            cta_axes.clear();
        }
        let active = ActiveSet::initial(lanes, warps, ctas, &cta_axes)?;
        if scope_switch(&active, ScopeKind::Thread)?.is_none() {
            bail!(INTERNAL_ERROR, "kernel-entry scope switch failed");
        }
        self.stack.push(ExecContext { active, kind: ScopeKind::Thread });
        Ok(true)
    }

    fn overlaps_cluster_cta(&self, target: &ScopeIdTarget) -> bool {
        target.binding == ScopeBinding::KernelCta && !self.cluster_cta_axis_extents.is_empty()
    }

    /// C++ `ResolveScopeIdTarget`: the CUDA index `expr` reads, directly or through a variable.
    fn resolve(&self, expr: &Expr) -> Result<Option<ScopeIdTarget>> {
        if let Some(cast) = expr.as_node::<CastObj>() {
            return self.resolve(&cast.value);
        }
        if let Some(call) = expr.as_node::<CallObj>() {
            let Some(op) = call.op.as_node::<OpObj>() else { return Ok(None) };
            let (binding, tag) = match op.name.as_str() {
                "tirx.cuda.block_idx" => (ScopeBinding::KernelCta, "blockIdx."),
                "tirx.cuda.cluster_cta_id" => (ScopeBinding::ClusterCta, "clusterCtaIdx."),
                "tirx.cuda.thread_idx" => (ScopeBinding::CtaThread, "threadIdx."),
                "tirx.cuda.cluster_id" => {
                    return Ok(Some(ScopeIdTarget {
                        binding: ScopeBinding::KernelCluster,
                        dim: 0,
                        ndim: 3,
                    }))
                }
                "tirx.cuda.warp_id" => return Ok(Some(ScopeIdTarget::flat(ScopeBinding::CtaWarp))),
                "tirx.cuda.warpgroup_id" => {
                    return Ok(Some(ScopeIdTarget::flat(ScopeBinding::CtaWarpgroup)))
                }
                "tirx.cuda.warp_in_warpgroup" => {
                    return Ok(Some(ScopeIdTarget::flat(ScopeBinding::WarpgroupWarp)))
                }
                "tirx.cuda.lane_id" => {
                    return Ok(Some(ScopeIdTarget::flat(ScopeBinding::WarpThread)))
                }
                "tirx.cuda.thread_in_warpgroup" => {
                    return Ok(Some(ScopeIdTarget::flat(ScopeBinding::WarpgroupThread)))
                }
                "tirx.cuda.cta_pair_id" => {
                    return Ok(Some(ScopeIdTarget::flat(ScopeBinding::ClusterCtaPair)))
                }
                "tirx.cuda.linear_thread_id" => {
                    return Ok(Some(ScopeIdTarget::flat(ScopeBinding::CtaThread)))
                }
                _ => return Ok(None),
            };
            let axis: StringImm = call.args.get(0)?.try_cast()?;
            let dim = axis_dim(axis.value.as_str());
            let mut ndim = 1;
            for i in 1..3 {
                let key = format!("{tag}{}", axis_letter(i));
                if let Some((_, _, extent)) = launch_param(&self.launch_params, &key) {
                    let extent = self.cluster_bound(&key).unwrap_or(extent);
                    if !is_const_int(extent, 1) {
                        ndim = i + 1;
                    }
                }
            }
            return Ok(Some(ScopeIdTarget { binding, dim, ndim: ndim.max(dim + 1) }));
        }
        let Some(_) = expr.as_node::<crate::ir::VarObj>() else { return Ok(None) };
        Ok(self.scope_aliases.iter().find(|(var, _)| var.same_as(expr)).map(|(_, target)| *target))
    }

    fn push(&mut self, context: Option<ExecContext>) -> bool {
        match context {
            Some(context) => {
                self.stack.push(context);
                true
            }
            None => false,
        }
    }

    fn top(&self) -> &ExecContext {
        self.stack.last().expect("checked to be non-empty")
    }

    fn push_range(&mut self, target: ScopeIdTarget, lo: i64, hi: i64) -> Result<bool> {
        if self.stack.is_empty() {
            return Ok(false);
        }
        if target.binding == ScopeBinding::ClusterCtaPair {
            if self.variable_cluster || hi != lo.wrapping_add(1) || !(0..=1).contains(&lo) {
                return Ok(false);
            }
            return self.push_cta_pair(lo);
        }
        if self.overlaps_cluster_cta(&target) {
            return Ok(false);
        }
        if target.ndim != 1 {
            let Some(axis) = cta_axis_name(&target) else { return Ok(false) };
            let context = self.top().with_cta_axis_filter(axis, lo, hi)?;
            return Ok(self.push(context));
        }
        let context = self.top().with_filter(target.binding, lo, hi)?;
        Ok(self.push(context))
    }

    fn push_modulo(&mut self, target: ScopeIdTarget, modulus: i64, residue: i64) -> Result<bool> {
        if self.stack.is_empty()
            || target.binding == ScopeBinding::ClusterCtaPair
            || self.overlaps_cluster_cta(&target)
        {
            return Ok(false);
        }
        if target.ndim != 1 {
            let Some(axis) = cta_axis_name(&target) else { return Ok(false) };
            let context = self.top().with_cta_axis_modulo(axis, modulus, residue)?;
            return Ok(self.push(context));
        }
        if matches!(target.binding, ScopeBinding::KernelCta | ScopeBinding::ClusterCta) {
            let context = self.top().with_cta_axis_modulo("cta_id", modulus, residue)?;
            return Ok(self.push(context));
        }
        Ok(false)
    }

    fn push_cta_pair(&mut self, value: i64) -> Result<bool> {
        if self.stack.is_empty() || self.cluster_cta_axis_extents.is_empty() {
            return Ok(false);
        }
        if self.cluster_cta_axis_extents.len() <= 1 {
            let context = self.top().with_cta_axis_modulo("cta_id", 2, value)?;
            return Ok(self.push(context));
        }
        let mut parity_axis: Option<std::string::String> = None;
        let (mut coeff, mut fixed) = (1i64, 0i64);
        for (axis, extent) in &self.cluster_cta_axis_extents {
            let Some(range) = self.top().active.get(axis) else { return Ok(false) };
            let Some((offset, active_extent, stride)) = range.ints() else { return Ok(false) };
            fixed += coeff * offset;
            if active_extent > 1 && (coeff * stride) % 2 != 0 {
                if parity_axis.is_some() {
                    return Ok(false);
                }
                parity_axis = Some(axis.clone());
            }
            coeff *= extent;
        }
        let mut residue = (value - fixed) % 2;
        if residue < 0 {
            residue += 2;
        }
        let Some(axis) = parity_axis else {
            if residue != 0 {
                return Ok(false);
            }
            let context = self.top().clone();
            self.stack.push(context);
            return Ok(true);
        };
        let context = self.top().with_cta_axis_modulo(&axis, 2, residue)?;
        Ok(self.push(context))
    }

    fn push_selector(&mut self, target: ScopeIdTarget, selector: Expr) -> Result<bool> {
        if self.stack.is_empty() || target.ndim != 1 || self.overlaps_cluster_cta(&target) {
            return Ok(false);
        }
        let context = self.top().with_selector(target.binding, selector)?;
        Ok(self.push(context))
    }

    /// C++ `TryExtractLinearScopeDiff`: `diff` as `coeff * v + base` with `coeff` = +/-1.
    fn linear_scope_diff(&self, diff: &Expr) -> Result<Option<(ScopeIdTarget, i64, i64)>> {
        let simplified = self.analyzer.simplify(diff, 2)?;
        for (var, target) in &self.scope_aliases {
            let linear = detect_linear_equation(&simplified, &Array::new(vec![var.clone()]))?;
            if linear.len() != 2 {
                continue;
            }
            let c = as_i64(&self.analyzer.simplify(&linear.get(0)?, 2)?);
            let b = as_i64(&self.analyzer.simplify(&linear.get(1)?, 2)?);
            let (Some(c), Some(b)) = (c, b) else { continue };
            if c != 1 && c != -1 {
                continue;
            }
            return Ok(Some((*target, c, b)));
        }
        Ok(None)
    }

    fn linear_compare_range(
        &self,
        lhs: &Expr,
        rhs: &Expr,
        inclusive: bool,
        less: bool,
    ) -> Result<Option<ScopeIdRange>> {
        let Some((target, coeff, base)) = self.linear_scope_diff(&(lhs - rhs).expr()?)? else {
            return Ok(None);
        };
        let (mut lo, mut hi) = (NEG_INF, POS_INF);
        match (less, coeff == 1) {
            (true, true) => {
                hi = if inclusive { 1i64.wrapping_sub(base) } else { base.wrapping_neg() }
            }
            (true, false) => lo = if inclusive { base } else { base.wrapping_add(1) },
            (false, true) => {
                lo = if inclusive { base.wrapping_neg() } else { 1i64.wrapping_sub(base) }
            }
            (false, false) => hi = if inclusive { base.wrapping_add(1) } else { base },
        }
        Ok(Some(ScopeIdRange { target, lo, hi }))
    }

    fn linear_equality_range(&self, lhs: &Expr, rhs: &Expr) -> Result<Option<ScopeIdRange>> {
        let Some((target, coeff, base)) = self.linear_scope_diff(&(lhs - rhs).expr()?)? else {
            return Ok(None);
        };
        let value = if coeff == 1 { base.wrapping_neg() } else { base };
        Ok(Some(ScopeIdRange { target, lo: value, hi: value.wrapping_add(1) }))
    }

    fn modulo_target(&self, expr: &Expr) -> Result<Option<(ScopeIdTarget, i64)>> {
        let (lhs, rhs) = if let Some(node) = expr.as_node::<ModObj>() {
            (&node.a, &node.b)
        } else if let Some(node) = expr.as_node::<FloorModObj>() {
            (&node.a, &node.b)
        } else {
            return Ok(None);
        };
        let Some(target) = self.resolve(lhs)? else { return Ok(None) };
        match as_i64(&self.analyzer.simplify(rhs, 2)?) {
            Some(modulus) if modulus > 0 => Ok(Some((target, modulus))),
            _ => Ok(None),
        }
    }

    fn push_modulo_equality(&mut self, lhs: &Expr, rhs: &Expr) -> Result<bool> {
        for (modulo, value) in [(lhs, rhs), (rhs, lhs)] {
            if let Some((target, modulus)) = self.modulo_target(modulo)? {
                if let Some(residue) = as_i64(&self.analyzer.simplify(value, 2)?) {
                    return self.push_modulo(target, modulus, residue);
                }
            }
        }
        Ok(false)
    }

    /// C++ `TryExtractComparisonRange`.
    fn comparison_range(&self, pred: &Expr) -> Result<Option<ScopeIdRange>> {
        match comparison(pred) {
            Some((CmpOp::Eq, a, b)) => self.linear_equality_range(a, b),
            Some((CmpOp::Lt, a, b)) => self.linear_compare_range(a, b, false, true),
            Some((CmpOp::Le, a, b)) => self.linear_compare_range(a, b, true, true),
            Some((CmpOp::Gt, a, b)) => self.linear_compare_range(a, b, false, false),
            Some((CmpOp::Ge, a, b)) => self.linear_compare_range(a, b, true, false),
            None => Ok(None),
        }
    }

    /// C++ `TryPushComparisonPredicate`.
    fn push_comparison(&mut self, pred: &Expr) -> Result<bool> {
        if let Some((CmpOp::Eq, a, b)) = comparison(pred) {
            if let Some(range) = self.linear_equality_range(a, b)? {
                if self.push_range(range.target, range.lo, range.hi)? {
                    return Ok(true);
                }
            }
            return self.push_modulo_equality(a, b);
        }
        match self.comparison_range(pred)? {
            Some(range) => self.push_range(range.target, range.lo, range.hi),
            None => Ok(false),
        }
    }

    fn push_filter_predicate(&mut self, call: &CallObj) -> Result<i32> {
        check_thread_filter_arity(call)?;
        let (var, cond) = (call.args.get(0)?, call.args.get(1)?);
        if let Some(target) = self.resolve(&var)? {
            if contains_call(&cond, &CUDA_ELECT_SYNC)? {
                let selector = self.selector(&var, cond.clone())?;
                let pushed = i32::from(self.push_selector(target, selector)?);
                return Ok(pushed + self.push_predicate_ctx(&cond)?);
            }
        }
        self.push_predicate_ctx(&cond)
    }

    /// `tirx.gpu_active_thread_selector(var, cond)`, typed as `var`.
    fn selector(&self, var: &Expr, cond: Expr) -> Result<Expr> {
        let op = GPU_ACTIVE_THREAD_SELECTOR.get()?;
        Ok(Call::new(&var.ty, op, vec![var.clone(), cond])?.into())
    }

    fn push_conjunctive(&mut self, pred: &Expr) -> Result<i32> {
        let mut terms = Vec::new();
        flatten_conjuncts(pred, &mut terms);
        let mut consumed = vec![false; terms.len()];
        // Ranges on the same target, intersected: (target, lo, hi, term indices).
        let mut groups: Vec<(ScopeIdTarget, i64, i64, Vec<usize>)> = Vec::new();
        let mut term_group: Vec<Option<usize>> = vec![None; terms.len()];
        for (i, term) in terms.iter().enumerate() {
            let Some(range) = self.comparison_range(term)? else { continue };
            match groups.iter().position(|group| group.0 == range.target) {
                Some(index) => {
                    let group = &mut groups[index];
                    group.1 = group.1.max(range.lo);
                    group.2 = group.2.min(range.hi);
                    group.3.push(i);
                    term_group[i] = Some(index);
                }
                None => {
                    groups.push((range.target, range.lo, range.hi, vec![i]));
                    term_group[i] = Some(groups.len() - 1);
                }
            }
        }
        let mut pushed = 0;
        let mut progress = true;
        while progress {
            progress = false;
            for i in 0..terms.len() {
                if consumed[i] {
                    continue;
                }
                if let Some(index) = term_group[i] {
                    let (target, lo, hi, indices) = groups[index].clone();
                    if (indices.len() > 1 && indices[0] != i) || lo >= hi {
                        continue;
                    }
                    if self.push_range(target, lo, hi)? {
                        for index in indices {
                            consumed[index] = true;
                        }
                        pushed += 1;
                        progress = true;
                    }
                    continue;
                }
                if self.push_comparison(&terms[i])? {
                    consumed[i] = true;
                    pushed += 1;
                    progress = true;
                }
            }
        }
        for i in 0..terms.len() {
            if consumed[i] {
                continue;
            }
            if term_group[i].is_some() {
                consumed[i] = true;
                continue;
            }
            pushed += self.push_predicate_ctx(&terms[i])?;
        }
        Ok(pushed)
    }

    fn lane_scope_var(&self) -> Option<Var> {
        self.scope_aliases
            .iter()
            .find(|(_, target)| target.binding == ScopeBinding::WarpThread)
            .map(|(var, _)| var.clone())
    }

    /// C++ `TryPushCanonicalCtx`: the frames pushed, or `None` when `cond` is not canonical.
    fn push_canonical(&mut self, cond: &Expr) -> Result<Option<i32>> {
        if self.stack.is_empty() {
            return Ok(None);
        }
        let mut resolve_error = None;
        let atoms = {
            let mut is_scope_id = |var: &Var| match self.resolve(&Expr::from(var)) {
                Ok(target) => target.is_some(),
                Err(error) => {
                    resolve_error.get_or_insert(error);
                    false
                }
            };
            classify_canonical(cond, &mut is_scope_id)
        };
        if let Some(error) = resolve_error {
            return Err(error);
        }
        let Some(atoms) = atoms else {
            if contains_call(cond, &CUDA_ELECT_SYNC)? {
                let Some(lane) = self.lane_scope_var() else { return Ok(None) };
                let selector = self.selector(&lane.into(), cond.clone())?;
                let target = ScopeIdTarget::flat(ScopeBinding::WarpThread);
                return Ok(Some(i32::from(self.push_selector(target, selector)?)));
            }
            return Ok(None);
        };
        let mut groups: Vec<(ScopeIdTarget, i64, i64)> = Vec::new();
        let mut elect = Vec::new();
        for atom in atoms {
            match atom {
                FilterAtom::ElectSync(call) => elect.push(call),
                FilterAtom::Range { var, lo, hi } => {
                    let Some(target) = self.resolve(&var.into())? else { continue };
                    match groups.iter_mut().find(|group| group.0 == target) {
                        Some(group) => {
                            group.1 = group.1.max(lo);
                            group.2 = group.2.min(hi);
                        }
                        None => groups.push((target, lo, hi)),
                    }
                }
            }
        }
        let mut consumed = vec![false; groups.len()];
        let mut pushed = 0;
        let mut progress = true;
        while progress {
            progress = false;
            for i in 0..groups.len() {
                if consumed[i] {
                    continue;
                }
                let (target, lo, hi) = groups[i];
                if lo >= hi {
                    consumed[i] = true;
                    continue;
                }
                if self.push_range(target, lo, hi)? {
                    consumed[i] = true;
                    pushed += 1;
                    progress = true;
                }
            }
        }
        for call in elect {
            let Some(lane) = self.lane_scope_var() else { continue };
            let selector = self.selector(&lane.into(), call)?;
            if self.push_selector(ScopeIdTarget::flat(ScopeBinding::WarpThread), selector)? {
                pushed += 1;
            }
        }
        Ok(Some(pushed))
    }

    /// C++ `PushPredicateCtx`.
    fn push_predicate_ctx(&mut self, pred: &Expr) -> Result<i32> {
        if self.stack.is_empty() {
            return Ok(0);
        }
        if pred.as_node::<AndObj>().is_some() || pred.as_node::<BitwiseAndObj>().is_some() {
            return self.push_conjunctive(pred);
        }
        if let Some(call) = pred.as_node::<CallObj>() {
            if GPU_THREAD_FILTER.is(&call.op) {
                return self.push_filter_predicate(call);
            }
        }
        Ok(i32::from(self.push_comparison(pred)?))
    }

    /// C++ `ExecAnalysis::Bind`: a variable bound to a CUDA index reads that index.
    pub(crate) fn bind(&mut self, var: &Var, value: &Expr) -> Result<()> {
        if let Some(target) = self.resolve(value)? {
            match self.scope_aliases.iter_mut().find(|(alias, _)| alias.same_as(var)) {
                Some(entry) => entry.1 = target,
                None => self.scope_aliases.push((var.clone(), target)),
            }
        }
        Ok(())
    }

    /// C++ `ExecAnalysis::PushPredicate`: the number of frames pushed for `cond`.
    pub(crate) fn push_predicate(&mut self, cond: &Expr) -> Result<i32> {
        match self.push_canonical(cond)? {
            Some(count) => Ok(count),
            None => self.push_predicate_ctx(cond),
        }
    }

    pub(crate) fn pop(&mut self, count: i32) {
        for _ in 0..count {
            self.stack.pop();
        }
    }

    /// C++ `ExecAnalysis::RewritePredicate`: each `gpu_thread_filter(var, cond)` becomes `cond`
    /// as a bool.
    pub(crate) fn rewrite_predicate(&self, pred: &Expr) -> Result<Expr> {
        self.rewrite_filter_calls(pred)
    }

    fn rewrite_filter_calls(&self, pred: &Expr) -> Result<Expr> {
        if let Some((a, b, rebuild)) = filter_binary(pred) {
            let (new_a, new_b) = (self.rewrite_filter_calls(a)?, self.rewrite_filter_calls(b)?);
            if new_a.same_as(a) && new_b.same_as(b) {
                return Ok(pred.clone());
            }
            return rebuild(new_a, new_b).expr();
        }
        if let Some(node) = pred.as_node::<BitwiseNotObj>() {
            let a = self.rewrite_filter_calls(&node.a)?;
            if a.same_as(&node.a) {
                return Ok(pred.clone());
            }
            return Ok(BitwiseNot::new_full(a, &node.loc)?.into());
        }
        if let Some(call) = pred.as_node::<CallObj>() {
            if GPU_THREAD_FILTER.is(&call.op) {
                check_thread_filter_arity(call)?;
                return self.rewrite_filter_calls(&as_bool(&call.args.get(1)?)?);
            }
            let mut changed = false;
            let mut args = Vec::with_capacity(call.args.len());
            for arg in call.args.iter() {
                let new_arg = if arg.prim_dtype().is_ok() {
                    self.rewrite_filter_calls(&arg)?
                } else {
                    arg.clone()
                };
                changed |= !new_arg.same_as(&arg);
                args.push(new_arg);
            }
            if changed {
                let call = Call::new_full(
                    Some(call.ty.clone()),
                    call.op.clone(),
                    Array::new(args),
                    call.attrs.clone(),
                    Array::new(vec![]),
                    &call.loc,
                )?;
                return Ok(call.into());
            }
        }
        Ok(pred.clone())
    }

    /// C++ `ExecAnalysis::Split`: the `(inter, intra)` maps for a tile instruction at `scope`.
    pub(crate) fn split(&self, scope: &ExecScope) -> Result<(SplitMap, SplitMap)> {
        check!(
            !self.variable_cluster || scope.name.as_str() != "cluster",
            VALUE_ERROR,
            "!impl_->variable_cluster_ || scope->name() != \"cluster\"",
            "Cluster tensor instructions require one static cluster shape; use explicit CUDA/PTX \
             instructions for variable or preferred cluster shapes",
        );
        let empty = SplitMap::new;
        let Some(context) = self.stack.last() else { return Ok((empty(), empty())) };
        // C++ logs a warning when the split fails.
        match scope_switch(&context.active, scope_kind(scope.name.as_str())?)? {
            Some((inter, intra)) => Ok((encode_side(&inter)?, encode_side(&intra)?)),
            None => Ok((empty(), empty())),
        }
    }
}

/// How [`ExecAnalysis::rewrite_predicate`] rebuilds a binary node over rewritten operands.
type Rebuild = fn(Expr, Expr) -> Ex;

/// The operands of a node `rewrite_predicate` descends into: `&&`, shifts and bitwise ops.
fn filter_binary(pred: &Expr) -> Option<(&Expr, &Expr, Rebuild)> {
    if let Some(n) = pred.as_node::<AndObj>() {
        return Some((&n.a, &n.b, |a, b| a.logical_and(b)));
    }
    if let Some(n) = pred.as_node::<LShiftObj>() {
        return Some((&n.a, &n.b, |a, b| a << b));
    }
    if let Some(n) = pred.as_node::<RShiftObj>() {
        return Some((&n.a, &n.b, |a, b| a >> b));
    }
    if let Some(n) = pred.as_node::<BitwiseAndObj>() {
        return Some((&n.a, &n.b, |a, b| a & b));
    }
    if let Some(n) = pred.as_node::<BitwiseOrObj>() {
        return Some((&n.a, &n.b, |a, b| a | b));
    }
    if let Some(n) = pred.as_node::<BitwiseXorObj>() {
        return Some((&n.a, &n.b, |a, b| a ^ b));
    }
    None
}

fn flatten_conjuncts(pred: &Expr, out: &mut Vec<Expr>) {
    if let Some(node) = pred.as_node::<AndObj>() {
        flatten_conjuncts(&node.a, out);
        flatten_conjuncts(&node.b, out);
    } else if let Some(node) = pred.as_node::<BitwiseAndObj>() {
        flatten_conjuncts(&node.a, out);
        flatten_conjuncts(&node.b, out);
    } else {
        out.push(pred.clone());
    }
}

/// C++ `AsBool`: a bool itself, anything else `!= 0`.
fn as_bool(pred: &Expr) -> Result<Expr> {
    let dtype = pred.prim_dtype()?;
    if dtype.code == tvm_ffi::DLDataTypeCode::kDLBool as u8 {
        return Ok(pred.clone());
    }
    pred.not_equal(int_imm(dtype, 0)?).expr()
}

fn cta_axis_name(target: &ScopeIdTarget) -> Option<&'static str> {
    if !(0..3).contains(&target.dim) {
        return None;
    }
    match target.binding {
        ScopeBinding::KernelCta => Some(["bx", "by", "bz"][target.dim as usize]),
        ScopeBinding::ClusterCta => Some(["cbx", "cby", "cbz"][target.dim as usize]),
        _ => None,
    }
}

fn axis_letter(axis: i32) -> char {
    (b'x' + axis as u8) as char
}

/// `value.c_str()[0] - 'x'`.
fn axis_dim(value: &str) -> i32 {
    value.as_bytes().first().map_or(0, |c| *c as i32) - b'x' as i32
}

// ---------------------------------------------------------------------------------------------
// LowerIndices (`CudaIndexLowerer`)

/// The CUDA special register `name` (`ctaid.x`, `tid.y`, ...) as an int32 call.
fn register(name: &str) -> Result<Expr> {
    let int32_ty = prim_type(int32_dtype())?;
    let op = if name.starts_with("ctaid.") {
        &CUDA_BLOCK_IDX
    } else if name.starts_with("tid.") {
        &CUDA_THREAD_IDX
    } else if name.starts_with("ntid.") {
        &CUDA_BLOCK_DIM
    } else if name.starts_with("nctaid.") {
        &CUDA_GRID_DIM
    } else {
        let args = vec![int32(32)?.into(), StringImm::new(name)?.into()];
        return Ok(Call::new(int32_ty, CUDA_MOV_SREG.get()?, args)?.into());
    };
    let axis = StringImm::new(&name[name.len() - 1..])?;
    Ok(Call::new(int32_ty, op.get()?, vec![axis.into()])?.into())
}

struct CudaIndexLowerer {
    params: LaunchParams,
    remap: Vec<(ObjectIdentity, Expr)>,
    warp: Var,
    needs_warp: bool,
}

#[tvm_ffi::dispatch(mutate)]
impl CudaIndexLowerer {
    /// A launch variable is its hardware index.
    fn mutate_var(&mut self, var: Var, m: &mut tvm_ffi::Mutator) -> Result<tvm_ffi::Any> {
        let identity = ObjectIdentity::of(&var);
        if let Some((_, value)) = self.remap.iter().find(|(key, _)| *key == identity) {
            return Ok(value.to_any());
        }
        m.default_mutate(self, &var)
    }

    fn mutate_call(&mut self, call: Call, m: &mut tvm_ffi::Mutator) -> Result<tvm_ffi::Any> {
        let Some(op) = call.op.as_node::<OpObj>() else { return m.default_mutate(self, &call) };
        let Some(name) = op.name.as_str().strip_prefix("tirx.cuda.") else {
            return m.default_mutate(self, &call);
        };
        let axis = || -> Result<i32> {
            let axis: StringImm = call.args.get(0)?.try_cast()?;
            Ok(axis_dim(axis.value.as_str()))
        };
        let requires_cluster = |what: &str| -> Result<()> {
            match launch_param(&self.params, "clusterCtaIdx.x") {
                Some(_) => Ok(()),
                None => Err(error!(
                    VALUE_ERROR,
                    "Check failed: (params_.count(\"clusterCtaIdx.x\")) is false: {what} requires \
                     an explicit launch cluster"
                )),
            }
        };
        let lowered: Expr = match name {
            "block_idx" => self.axis("blockIdx.", axis()?)?,
            "thread_idx" => self.axis("threadIdx.", axis()?)?,
            "cluster_cta_id" => {
                requires_cluster("cluster_cta_id")?;
                self.axis("clusterCtaIdx.", axis()?)?
            }
            "cluster_id" | "grid_dim" | "block_dim" | "cluster_dim" => {
                if name == "cluster_id" || name == "cluster_dim" {
                    requires_cluster(name)?;
                }
                let reg = match name {
                    "cluster_id" => "clusterid.",
                    "grid_dim" => "nctaid.",
                    "block_dim" => "ntid.",
                    _ => "cluster_nctaid.",
                };
                register(&format!("{reg}{}", axis_letter(axis()?)))?
            }
            "linear_thread_id" => self.linear_thread()?,
            "lane_id" => FloorMod::new(self.linear_thread()?, 32)?.into(),
            "thread_in_warpgroup" => FloorMod::new(self.linear_thread()?, 128)?.into(),
            "warp_id" | "warpgroup_id" | "warp_in_warpgroup" => {
                self.needs_warp = true;
                match name {
                    "warp_id" => Expr::from(&self.warp),
                    "warpgroup_id" => FloorDiv::new(&self.warp, 4)?.into(),
                    _ => FloorMod::new(&self.warp, 4)?.into(),
                }
            }
            "cta_pair_id" => {
                requires_cluster("cta_pair_id")?;
                let rank = (crate::ops::ex(self.axis("clusterCtaIdx.", 0)?)
                    + crate::ops::ex(register("cluster_nctaid.x")?)
                        * (crate::ops::ex(self.axis("clusterCtaIdx.", 1)?)
                            + crate::ops::ex(register("cluster_nctaid.y")?)
                                * self.axis("clusterCtaIdx.", 2)?))
                .expr()?;
                FloorMod::new(rank, 2)?.into()
            }
            _ => return m.default_mutate(self, &call),
        };
        Ok(lowered.into())
    }
}

impl CudaIndexLowerer {
    /// C++ `Axis(tag, axis)`: the index register, or `0` for an absent or unit dimension.
    fn axis(&self, tag: &str, axis: i32) -> Result<Expr> {
        let Some((_, _, extent)) =
            launch_param(&self.params, &format!("{tag}{}", axis_letter(axis)))
        else {
            return Ok(int32(0)?.into());
        };
        // An explicit singleton cluster still needs a hardware index when preferred clusters
        // are enabled, so cluster coordinates are not folded.
        if tag != "clusterCtaIdx." && is_const_int(extent, 1) {
            return Ok(int32(0)?.into());
        }
        let reg = match tag {
            "blockIdx." => "ctaid.",
            "threadIdx." => "tid.",
            _ => "cluster_ctaid.",
        };
        register(&format!("{reg}{}", axis_letter(axis)))
    }

    fn block_dimension(&self, axis: i32) -> Result<Expr> {
        if let Some((_, _, extent)) =
            launch_param(&self.params, &format!("threadIdx.{}", axis_letter(axis)))
        {
            if extent.as_node::<IntImmObj>().is_some() {
                return Ok(extent.clone());
            }
        }
        register(&format!("ntid.{}", axis_letter(axis)))
    }

    fn linear_thread(&self) -> Result<Expr> {
        let thread = crate::ops::ex(self.axis("threadIdx.", 0)?)
            + crate::ops::ex(self.block_dimension(0)?)
                * (crate::ops::ex(self.axis("threadIdx.", 1)?)
                    + crate::ops::ex(self.block_dimension(1)?) * self.axis("threadIdx.", 2)?);
        Analyzer::new()?.simplify(&thread.expr()?, 2)
    }
}

/// C++ `backend::cuda::LowerIndices`: CUDA index calls and launch variables as hardware
/// registers, with the warp index computed once at the front when used.
pub(crate) fn lower_indices(stmt: &Stmt, params: &LaunchParams) -> Result<Stmt> {
    let mut remap = Vec::new();
    for (tag, var, _) in params {
        let reg = if let Some(axis) = tag.strip_prefix("blockIdx.") {
            format!("ctaid.{axis}")
        } else if let Some(axis) = tag.strip_prefix("threadIdx.") {
            format!("tid.{axis}")
        } else if let Some(axis) = tag.strip_prefix("clusterCtaIdx.") {
            format!("cluster_ctaid.{axis}")
        } else {
            continue;
        };
        remap.push((ObjectIdentity::of(var), register(&reg)?));
    }
    let warp = Var::prim("warp_id_in_cta", int32_dtype())?;
    let mut lowerer = CudaIndexLowerer { params: params.clone(), remap, warp, needs_warp: false };
    let result: Stmt = tvm_ffi::structural_mutate(stmt.clone(), &mut lowerer)?.try_into()?;
    let mut prefix: Vec<Stmt> = Vec::new();
    if lowerer.needs_warp {
        let mut threads: i64 = 1;
        for axis in 0..3 {
            let extent = lowerer.block_dimension(axis)?;
            let Some(imm) = extent.as_node::<IntImmObj>() else {
                bail!(
                    VALUE_ERROR,
                    "Check failed: (extent) is false: Warp indices require a static block size"
                );
            };
            let dimension = imm.value.to_i64();
            let Some(dimension) = dimension.filter(|&d| d > 0 && d <= 1024 / threads) else {
                bail!(
                    VALUE_ERROR,
                    "Check failed: (dimension && *dimension > 0 && *dimension <= 1024 / threads) \
                     is false: Block size must be positive and contain at most 1024 threads"
                );
            };
            threads *= dimension;
        }
        check!(
            threads % 32 == 0,
            VALUE_ERROR,
            "threads % 32 == 0",
            "Warp indices require a block size divisible by 32",
        );
        let uint32 = tvm_ffi::DLDataType { code: 1, bits: 32, lanes: 1 };
        let shuffle = Call::new(
            prim_type(int32_dtype())?,
            GPU_WARP_SHUFFLE.get()?,
            vec![
                int_imm(uint32, 0xffff_ffff)?.into(),
                FloorDiv::new(lowerer.linear_thread()?, 32)?.into(),
                int32(0)?.into(),
                int32(32)?.into(),
                int32(32)?.into(),
            ],
        )?;
        prefix.push(Bind::new(lowerer.warp, shuffle)?.into());
    }
    prefix.push(result);
    Ok(SeqStmt::new(prefix)?.into())
}
