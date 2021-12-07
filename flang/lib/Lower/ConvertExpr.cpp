//===-- ConvertExpr.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertExpr.h"
#include "flang/Common/default-kinds.h"
#include "flang/Common/unwrap.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/real.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/BuiltinModules.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/Coarray.h"
#include "flang/Lower/ComponentPath.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/DumpEvaluateExpr.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/Factory.h"
#include "flang/Optimizer/Builder/Runtime/Character.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Ragged.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

#define DEBUG_TYPE "flang-lower-expr"

//===----------------------------------------------------------------------===//
// The composition and structure of Fortran::evaluate::Expr is defined in
// the various header files in include/flang/Evaluate. You are referred
// there for more information on these data structures. Generally speaking,
// these data structures are a strongly typed family of abstract data types
// that, composed as trees, describe the syntax of Fortran expressions.
//
// This part of the bridge can traverse these tree structures and lower them
// to the correct FIR representation in SSA form.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool> generateArrayCoordinate(
    "gen-array-coor",
    llvm::cl::desc("in lowering create ArrayCoorOp instead of CoordinateOp"),
    llvm::cl::init(false));

// The default attempts to balance a modest allocation size with expected user
// input to minimize bounds checks and reallocations during dynamic array
// construction. Some user codes may have very large array constructors for
// which the default can be increased.
static llvm::cl::opt<unsigned> clInitialBufferSize(
    "array-constructor-initial-buffer-size",
    llvm::cl::desc(
        "set the incremental array construction buffer size (default=32)"),
    llvm::cl::init(32u));

/// The various semantics of a program constituent (or a part thereof) as it may
/// appear in an expression.
///
/// Given the following Fortran declarations.
/// ```fortran
///   REAL :: v1, v2, v3
///   REAL, POINTER :: vp1
///   REAL :: a1(c), a2(c)
///   REAL ELEMENTAL FUNCTION f1(arg) ! array -> array
///   FUNCTION f2(arg)                ! array -> array
///   vp1 => v3       ! 1
///   v1 = v2 * vp1   ! 2
///   a1 = a1 + a2    ! 3
///   a1 = f1(a2)     ! 4
///   a1 = f2(a2)     ! 5
/// ```
///
/// In line 1, `vp1` is a BoxAddr to copy a box value into. The box value is
/// constructed from the DataAddr of `v3`.
/// In line 2, `v1` is a DataAddr to copy a value into. The value is constructed
/// from the DataValue of `v2` and `vp1`. DataValue is implicitly a double
/// dereference in the `vp1` case.
/// In line 3, `a1` and `a2` on the rhs are RefTransparent. The `a1` on the lhs
/// is CopyInCopyOut as `a1` is replaced elementally by the additions.
/// In line 4, `a2` can be RefTransparent, ByValueArg, RefOpaque, or BoxAddr if
/// `arg` is declared as C-like pass-by-value, VALUE, INTENT(?), or ALLOCATABLE/
/// POINTER, respectively. `a1` on the lhs is CopyInCopyOut.
///  In line 5, `a2` may be DataAddr or BoxAddr assuming f2 is transformational.
///  `a1` on the lhs is again CopyInCopyOut.
enum class ConstituentSemantics {
  // Scalar data reference semantics.
  //
  // For these let `v` be the location in memory of a variable with value `x`
  DataValue, // refers to the value `x`
  DataAddr,  // refers to the address `v`
  BoxValue,  // refers to a box value containing `v`
  BoxAddr,   // refers to the address of a box value containing `v`

  // Array data reference semantics.
  //
  // For these let `a` be the location in memory of a sequence of value `[xs]`.
  // Let `x_i` be the `i`-th value in the sequence `[xs]`.

  // Referentially transparent. Refers to the array's value, `[xs]`.
  RefTransparent,
  // Refers to an ephemeral address `tmp` containing value `x_i` (15.5.2.3.p7
  // note 2). (Passing a copy by reference to simulate pass-by-value.)
  ByValueArg,
  // Refers to the merge of array value `[xs]` with another array value `[ys]`.
  // This merged array value will be written into memory location `a`.
  CopyInCopyOut,
  // Similar to CopyInCopyOut but `a` may be a transient projection (rather than
  // a whole array).
  ProjectedCopyInCopyOut,
  // Similar to ProjectedCopyInCopyOut, except the merge value is not assigned
  // automatically by the framework. Instead, and address for `[xs]` is made
  // accessible so that custom assignments to `[xs]` can be implemented.
  CustomCopyInCopyOut,
  // Referentially opaque. Refers to the address of `x_i`.
  RefOpaque
};

/// Convert parser's INTEGER relational operators to MLIR.  TODO: using
/// unordered, but we may want to cons ordered in certain situation.
static mlir::arith::CmpIPredicate
translateRelational(Fortran::common::RelationalOperator rop) {
  switch (rop) {
  case Fortran::common::RelationalOperator::LT:
    return mlir::arith::CmpIPredicate::slt;
  case Fortran::common::RelationalOperator::LE:
    return mlir::arith::CmpIPredicate::sle;
  case Fortran::common::RelationalOperator::EQ:
    return mlir::arith::CmpIPredicate::eq;
  case Fortran::common::RelationalOperator::NE:
    return mlir::arith::CmpIPredicate::ne;
  case Fortran::common::RelationalOperator::GT:
    return mlir::arith::CmpIPredicate::sgt;
  case Fortran::common::RelationalOperator::GE:
    return mlir::arith::CmpIPredicate::sge;
  }
  llvm_unreachable("unhandled INTEGER relational operator");
}

/// Convert parser's REAL relational operators to MLIR.
/// The choice of order (O prefix) vs unorder (U prefix) follows Fortran 2018
/// requirements in the IEEE context (table 17.1 of F2018). This choice is
/// also applied in other contexts because it is easier and in line with
/// other Fortran compilers.
/// FIXME: The signaling/quiet aspect of the table 17.1 requirement is not
/// fully enforced. FIR and LLVM `fcmp` instructions do not give any guarantee
/// whether the comparison will signal or not in case of quiet NaN argument.
static mlir::arith::CmpFPredicate
translateFloatRelational(Fortran::common::RelationalOperator rop) {
  switch (rop) {
  case Fortran::common::RelationalOperator::LT:
    return mlir::arith::CmpFPredicate::OLT;
  case Fortran::common::RelationalOperator::LE:
    return mlir::arith::CmpFPredicate::OLE;
  case Fortran::common::RelationalOperator::EQ:
    return mlir::arith::CmpFPredicate::OEQ;
  case Fortran::common::RelationalOperator::NE:
    return mlir::arith::CmpFPredicate::UNE;
  case Fortran::common::RelationalOperator::GT:
    return mlir::arith::CmpFPredicate::OGT;
  case Fortran::common::RelationalOperator::GE:
    return mlir::arith::CmpFPredicate::OGE;
  }
  llvm_unreachable("unhandled REAL relational operator");
}

/// Lower `opt` (from front-end shape analysis) to MLIR. If `opt` is `nullopt`
/// then issue an error.
static mlir::Value
convertOptExtentExpr(Fortran::lower::AbstractConverter &converter,
                     Fortran::lower::StatementContext &stmtCtx,
                     const Fortran::evaluate::MaybeExtentExpr &opt) {
  auto loc = converter.getCurrentLocation();
  if (!opt.has_value())
    fir::emitFatalError(loc, "shape analysis failed to return an expression");
  auto e = toEvExpr(*opt);
  return fir::getBase(converter.genExprValue(&e, stmtCtx, loc));
}

/// Does this expr designate an allocatable or pointer entity ?
static bool
isAllocatableOrPointer(const Fortran::lower::SomeExpr &expr,
                       Fortran::lower::AbstractConverter &converter) {
  // First check if this is a pointer function result.
  if (Fortran::evaluate::IsObjectPointer(expr, converter.getFoldingContext()))
    return true;
  const auto *sym =
      Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(expr);
  return sym && Fortran::semantics::IsAllocatableOrPointer(*sym);
}

/// Convert the array_load, `load`, to an extended value. If `path` is not
/// empty, then traverse through the components designated. The base value is
/// `newBase`. This does not accept an array_load with a slice operand.
static fir::ExtendedValue
arrayLoadExtValue(fir::FirOpBuilder &builder, mlir::Location loc,
                  fir::ArrayLoadOp load, llvm::ArrayRef<mlir::Value> path,
                  mlir::Value newBase, mlir::Value newLen = {}) {
  // Recover the extended value from the load.
  assert(!load.slice() && "slice is not allowed");
  auto arrTy = load.getType();
  if (!path.empty()) {
    auto ty = fir::applyPathToType(arrTy, path);
    if (!ty)
      fir::emitFatalError(loc, "path does not apply to type");
    if (!ty.isa<fir::SequenceType>()) {
      if (fir::isa_char(ty)) {
        auto len = newLen;
        if (!len)
          len = fir::factory::CharacterExprHelper{builder, loc}.getLength(
              load.memref());
        if (!len) {
          assert(load.typeparams().size() == 1 &&
                 "length must be in array_load");
          len = load.typeparams()[0];
        }
        return fir::CharBoxValue{newBase, len};
      }
      return newBase;
    }
    arrTy = ty.cast<fir::SequenceType>();
  }

  // Use the shape op, if there is one.
  mlir::Value shapeVal = load.shape();
  if (shapeVal) {
    if (!mlir::isa<fir::ShiftOp>(shapeVal.getDefiningOp())) {
      mlir::Type eleTy = fir::unwrapSequenceType(arrTy);
      std::vector<mlir::Value> extents = fir::factory::getExtents(shapeVal);
      std::vector<mlir::Value> origins = fir::factory::getOrigins(shapeVal);
      if (fir::isa_char(eleTy)) {
        mlir::Value len = newLen;
        if (!len)
          len = fir::factory::CharacterExprHelper{builder, loc}.getLength(
              load.memref());
        if (!len) {
          assert(load.typeparams().size() == 1 &&
                 "length must be in array_load");
          len = load.typeparams()[0];
        }
        return fir::CharArrayBoxValue(newBase, len, extents, origins);
      }
      return fir::ArrayBoxValue(newBase, extents, origins);
    }
    if (!fir::isa_box_type(load.memref().getType()))
      fir::emitFatalError(loc, "shift op is invalid in this context");
  }

  // There is no shape or the array is in a box. Extents and lower bounds must
  // be read at runtime.
  if (path.empty() && !shapeVal) {
    fir::ExtendedValue exv =
        fir::factory::readBoxValue(builder, loc, load.memref());
    return fir::substBase(exv, newBase);
  }
  TODO(loc, "component is boxed, retreive its type parameters");
}

/// Is this a call to an elemental procedure with at least one array argument ?
static bool
isElementalProcWithArrayArgs(const Fortran::evaluate::ProcedureRef &procRef) {
  if (procRef.IsElemental())
    for (const auto &arg : procRef.arguments())
      if (arg && arg->Rank() != 0)
        return true;
  return false;
}
template <typename T>
static bool isElementalProcWithArrayArgs(const Fortran::evaluate::Expr<T> &) {
  return false;
}
template <>
bool isElementalProcWithArrayArgs(const Fortran::lower::SomeExpr &x) {
  if (const auto *procRef = std::get_if<Fortran::evaluate::ProcedureRef>(&x.u))
    return isElementalProcWithArrayArgs(*procRef);
  return false;
}

namespace {

/// Lowering of Fortran::evaluate::Expr<T> expressions
class ScalarExprLowering {
public:
  using ExtValue = fir::ExtendedValue;

  explicit ScalarExprLowering(mlir::Location loc,
                              Fortran::lower::AbstractConverter &converter,
                              Fortran::lower::SymMap &symMap,
                              Fortran::lower::StatementContext &stmtCtx,
                              bool initializer = false)
      : location{loc}, converter{converter},
        builder{converter.getFirOpBuilder()}, stmtCtx{stmtCtx}, symMap{symMap},
        inInitializer{initializer} {}

  ExtValue genExtAddr(const Fortran::lower::SomeExpr &expr) {
    return gen(expr);
  }

  /// Lower `expr` to be passed as a fir.box argument. Do not create a temp
  /// for the expr if it is a variable that can be described as a fir.box.
  ExtValue genBoxArg(const Fortran::lower::SomeExpr &expr) {
    bool saveUseBoxArg = useBoxArg;
    useBoxArg = true;
    auto result = gen(expr);
    useBoxArg = saveUseBoxArg;
    return result;
  }

  ExtValue genExtValue(const Fortran::lower::SomeExpr &expr) {
    return genval(expr);
  }

  /// Lower an expression that is a pointer or an allocatable to a
  /// MutableBoxValue.
  fir::MutableBoxValue
  genMutableBoxValue(const Fortran::lower::SomeExpr &expr) {
    // Pointers and allocatables can only be:
    //    - a simple designator "x"
    //    - a component designator "a%b(i,j)%x"
    //    - a function reference "foo()"
    //    - result of NULL() or NULL(MOLD) intrinsic.
    //    NULL() requires some context to be lowered, so it is not handled
    //    here and must be lowered according to the context where it appears.
    auto exv = std::visit(
        [&](const auto &x) { return genMutableBoxValueImpl(x); }, expr.u);
    auto *mutableBox = exv.getBoxOf<fir::MutableBoxValue>();
    if (!mutableBox)
      fir::emitFatalError(getLoc(), "expr was not lowered to MutableBoxValue");
    return *mutableBox;
  }

  template <typename T>
  ExtValue genMutableBoxValueImpl(const T &) {
    // NULL() case should not be handled here.
    fir::emitFatalError(getLoc(), "NULL() must be lowered in its context");
  }

  template <typename T>
  ExtValue
  genMutableBoxValueImpl(const Fortran::evaluate::FunctionRef<T> &funRef) {
    return genRawProcedureRef(funRef, converter.genType(toEvExpr(funRef)));
  }

  template <typename T>
  ExtValue
  genMutableBoxValueImpl(const Fortran::evaluate::Designator<T> &designator) {
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::SymbolRef &sym) -> ExtValue {
              return symMap.lookupSymbol(*sym).toExtendedValue();
            },
            [&](const Fortran::evaluate::Component &comp) -> ExtValue {
              return genComponent(comp);
            },
            [&](const auto &) -> ExtValue {
              fir::emitFatalError(getLoc(),
                                  "not an allocatable or pointer designator");
            }},
        designator.u);
  }

  template <typename T>
  ExtValue genMutableBoxValueImpl(const Fortran::evaluate::Expr<T> &expr) {
    return std::visit([&](const auto &x) { return genMutableBoxValueImpl(x); },
                      expr.u);
  }

  mlir::Location getLoc() { return location; }

  template <typename A>
  mlir::Value genunbox(const A &expr) {
    auto e = genval(expr);
    if (auto *r = e.getUnboxed())
      return *r;
    fir::emitFatalError(getLoc(), "unboxed expression expected");
  }

  /// Generate an integral constant of `value`
  template <int KIND>
  mlir::Value genIntegerConstant(mlir::MLIRContext *context,
                                 std::int64_t value) {
    auto type = converter.genType(Fortran::common::TypeCategory::Integer, KIND);
    return builder.createIntegerConstant(getLoc(), type, value);
  }

  /// Generate a logical/boolean constant of `value`
  mlir::Value genBoolConstant(bool value) {
    return builder.createBool(getLoc(), value);
  }

  /// Generate a real constant with a value `value`.
  template <int KIND>
  mlir::Value genRealConstant(mlir::MLIRContext *context,
                              const llvm::APFloat &value) {
    auto fltTy = Fortran::lower::convertReal(context, KIND);
    return builder.createRealConstant(getLoc(), fltTy, value);
  }

  mlir::Type getSomeKindInteger() { return builder.getIndexType(); }

  mlir::FuncOp getFunction(llvm::StringRef name, mlir::FunctionType funTy) {
    if (auto func = builder.getNamedFunction(name))
      return func;
    return builder.createFunction(getLoc(), name, funTy);
  }

  template <typename OpTy>
  mlir::Value createCompareOp(mlir::arith::CmpIPredicate pred,
                              const ExtValue &left, const ExtValue &right) {
    if (auto *lhs = left.getUnboxed())
      if (auto *rhs = right.getUnboxed())
        return builder.create<OpTy>(getLoc(), pred, *lhs, *rhs);
    fir::emitFatalError(getLoc(), "array compare should be handled in genarr");
  }
  template <typename OpTy, typename A>
  mlir::Value createCompareOp(const A &ex, mlir::arith::CmpIPredicate pred) {
    auto left = genval(ex.left());
    return createCompareOp<OpTy>(pred, left, genval(ex.right()));
  }

  template <typename OpTy>
  mlir::Value createFltCmpOp(mlir::arith::CmpFPredicate pred,
                             const ExtValue &left, const ExtValue &right) {
    if (auto *lhs = left.getUnboxed())
      if (auto *rhs = right.getUnboxed())
        return builder.create<OpTy>(getLoc(), pred, *lhs, *rhs);
    fir::emitFatalError(getLoc(), "array compare should be handled in genarr");
  }
  template <typename OpTy, typename A>
  mlir::Value createFltCmpOp(const A &ex, mlir::arith::CmpFPredicate pred) {
    auto left = genval(ex.left());
    return createFltCmpOp<OpTy>(pred, left, genval(ex.right()));
  }

  /// Create a call to the runtime to compare two CHARACTER values.
  /// Precondition: This assumes that the two values have `fir.boxchar` type.
  mlir::Value createCharCompare(mlir::arith::CmpIPredicate pred,
                                const ExtValue &left, const ExtValue &right) {
    return fir::runtime::genCharCompare(builder, getLoc(), pred, left, right);
  }

  template <typename A>
  mlir::Value createCharCompare(const A &ex, mlir::arith::CmpIPredicate pred) {
    auto left = genval(ex.left());
    return createCharCompare(pred, left, genval(ex.right()));
  }

  /// Returns a reference to a symbol or its box/boxChar descriptor if it has
  /// one.
  ExtValue gen(Fortran::semantics::SymbolRef sym) {
    if (auto val = symMap.lookupSymbol(sym))
      return val.match(
          [&](const Fortran::lower::SymbolBox::PointerOrAllocatable &boxAddr) {
            return fir::factory::genMutableBoxRead(builder, getLoc(), boxAddr);
          },
          [&val](auto &) { return val.toExtendedValue(); });
    LLVM_DEBUG(llvm::dbgs()
               << "unknown symbol: " << sym << "\nmap: " << symMap << '\n');
    fir::emitFatalError(getLoc(), "symbol is not mapped to any IR value");
  }

  /// Generate a load of a value from an address.
  ExtValue genLoad(const ExtValue &addr) {
    auto loc = getLoc();
    return addr.match(
        [](const fir::CharBoxValue &box) -> ExtValue { return box; },
        [&](const fir::UnboxedValue &v) -> ExtValue {
          if (fir::unwrapRefType(fir::getBase(v).getType())
                  .isa<fir::RecordType>())
            return v;
          return builder.create<fir::LoadOp>(loc, fir::getBase(v));
        },
        [&](const auto &v) -> ExtValue {
          TODO(getLoc(), "loading array or descriptor");
        });
  }

  ExtValue genval(Fortran::semantics::SymbolRef sym) {
    auto loc = getLoc();
    auto var = gen(sym);
    if (auto *s = var.getUnboxed())
      if (fir::isReferenceLike(s->getType())) {
        // A function with multiple entry points returning different types
        // tags all result variables with one of the largest types to allow
        // them to share the same storage.  A reference to a result variable
        // of one of the other types requires conversion to the actual type.
        auto addr = *s;
        if (Fortran::semantics::IsFunctionResult(sym)) {
          auto resultType = converter.genType(*sym);
          if (addr.getType() != resultType)
            addr = builder.createConvert(loc, builder.getRefType(resultType),
                                         addr);
        }
        return genLoad(addr);
      }
    return var;
  }

  ExtValue genval(const Fortran::evaluate::BOZLiteralConstant &) {
    TODO(getLoc(), "BOZ");
  }

  /// Return indirection to function designated in ProcedureDesignator.
  /// The type of the function indirection is not guaranteed to match the one
  /// of the ProcedureDesignator due to Fortran implicit typing rules.
  ExtValue genval(const Fortran::evaluate::ProcedureDesignator &proc) {
    if (const auto *intrinsic = proc.GetSpecificIntrinsic()) {
      auto signature = Fortran::lower::translateSignature(proc, converter);
      // Intrinsic lowering is based on the generic name, so retrieve it here in
      // case it is different from the specific name. The type of the specific
      // intrinsic is retained in the signature.
      auto genericName =
          converter.getFoldingContext().intrinsics().GetGenericIntrinsicName(
              intrinsic->name);
      auto symbolRefAttr =
          Fortran::lower::getUnrestrictedIntrinsicSymbolRefAttr(
              builder, getLoc(), genericName, signature);
      mlir::Value funcPtr =
          builder.create<fir::AddrOfOp>(getLoc(), signature, symbolRefAttr);
      return funcPtr;
    }
    const auto *symbol = proc.GetSymbol();
    assert(symbol && "expected symbol in ProcedureDesignator");
    if (Fortran::semantics::IsDummy(*symbol)) {
      auto val = symMap.lookupSymbol(*symbol);
      assert(val && "Dummy procedure not in symbol map");
      return val.getAddr();
    }
    auto name = converter.mangleName(*symbol);
    auto func = Fortran::lower::getOrDeclareFunction(name, proc, converter);
    mlir::Value funcPtr = builder.create<fir::AddrOfOp>(
        getLoc(), func.getType(), builder.getSymbolRefAttr(name));
    return funcPtr;
  }
  ExtValue genval(const Fortran::evaluate::NullPointer &) {
    return builder.createNullConstant(getLoc());
  }

  static bool
  isDerivedTypeWithLengthParameters(const Fortran::semantics::Symbol &sym) {
    if (const auto *declTy = sym.GetType())
      if (const auto *derived = declTy->AsDerived())
        return Fortran::semantics::CountLenParameters(*derived) > 0;
    return false;
  }

  static bool isBuiltinCPtr(const Fortran::semantics::Symbol &sym) {
    if (const auto *declType = sym.GetType())
      if (const auto *derived = declType->AsDerived())
        return Fortran::semantics::IsIsoCType(derived);
    return false;
  }

  /// Lower structure constructor without a temporary. This can be used in
  /// fir::GloablOp, and assumes that the structure component is a constant.
  ExtValue genStructComponentInInitializer(
      const Fortran::evaluate::StructureConstructor &ctor) {
    auto loc = getLoc();
    auto ty = translateSomeExprToFIRType(converter, toEvExpr(ctor));
    auto recTy = ty.cast<fir::RecordType>();
    auto fieldTy = fir::FieldType::get(ty.getContext());
    mlir::Value res = builder.create<fir::UndefOp>(loc, recTy);

    for (auto [sym, expr] : ctor.values()) {
      // Parent components need more work because they do not appear in the
      // fir.rec type.
      if (sym->test(Fortran::semantics::Symbol::Flag::ParentComp))
        TODO(loc, "parent component in structure constructor");

      auto name = toStringRef(sym->name());
      auto componentTy = recTy.getType(name);
      // FIXME: type parameters must come from the derived-type-spec
      auto field = builder.create<fir::FieldIndexOp>(
          loc, fieldTy, name, ty,
          /*typeParams=*/mlir::ValueRange{} /*TODO*/);

      if (Fortran::semantics::IsAllocatable(sym))
        TODO(loc, "allocatable component in structure constructor");

      if (Fortran::semantics::IsPointer(sym)) {
        auto initialTarget = Fortran::lower::genInitialDataTarget(
            converter, loc, componentTy, expr.value());
        res = builder.create<fir::InsertValueOp>(
            loc, recTy, res, initialTarget,
            builder.getArrayAttr(field.getAttributes()));
        continue;
      }

      if (isDerivedTypeWithLengthParameters(sym))
        TODO(loc, "component with length parameters in structure constructor");

      if (isBuiltinCPtr(sym)) {
        // Builtin c_ptr and c_funptr have special handling because initial
        // value are handled for them as an extension.
        auto addr = Fortran::lower::genExtAddrInInitializer(converter, loc,
                                                            expr.value());
        auto baseAddr = fir::getBase(addr);
        auto undef = builder.create<fir::UndefOp>(loc, componentTy);
        auto cPtrRecTy = componentTy.dyn_cast<fir::RecordType>();
        assert(cPtrRecTy && "c_ptr and c_funptr must be derived types");
        llvm::StringRef addrFieldName = Fortran::lower::builtin::cptrFieldName;
        auto addrFieldTy = cPtrRecTy.getType(addrFieldName);
        auto addrField = builder.create<fir::FieldIndexOp>(
            loc, fieldTy, addrFieldName, componentTy,
            /*typeParams=*/mlir::ValueRange{});
        auto castAddr = builder.createConvert(loc, addrFieldTy, baseAddr);
        auto val = builder.create<fir::InsertValueOp>(
            loc, componentTy, undef, castAddr,
            builder.getArrayAttr(addrField.getAttributes()));
        res = builder.create<fir::InsertValueOp>(
            loc, recTy, res, val, builder.getArrayAttr(field.getAttributes()));
        continue;
      }

      auto val = fir::getBase(genval(expr.value()));
      assert(!fir::isa_ref_type(val.getType()) && "expecting a constant value");
      auto castVal = builder.createConvert(loc, componentTy, val);
      res = builder.create<fir::InsertValueOp>(
          loc, recTy, res, castVal,
          builder.getArrayAttr(field.getAttributes()));
    }
    return res;
  }

  /// A structure constructor is lowered two ways. In an initializer context,
  /// the entire structure must be constant, so the aggregate value is
  /// constructed inline. This allows it to be the body of a GlobalOp.
  /// Otherwise, the structure constructor is in an expression. In that case, a
  /// temporary object is constructed in the stack frame of the procedure.
  ExtValue genval(const Fortran::evaluate::StructureConstructor &ctor) {
    if (inInitializer)
      return genStructComponentInInitializer(ctor);
    auto loc = getLoc();
    auto ty = translateSomeExprToFIRType(converter, toEvExpr(ctor));
    auto recTy = ty.cast<fir::RecordType>();
    auto fieldTy = fir::FieldType::get(ty.getContext());
    mlir::Value res = builder.createTemporary(loc, recTy);

    for (auto value : ctor.values()) {
      const auto &sym = value.first;
      auto &expr = value.second;
      // Parent components need more work because they do not appear in the
      // fir.rec type.
      if (sym->test(Fortran::semantics::Symbol::Flag::ParentComp))
        TODO(loc, "parent component in structure constructor");

      if (isDerivedTypeWithLengthParameters(sym))
        TODO(loc, "component with length parameters in structure constructor");

      auto name = toStringRef(sym->name());
      // FIXME: type parameters must come from the derived-type-spec
      mlir::Value field = builder.create<fir::FieldIndexOp>(
          loc, fieldTy, name, ty,
          /*typeParams=*/mlir::ValueRange{} /*TODO*/);
      auto coorTy = builder.getRefType(recTy.getType(name));
      auto coor = builder.create<fir::CoordinateOp>(loc, coorTy,
                                                    fir::getBase(res), field);
      auto to = fir::factory::componentToExtendedValue(builder, loc, coor);
      to.match(
          [&](const fir::UnboxedValue &toPtr) {
            ExtValue value = genval(expr.value());
            fir::factory::genScalarAssignment(builder, loc, to, value);
          },
          [&](const fir::CharBoxValue &) {
            ExtValue value = genval(expr.value());
            fir::factory::genScalarAssignment(builder, loc, to, value);
          },
          [&](const fir::ArrayBoxValue &) {
            Fortran::lower::createSomeArrayAssignment(
                converter, to, expr.value(), symMap, stmtCtx);
          },
          [&](const fir::CharArrayBoxValue &) {
            Fortran::lower::createSomeArrayAssignment(
                converter, to, expr.value(), symMap, stmtCtx);
          },
          [&](const fir::BoxValue &toBox) {
            fir::emitFatalError(loc, "derived type components must not be "
                                     "represented by fir::BoxValue");
          },
          [&](const fir::MutableBoxValue &toBox) {
            if (toBox.isPointer()) {
              Fortran::lower::associateMutableBox(
                  converter, loc, toBox, expr.value(), /*lbounds=*/llvm::None,
                  stmtCtx);
              return;
            }
            // For allocatable components, a deep copy is needed.
            TODO(loc, "allocatable components in derived type assignment");
          },
          [&](const fir::ProcBoxValue &toBox) {
            TODO(loc, "procedure pointer component in derived type assignment");
          });
    }
    return res;
  }

  /// Lowering of an <i>ac-do-variable</i>, which is not a Symbol.
  ExtValue genval(const Fortran::evaluate::ImpliedDoIndex &var) {
    return converter.impliedDoBinding(toStringRef(var.name));
  }

  ExtValue genval(const Fortran::evaluate::DescriptorInquiry &desc) {
    auto exv = desc.base().IsSymbol() ? gen(desc.base().GetLastSymbol())
                                      : gen(desc.base().GetComponent());
    auto idxTy = builder.getIndexType();
    auto loc = getLoc();
    auto castResult = [&](mlir::Value v) {
      using ResTy = Fortran::evaluate::DescriptorInquiry::Result;
      return builder.createConvert(
          loc, converter.genType(ResTy::category, ResTy::kind), v);
    };
    switch (desc.field()) {
    case Fortran::evaluate::DescriptorInquiry::Field::Len:
      return castResult(fir::factory::readCharLen(builder, loc, exv));
    case Fortran::evaluate::DescriptorInquiry::Field::LowerBound:
      return castResult(fir::factory::readLowerBound(
          builder, loc, exv, desc.dimension(),
          builder.createIntegerConstant(loc, idxTy, 1)));
    case Fortran::evaluate::DescriptorInquiry::Field::Extent:
      return castResult(
          fir::factory::readExtent(builder, loc, exv, desc.dimension()));
    case Fortran::evaluate::DescriptorInquiry::Field::Rank:
      TODO(loc, "rank inquiry on assumed rank");
    case Fortran::evaluate::DescriptorInquiry::Field::Stride:
      // So far the front end does not generate this inquiry.
      TODO(loc, "Stride inquiry");
    }
    llvm_unreachable("unknown descriptor inquiry");
  }

  ExtValue genval(const Fortran::evaluate::TypeParamInquiry &) {
    TODO(getLoc(), "type parameter inquiry");
  }

  mlir::Value extractComplexPart(mlir::Value cplx, bool isImagPart) {
    return fir::factory::Complex{builder, getLoc()}.extractComplexPart(
        cplx, isImagPart);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::ComplexComponent<KIND> &part) {
    return extractComplexPart(genunbox(part.left()), part.isImaginaryPart);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, KIND>> &op) {
    auto input = genunbox(op.left());
    // Like LLVM, integer negation is the binary op "0 - value"
    auto zero = genIntegerConstant<KIND>(builder.getContext(), 0);
    return builder.create<mlir::arith::SubIOp>(getLoc(), zero, input);
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Real, KIND>> &op) {
    return builder.create<mlir::arith::NegFOp>(getLoc(), genunbox(op.left()));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Complex, KIND>> &op) {
    return builder.create<fir::NegcOp>(getLoc(), genunbox(op.left()));
  }

  template <typename OpTy>
  mlir::Value createBinaryOp(const ExtValue &left, const ExtValue &right) {
    assert(fir::isUnboxedValue(left) && fir::isUnboxedValue(right));
    auto lhs = fir::getBase(left);
    auto rhs = fir::getBase(right);
    assert(lhs.getType() == rhs.getType() && "types must be the same");
    return builder.create<OpTy>(getLoc(), lhs, rhs);
  }

  template <typename OpTy, typename A>
  mlir::Value createBinaryOp(const A &ex) {
    auto left = genval(ex.left());
    return createBinaryOp<OpTy>(left, genval(ex.right()));
  }

#undef GENBIN
#define GENBIN(GenBinEvOp, GenBinTyCat, GenBinFirOp)                           \
  template <int KIND>                                                          \
  ExtValue genval(const Fortran::evaluate::GenBinEvOp<Fortran::evaluate::Type< \
                      Fortran::common::TypeCategory::GenBinTyCat, KIND>> &x) { \
    return createBinaryOp<GenBinFirOp>(x);                                     \
  }

  GENBIN(Add, Integer, mlir::arith::AddIOp)
  GENBIN(Add, Real, mlir::arith::AddFOp)
  GENBIN(Add, Complex, fir::AddcOp)
  GENBIN(Subtract, Integer, mlir::arith::SubIOp)
  GENBIN(Subtract, Real, mlir::arith::SubFOp)
  GENBIN(Subtract, Complex, fir::SubcOp)
  GENBIN(Multiply, Integer, mlir::arith::MulIOp)
  GENBIN(Multiply, Real, mlir::arith::MulFOp)
  GENBIN(Multiply, Complex, fir::MulcOp)
  GENBIN(Divide, Integer, mlir::arith::DivSIOp)
  GENBIN(Divide, Real, mlir::arith::DivFOp)
  GENBIN(Divide, Complex, fir::DivcOp)

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genval(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &op) {
    auto ty = converter.genType(TC, KIND);
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    return Fortran::lower::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genval(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &op) {
    auto ty = converter.genType(TC, KIND);
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    return Fortran::lower::genPow(builder, getLoc(), ty, lhs, rhs);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::ComplexConstructor<KIND> &op) {
    return fir::factory::Complex{builder, getLoc()}.createComplex(
        KIND, genunbox(op.left()), genunbox(op.right()));
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Concat<KIND> &op) {
    auto lhs = genval(op.left());
    auto rhs = genval(op.right());
    auto *lhsChar = lhs.getCharBox();
    auto *rhsChar = rhs.getCharBox();
    if (lhsChar && rhsChar)
      return fir::factory::CharacterExprHelper{builder, getLoc()}
          .createConcatenate(*lhsChar, *rhsChar);
    TODO(getLoc(), "character array concatenate");
  }

  /// MIN and MAX operations
  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    switch (op.ordering) {
    case Fortran::evaluate::Ordering::Greater:
      return Fortran::lower::genMax(builder, getLoc(),
                                    llvm::ArrayRef<mlir::Value>{lhs, rhs});
    case Fortran::evaluate::Ordering::Less:
      return Fortran::lower::genMin(builder, getLoc(),
                                    llvm::ArrayRef<mlir::Value>{lhs, rhs});
    case Fortran::evaluate::Ordering::Equal:
      llvm_unreachable("Equal is not a valid ordering in this context");
    }
    llvm_unreachable("unknown ordering");
  }

  // Change the dynamic length information without actually changing the
  // underlying character storage.
  fir::ExtendedValue
  replaceScalarCharacterLength(const fir::ExtendedValue &scalarChar,
                               mlir::Value newLenValue) {
    mlir::Location loc = getLoc();
    const fir::CharBoxValue *charBox = scalarChar.getCharBox();
    if (!charBox)
      fir::emitFatalError(loc, "expected scalar character");
    mlir::Value charAddr = charBox->getAddr();
    auto charType =
        fir::unwrapPassByRefType(charAddr.getType()).cast<fir::CharacterType>();
    if (charType.hasConstantLen()) {
      // Erase previous constant length from the base type.
      fir::CharacterType::LenType newLen = fir::CharacterType::unknownLen();
      mlir::Type newCharTy = fir::CharacterType::get(
          builder.getContext(), charType.getFKind(), newLen);
      mlir::Type newType = fir::ReferenceType::get(newCharTy);
      charAddr = builder.createConvert(loc, newType, charAddr);
      return fir::CharBoxValue{charAddr, newLenValue};
    }
    return fir::CharBoxValue{charAddr, newLenValue};
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::SetLength<KIND> &x) {
    mlir::Value newLenValue = genunbox(x.right());
    fir::ExtendedValue lhs = gen(x.left());
    return replaceScalarCharacterLength(lhs, newLenValue);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, KIND>> &op) {
    return createCompareOp<mlir::arith::CmpIOp>(op,
                                                translateRelational(op.opr));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Real, KIND>> &op) {
    return createFltCmpOp<mlir::arith::CmpFOp>(
        op, translateFloatRelational(op.opr));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Complex, KIND>> &op) {
    return createFltCmpOp<fir::CmpcOp>(op, translateFloatRelational(op.opr));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Character, KIND>> &op) {
    return createCharCompare(op, translateRelational(op.opr));
  }

  ExtValue
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &op) {
    return std::visit([&](const auto &x) { return genval(x); }, op.u);
  }

  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  ExtValue
  genval(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>,
                                          TC2> &convert) {
    auto ty = converter.genType(TC1, KIND);
    auto operand = genunbox(convert.left());
    return builder.convertWithSemantics(getLoc(), ty, operand);
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::Parentheses<A> &op) {
    auto input = genval(op.left());
    auto base = fir::getBase(input);
    mlir::Value newBase =
        builder.create<fir::NoReassocOp>(getLoc(), base.getType(), base);
    return fir::substBase(input, newBase);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Not<KIND> &op) {
    auto logical = genunbox(op.left());
    auto one = genBoolConstant(true);
    auto val = builder.createConvert(getLoc(), builder.getI1Type(), logical);
    return builder.create<mlir::arith::XOrIOp>(getLoc(), val, one);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::LogicalOperation<KIND> &op) {
    auto i1Type = builder.getI1Type();
    auto slhs = genunbox(op.left());
    auto srhs = genunbox(op.right());
    auto lhs = builder.createConvert(getLoc(), i1Type, slhs);
    auto rhs = builder.createConvert(getLoc(), i1Type, srhs);
    switch (op.logicalOperator) {
    case Fortran::evaluate::LogicalOperator::And:
      return createBinaryOp<mlir::arith::AndIOp>(lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Or:
      return createBinaryOp<mlir::arith::OrIOp>(lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Eqv:
      return createCompareOp<mlir::arith::CmpIOp>(
          mlir::arith::CmpIPredicate::eq, lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Neqv:
      return createCompareOp<mlir::arith::CmpIOp>(
          mlir::arith::CmpIPredicate::ne, lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Not:
      // lib/evaluate expression for .NOT. is Fortran::evaluate::Not<KIND>.
      llvm_unreachable(".NOT. is not a binary operator");
    }
    llvm_unreachable("unhandled logical operation");
  }

  /// Convert a scalar literal constant to IR.
  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genScalarLit(
      const Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>
          &value) {
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      return genIntegerConstant<KIND>(builder.getContext(), value.ToInt64());
    } else if constexpr (TC == Fortran::common::TypeCategory::Logical) {
      return genBoolConstant(value.IsTrue());
    } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
      std::string str = value.DumpHexadecimal();
      if constexpr (KIND == 2) {
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEhalf(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 3) {
        llvm::APFloat floatVal{llvm::APFloatBase::BFloat(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 4) {
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEsingle(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 10) {
        llvm::APFloat floatVal{llvm::APFloatBase::x87DoubleExtended(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 16) {
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEquad(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else {
        // convert everything else to double
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEdouble(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      }
    } else if constexpr (TC == Fortran::common::TypeCategory::Complex) {
      using TR =
          Fortran::evaluate::Type<Fortran::common::TypeCategory::Real, KIND>;
      Fortran::evaluate::ComplexConstructor<KIND> ctor(
          Fortran::evaluate::Expr<TR>{
              Fortran::evaluate::Constant<TR>{value.REAL()}},
          Fortran::evaluate::Expr<TR>{
              Fortran::evaluate::Constant<TR>{value.AIMAG()}});
      return genunbox(ctor);
    } else /*constexpr*/ {
      llvm_unreachable("unhandled constant");
    }
  }
  /// Convert a ascii scalar literal CHARACTER to IR. (specialization)
  ExtValue
  genAsciiScalarLit(const Fortran::evaluate::Scalar<Fortran::evaluate::Type<
                        Fortran::common::TypeCategory::Character, 1>> &value,
                    int64_t len) {
    assert(value.size() == static_cast<std::uint64_t>(len));
    // Outline character constant in ro data if it is not in an initializer.
    if (!inInitializer)
      return fir::factory::createStringLiteral(builder, getLoc(), value);
    // When in an initializer context, construct the literal op itself and do
    // not construct another constant object in rodata.
    auto stringLit = builder.createStringLitOp(getLoc(), value);
    auto lenp = builder.createIntegerConstant(
        getLoc(), builder.getCharacterLengthType(), len);
    return fir::CharBoxValue{stringLit.getResult(), lenp};
  }
  /// Convert a non ascii scalar literal CHARACTER to IR. (specialization)
  template <int KIND>
  ExtValue
  genScalarLit(const Fortran::evaluate::Scalar<Fortran::evaluate::Type<
                   Fortran::common::TypeCategory::Character, KIND>> &value,
               int64_t len) {
    using ET = typename std::decay_t<decltype(value)>::value_type;
    if constexpr (KIND == 1) {
      return genAsciiScalarLit(value, len);
    }
    auto type = fir::CharacterType::get(builder.getContext(), KIND, len);
    auto consLit = [&]() -> fir::StringLitOp {
      auto context = builder.getContext();
      std::int64_t size = static_cast<std::int64_t>(value.size());
      auto shape = mlir::VectorType::get(
          llvm::ArrayRef<std::int64_t>{size},
          mlir::IntegerType::get(builder.getContext(), sizeof(ET) * 8));
      auto strAttr = mlir::DenseElementsAttr::get(
          shape, llvm::ArrayRef<ET>{value.data(), value.size()});
      auto valTag = mlir::Identifier::get(fir::StringLitOp::value(), context);
      mlir::NamedAttribute dataAttr(valTag, strAttr);
      auto sizeTag = mlir::Identifier::get(fir::StringLitOp::size(), context);
      mlir::NamedAttribute sizeAttr(sizeTag, builder.getI64IntegerAttr(len));
      llvm::SmallVector<mlir::NamedAttribute> attrs{dataAttr, sizeAttr};
      return builder.create<fir::StringLitOp>(
          getLoc(), llvm::ArrayRef<mlir::Type>{type}, llvm::None, attrs);
    };

    auto lenp = builder.createIntegerConstant(
        getLoc(), builder.getCharacterLengthType(), len);
    // When in an initializer context, construct the literal op itself and do
    // not construct another constant object in rodata.
    if (inInitializer)
      return fir::CharBoxValue{consLit().getResult(), lenp};

    // Otherwise, the string is in a plain old expression so "outline" the value
    // by hashconsing it to a constant literal object.

    // FIXME: For wider char types, lowering ought to use an array of i16 or
    // i32. But for now, lowering just fakes that the string value is a range of
    // i8 to get it past the C++ compiler.
    std::string globalName =
        fir::factory::uniqueCGIdent("cl", (const char *)value.c_str());
    auto global = builder.getNamedGlobal(globalName);
    if (!global)
      global = builder.createGlobalConstant(
          getLoc(), type, globalName,
          [&](fir::FirOpBuilder &builder) {
            auto str = consLit();
            builder.create<fir::HasValueOp>(getLoc(), str);
          },
          builder.createLinkOnceLinkage());
    auto addr = builder.create<fir::AddrOfOp>(getLoc(), global.resultType(),
                                              global.getSymbol());
    return fir::CharBoxValue{addr, lenp};
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genArrayLit(
      const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
          &con) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto size = Fortran::evaluate::GetSize(con.shape());
    fir::SequenceType::Shape shape(con.shape().begin(), con.shape().end());
    mlir::Type eleTy;
    if constexpr (TC == Fortran::common::TypeCategory::Character)
      eleTy = converter.genType(TC, KIND, {con.LEN()});
    else
      eleTy = converter.genType(TC, KIND);
    auto arrayTy = fir::SequenceType::get(shape, eleTy);
    mlir::Value array = builder.create<fir::UndefOp>(loc, arrayTy);
    llvm::SmallVector<mlir::Value> lbounds;
    llvm::SmallVector<mlir::Value> extents;
    for (auto [lb, extent] : llvm::zip(con.lbounds(), shape)) {
      lbounds.push_back(builder.createIntegerConstant(loc, idxTy, lb - 1));
      extents.push_back(builder.createIntegerConstant(loc, idxTy, extent));
    }
    if (size == 0) {
      if constexpr (TC == Fortran::common::TypeCategory::Character) {
        auto len = builder.createIntegerConstant(loc, idxTy, con.LEN());
        return fir::CharArrayBoxValue{array, len, extents, lbounds};
      } else {
        return fir::ArrayBoxValue{array, extents, lbounds};
      }
    }
    Fortran::evaluate::ConstantSubscripts subscripts = con.lbounds();
    auto createIdx = [&]() {
      llvm::SmallVector<mlir::Attribute> idx;
      for (size_t i = 0; i < subscripts.size(); ++i)
        idx.push_back(
            builder.getIntegerAttr(idxTy, subscripts[i] - con.lbounds()[i]));
      return idx;
    };
    if constexpr (TC == Fortran::common::TypeCategory::Character) {
      do {
        auto elementVal =
            fir::getBase(genScalarLit<KIND>(con.At(subscripts), con.LEN()));
        array = builder.create<fir::InsertValueOp>(
            loc, arrayTy, array, elementVal, builder.getArrayAttr(createIdx()));
      } while (con.IncrementSubscripts(subscripts));
      auto len = builder.createIntegerConstant(loc, idxTy, con.LEN());
      return fir::CharArrayBoxValue{array, len, extents, lbounds};
    } else {
      llvm::SmallVector<mlir::Attribute> rangeStartIdx;
      uint64_t rangeSize = 0;
      do {
        auto getElementVal = [&]() {
          return builder.createConvert(
              loc, eleTy,
              fir::getBase(genScalarLit<TC, KIND>(con.At(subscripts))));
        };
        auto nextSubscripts = subscripts;
        bool nextIsSame = con.IncrementSubscripts(nextSubscripts) &&
                          con.At(subscripts) == con.At(nextSubscripts);
        if (!rangeSize && !nextIsSame) { // single (non-range) value
          array = builder.create<fir::InsertValueOp>(
              loc, arrayTy, array, getElementVal(),
              builder.getArrayAttr(createIdx()));
        } else if (!rangeSize) { // start a range
          rangeStartIdx = createIdx();
          rangeSize = 1;
        } else if (nextIsSame) { // expand a range
          ++rangeSize;
        } else { // end a range
          llvm::SmallVector<mlir::Attribute> rangeBounds;
          auto idx = createIdx();
          for (size_t i = 0; i < idx.size(); ++i) {
            rangeBounds.push_back(rangeStartIdx[i]);
            rangeBounds.push_back(idx[i]);
          }
          array = builder.create<fir::InsertOnRangeOp>(
              loc, arrayTy, array, getElementVal(),
              builder.getArrayAttr(rangeBounds));
          rangeSize = 0;
        }
      } while (con.IncrementSubscripts(subscripts));
      return fir::ArrayBoxValue{array, extents, lbounds};
    }
  }

  fir::ExtendedValue genArrayLit(
      const Fortran::evaluate::Constant<Fortran::evaluate::SomeDerived> &con) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto size = Fortran::evaluate::GetSize(con.shape());
    fir::SequenceType::Shape shape(con.shape().begin(), con.shape().end());
    auto eleTy = converter.genType(con.GetType().GetDerivedTypeSpec());
    auto arrayTy = fir::SequenceType::get(shape, eleTy);
    mlir::Value array = builder.create<fir::UndefOp>(loc, arrayTy);
    llvm::SmallVector<mlir::Value> lbounds;
    llvm::SmallVector<mlir::Value> extents;
    for (auto [lb, extent] : llvm::zip(con.lbounds(), con.shape())) {
      lbounds.push_back(builder.createIntegerConstant(loc, idxTy, lb - 1));
      extents.push_back(builder.createIntegerConstant(loc, idxTy, extent));
    }
    if (size == 0)
      return fir::ArrayBoxValue{array, extents, lbounds};
    Fortran::evaluate::ConstantSubscripts subscripts = con.lbounds();
    do {
      auto derivedVal = fir::getBase(genval(con.At(subscripts)));
      llvm::SmallVector<mlir::Attribute> idx;
      for (auto [dim, lb] : llvm::zip(subscripts, con.lbounds()))
        idx.push_back(builder.getIntegerAttr(idxTy, dim - lb));
      array = builder.create<fir::InsertValueOp>(
          loc, arrayTy, array, derivedVal, builder.getArrayAttr(idx));
    } while (con.IncrementSubscripts(subscripts));
    return fir::ArrayBoxValue{array, extents, lbounds};
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
             &con) {
    if (con.Rank() > 0)
      return genArrayLit(con);
    auto opt = con.GetScalarValue();
    assert(opt.has_value() && "constant has no value");
    if constexpr (TC == Fortran::common::TypeCategory::Character) {
      return genScalarLit<KIND>(opt.value(), con.LEN());
    } else {
      return genScalarLit<TC, KIND>(opt.value());
    }
  }
  fir::ExtendedValue genval(
      const Fortran::evaluate::Constant<Fortran::evaluate::SomeDerived> &con) {
    if (con.Rank() > 0)
      return genArrayLit(con);
    if (auto ctor = con.GetScalarValue())
      return genval(ctor.value());
    fir::emitFatalError(getLoc(),
                        "constant of derived type has no constructor");
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::ArrayConstructor<A> &) {
    fir::emitFatalError(getLoc(),
                        "array constructor: lowering should not reach here");
  }

  ExtValue gen(const Fortran::evaluate::ComplexPart &x) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto exv = gen(x.complex());
    auto base = fir::getBase(exv);
    fir::factory::Complex helper{builder, loc};
    auto eleTy =
        helper.getComplexPartType(fir::dyn_cast_ptrEleTy(base.getType()));
    auto offset = builder.createIntegerConstant(
        loc, idxTy,
        x.part() == Fortran::evaluate::ComplexPart::Part::RE ? 0 : 1);
    mlir::Value result = builder.create<fir::CoordinateOp>(
        loc, builder.getRefType(eleTy), base, mlir::ValueRange{offset});
    return {result};
  }
  ExtValue genval(const Fortran::evaluate::ComplexPart &x) {
    return genLoad(gen(x));
  }

  /// Reference to a substring.
  ExtValue gen(const Fortran::evaluate::Substring &s) {
    // Get base string
    auto baseString = std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::DataRef &x) { return gen(x); },
            [&](const Fortran::evaluate::StaticDataObject::Pointer &p)
                -> ExtValue {
              if (auto str = p->AsString())
                return fir::factory::createStringLiteral(builder, getLoc(),
                                                         *str);
              // TODO: convert StaticDataObject to Constant<T> and use normal
              // constant path. Beware that StaticDataObject data() takes into
              // account build machine endianness.
              TODO(getLoc(),
                   "StaticDataObject::Pointer substring with kind > 1");
            },
        },
        s.parent());
    llvm::SmallVector<mlir::Value> bounds;
    auto lower = genunbox(s.lower());
    bounds.push_back(lower);
    if (auto upperBound = s.upper()) {
      auto upper = genunbox(*upperBound);
      bounds.push_back(upper);
    }
    fir::factory::CharacterExprHelper charHelper{builder, getLoc()};
    return baseString.match(
        [&](const fir::CharBoxValue &x) -> ExtValue {
          return charHelper.createSubstring(x, bounds);
        },
        [&](const fir::CharArrayBoxValue &) -> ExtValue {
          fir::emitFatalError(
              getLoc(),
              "array substring should be handled in array expression");
        },
        [&](const auto &) -> ExtValue {
          fir::emitFatalError(getLoc(), "substring base is not a CharBox");
        });
  }

  /// The value of a substring.
  ExtValue genval(const Fortran::evaluate::Substring &ss) {
    // FIXME: why is the value of a substring being lowered the same as the
    // address of a substring?
    return gen(ss);
  }

  ExtValue genval(const Fortran::evaluate::Subscript &subs) {
    if (auto *s = std::get_if<Fortran::evaluate::IndirectSubscriptIntegerExpr>(
            &subs.u)) {
      if (s->value().Rank() > 0)
        fir::emitFatalError(getLoc(), "vector subscript is not scalar");
      return {genval(s->value())};
    }
    fir::emitFatalError(getLoc(), "subscript triple notation is not scalar");
  }
  ExtValue genSubscript(const Fortran::evaluate::Subscript &subs) {
    return genval(subs);
  }

  ExtValue gen(const Fortran::evaluate::DataRef &dref) {
    return std::visit([&](const auto &x) { return gen(x); }, dref.u);
  }
  ExtValue genval(const Fortran::evaluate::DataRef &dref) {
    return std::visit([&](const auto &x) { return genval(x); }, dref.u);
  }

  // Helper function to turn the left-recursive Component structure into a list
  // that does not contain allocatable or pointer components other than the last
  // one.
  // Returns the object used as the base coordinate for the component chain.
  static Fortran::evaluate::DataRef const *
  reverseComponents(const Fortran::evaluate::Component &cmpt,
                    std::list<const Fortran::evaluate::Component *> &list) {
    list.push_front(&cmpt);
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::Component &x) {
              // Stop the list when a component is an allocatable or pointer
              // because the component cannot be lowered into a single
              // fir.coordinate_of.
              if (Fortran::semantics::IsAllocatableOrPointer(x.GetLastSymbol()))
                return &cmpt.base();
              return reverseComponents(x, list);
            },
            [&](auto &) { return &cmpt.base(); },
        },
        cmpt.base().u);
  }

  // Return the coordinate of the component reference
  ExtValue genComponent(const Fortran::evaluate::Component &cmpt) {
    std::list<const Fortran::evaluate::Component *> list;
    auto *base = reverseComponents(cmpt, list);
    llvm::SmallVector<mlir::Value> coorArgs;
    auto obj = gen(*base);
    auto ty = fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(obj).getType());
    auto loc = getLoc();
    auto fldTy = fir::FieldType::get(&converter.getMLIRContext());
    // FIXME: need to thread the LEN type parameters here.
    for (auto *field : list) {
      auto recTy = ty.cast<fir::RecordType>();
      const auto *sym = &field->GetLastSymbol();
      auto name = toStringRef(sym->name());
      coorArgs.push_back(builder.create<fir::FieldIndexOp>(
          loc, fldTy, name, recTy, fir::getTypeParams(obj)));
      ty = recTy.getType(name);
    }
    ty = builder.getRefType(ty);
    return fir::factory::componentToExtendedValue(
        builder, loc,
        builder.create<fir::CoordinateOp>(loc, ty, fir::getBase(obj),
                                          coorArgs));
  }

  ExtValue gen(const Fortran::evaluate::Component &cmpt) {
    // Components may be pointer or allocatable. In the gen() path, the mutable
    // aspect is lost to simplify handling on the client side. To retain the
    // mutable aspect, genMutableBoxValue should be used.
    return genComponent(cmpt).match(
        [&](const fir::MutableBoxValue &mutableBox) {
          return fir::factory::genMutableBoxRead(builder, getLoc(), mutableBox);
        },
        [](auto &box) -> ExtValue { return box; });
  }

  ExtValue genval(const Fortran::evaluate::Component &cmpt) {
    return genLoad(gen(cmpt));
  }

  // Determine the result type after removing `dims` dimensions from the array
  // type `arrTy`
  mlir::Type genSubType(mlir::Type arrTy, unsigned dims) {
    auto unwrapTy = fir::dyn_cast_ptrOrBoxEleTy(arrTy);
    assert(unwrapTy && "must be a pointer or box type");
    auto seqTy = unwrapTy.cast<fir::SequenceType>();
    auto shape = seqTy.getShape();
    assert(shape.size() > 0 && "removing columns for sequence sans shape");
    assert(dims <= shape.size() && "removing more columns than exist");
    fir::SequenceType::Shape newBnds;
    // follow Fortran semantics and remove columns (from right)
    auto e = shape.size() - dims;
    for (decltype(e) i{0}; i < e; ++i)
      newBnds.push_back(shape[i]);
    if (!newBnds.empty())
      return fir::SequenceType::get(newBnds, seqTy.getEleTy());
    return seqTy.getEleTy();
  }

  // Generate the code for a Bound value.
  ExtValue genval(const Fortran::semantics::Bound &bound) {
    if (bound.isExplicit()) {
      auto sub = bound.GetExplicit();
      if (sub.has_value())
        return genval(*sub);
      return genIntegerConstant<8>(builder.getContext(), 1);
    }
    TODO(getLoc(), "non explicit semantics::Bound lowering");
  }

  static bool isSlice(const Fortran::evaluate::ArrayRef &aref) {
    for (auto &sub : aref.subscript())
      if (std::holds_alternative<Fortran::evaluate::Triplet>(sub.u))
        return true;
    return false;
  }

  /// Lower an ArrayRef to a fir.coordinate_of given its lowered base.
  ExtValue genCoordinateOp(const ExtValue &array,
                           const Fortran::evaluate::ArrayRef &aref) {
    auto loc = getLoc();
    // References to array of rank > 1 with non constant shape that are not
    // fir.box must be collapsed into an offset computation in lowering already.
    // The same is needed with dynamic length character arrays of all ranks.
    auto baseType = fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(array).getType());
    if ((array.rank() > 1 && fir::hasDynamicSize(baseType)) ||
        fir::characterWithDynamicLen(fir::unwrapSequenceType(baseType)))
      if (!array.getBoxOf<fir::BoxValue>())
        return genOffsetAndCoordinateOp(array, aref);
    // Generate a fir.coordinate_of with zero based array indexes.
    llvm::SmallVector<mlir::Value> args;
    for (auto &subsc : llvm::enumerate(aref.subscript())) {
      auto subVal = genSubscript(subsc.value());
      assert(fir::isUnboxedValue(subVal) && "subscript must be simple scalar");
      auto val = fir::getBase(subVal);
      auto ty = val.getType();
      auto lb = getLBound(array, subsc.index(), ty);
      args.push_back(builder.create<mlir::arith::SubIOp>(loc, ty, val, lb));
    }

    auto base = fir::getBase(array);
    auto seqTy =
        fir::dyn_cast_ptrOrBoxEleTy(base.getType()).cast<fir::SequenceType>();
    assert(args.size() == seqTy.getDimension());
    auto ty = builder.getRefType(seqTy.getEleTy());
    auto addr = builder.create<fir::CoordinateOp>(loc, ty, base, args);
    return fir::factory::arrayElementToExtendedValue(builder, loc, array, addr);
  }

  /// Lower an ArrayRef to a fir.coordinate_of using an element offset instead
  /// of array indexes.
  /// This generates offset computation from the indexes and length parameters,
  /// and use the offset to access the element with a fir.coordinate_of. This
  /// must only be used if it is not possible to generate a normal
  /// fir.coordinate_of using array indexes (i.e. when the shape information is
  /// unavailable in the IR).
  ExtValue genOffsetAndCoordinateOp(const ExtValue &array,
                                    const Fortran::evaluate::ArrayRef &aref) {
    auto loc = getLoc();
    auto addr = fir::getBase(array);
    auto arrTy = fir::dyn_cast_ptrEleTy(addr.getType());
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    auto seqTy = builder.getRefType(builder.getVarLenSeqTy(eleTy));
    auto refTy = builder.getRefType(eleTy);
    auto base = builder.createConvert(loc, seqTy, addr);
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    auto zero = builder.createIntegerConstant(loc, idxTy, 0);
    auto getLB = [&](const auto &arr, unsigned dim) -> mlir::Value {
      return arr.getLBounds().empty() ? one : arr.getLBounds()[dim];
    };
    auto genFullDim = [&](const auto &arr, mlir::Value delta) -> mlir::Value {
      mlir::Value total = zero;
      assert(arr.getExtents().size() == aref.subscript().size());
      delta = builder.createConvert(loc, idxTy, delta);
      unsigned dim = 0;
      for (auto [ext, sub] : llvm::zip(arr.getExtents(), aref.subscript())) {
        auto subVal = genSubscript(sub);
        assert(fir::isUnboxedValue(subVal));
        auto val = builder.createConvert(loc, idxTy, fir::getBase(subVal));
        auto lb = builder.createConvert(loc, idxTy, getLB(arr, dim));
        auto diff = builder.create<mlir::arith::SubIOp>(loc, val, lb);
        auto prod = builder.create<mlir::arith::MulIOp>(loc, delta, diff);
        total = builder.create<mlir::arith::AddIOp>(loc, prod, total);
        if (ext)
          delta = builder.create<mlir::arith::MulIOp>(loc, delta, ext);
        ++dim;
      }
      auto origRefTy = refTy;
      if (fir::factory::CharacterExprHelper::isCharacterScalar(refTy)) {
        auto chTy = fir::factory::CharacterExprHelper::getCharacterType(refTy);
        if (fir::characterWithDynamicLen(chTy)) {
          auto ctx = builder.getContext();
          auto kind = fir::factory::CharacterExprHelper::getCharacterKind(chTy);
          auto singleTy = fir::CharacterType::getSingleton(ctx, kind);
          refTy = builder.getRefType(singleTy);
          auto seqRefTy = builder.getRefType(builder.getVarLenSeqTy(singleTy));
          base = builder.createConvert(loc, seqRefTy, base);
        }
      }
      auto coor = builder.create<fir::CoordinateOp>(
          loc, refTy, base, llvm::ArrayRef<mlir::Value>{total});
      // Convert to expected, original type after address arithmetic.
      return builder.createConvert(loc, origRefTy, coor);
    };
    return array.match(
        [&](const fir::ArrayBoxValue &arr) -> ExtValue {
          // FIXME: this check can be removed when slicing is implemented
          if (isSlice(aref))
            fir::emitFatalError(
                getLoc(),
                "slice should be handled in array expression context");
          return genFullDim(arr, one);
        },
        [&](const fir::CharArrayBoxValue &arr) -> ExtValue {
          auto delta = arr.getLen();
          // If the length is known in the type, fir.coordinate_of will
          // already take the length into account.
          if (fir::factory::CharacterExprHelper::hasConstantLengthInType(arr))
            delta = one;
          return fir::CharBoxValue(genFullDim(arr, delta), arr.getLen());
        },
        [&](const fir::BoxValue &arr) -> ExtValue {
          // CoordinateOp for BoxValue is not generated here. The dimensions
          // must be kept in the fir.coordinate_op so that potential fir.box
          // strides can be applied by codegen.
          fir::emitFatalError(
              loc, "internal: BoxValue in dim-collapsed fir.coordinate_of");
        },
        [&](const auto &) -> ExtValue {
          fir::emitFatalError(loc, "internal: array lowering failed");
        });
  }

  /// Lower an ArrayRef to a fir.array_coor.
  ExtValue genArrayCoorOp(const ExtValue &exv,
                          const Fortran::evaluate::ArrayRef &aref) {
    auto loc = getLoc();
    auto addr = fir::getBase(exv);
    auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(addr.getType());
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    auto refTy = builder.getRefType(eleTy);
    auto idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value> arrayCoorArgs;
    // The ArrayRef is expected to be scalar here, arrays are handled in array
    // expression lowering. So no vector subscript or triplet is expected here.
    for (const auto &sub : aref.subscript()) {
      auto subVal = genSubscript(sub);
      assert(fir::isUnboxedValue(subVal));
      arrayCoorArgs.push_back(
          builder.createConvert(loc, idxTy, fir::getBase(subVal)));
    }
    auto shape = builder.createShape(loc, exv);
    auto elementAddr = builder.create<fir::ArrayCoorOp>(
        loc, refTy, addr, shape, /*slice=*/mlir::Value{}, arrayCoorArgs,
        fir::getTypeParams(exv));
    return fir::factory::arrayElementToExtendedValue(builder, loc, exv,
                                                     elementAddr);
  }

  /// Return the coordinate of the array reference.
  ExtValue gen(const Fortran::evaluate::ArrayRef &aref) {
    auto base = aref.base().IsSymbol() ? gen(aref.base().GetFirstSymbol())
                                       : gen(aref.base().GetComponent());
    // Check for command-line override to use array_coor op.
    if (generateArrayCoordinate)
      return genArrayCoorOp(base, aref);
    // Otherwise, use coordinate_of op.
    return genCoordinateOp(base, aref);
  }

  /// Return lower bounds of \p box in dimension \p dim. The returned value
  /// has type \ty.
  mlir::Value getLBound(const ExtValue &box, unsigned dim, mlir::Type ty) {
    assert(box.rank() > 0 && "must be an array");
    auto loc = getLoc();
    auto one = builder.createIntegerConstant(loc, ty, 1);
    auto lb = fir::factory::readLowerBound(builder, loc, box, dim, one);
    return builder.createConvert(loc, ty, lb);
  }

  ExtValue genval(const Fortran::evaluate::ArrayRef &aref) {
    return genLoad(gen(aref));
  }

  ExtValue gen(const Fortran::evaluate::CoarrayRef &coref) {
    return Fortran::lower::CoarrayExprHelper{converter, getLoc(), symMap}
        .genAddr(coref);
  }

  ExtValue genval(const Fortran::evaluate::CoarrayRef &coref) {
    return Fortran::lower::CoarrayExprHelper{converter, getLoc(), symMap}
        .genValue(coref);
  }

  template <typename A>
  ExtValue gen(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return gen(x); }, des.u);
  }
  template <typename A>
  ExtValue genval(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return genval(x); }, des.u);
  }

  mlir::Type genType(const Fortran::evaluate::DynamicType &dt) {
    if (dt.category() != Fortran::common::TypeCategory::Derived)
      return converter.genType(dt.category(), dt.kind());
    return converter.genType(dt.GetDerivedTypeSpec());
  }

  /// Apply the function `func` and return a reference to the resultant value.
  /// This is required for lowering expressions such as `f1(f2(v))`.
  template <typename A>
  ExtValue gen(const Fortran::evaluate::FunctionRef<A> &func) {
    if (!func.GetType().has_value())
      mlir::emitError(getLoc(), "internal: a function must have a type");
    auto resTy = genType(*func.GetType());
    auto retVal = genProcedureRef(func, {resTy});
    auto retValBase = fir::getBase(retVal);
    if (fir::conformsWithPassByRef(retValBase.getType()))
      return retVal;
    auto mem = builder.create<fir::AllocaOp>(getLoc(), retValBase.getType());
    builder.create<fir::StoreOp>(getLoc(), retValBase, mem);
    return fir::substBase(retVal, mem.getResult());
  }

  /// Helper to lower intrinsic arguments for inquiry intrinsic.
  ExtValue
  lowerIntrinsicArgumentAsInquired(const Fortran::lower::SomeExpr &expr) {
    if (isAllocatableOrPointer(expr, converter))
      return genMutableBoxValue(expr);
    return gen(expr);
  }

  /// Helper to lower intrinsic arguments to a fir::BoxValue.
  /// It preserves all the non default lower bounds/non deferred length
  /// parameter information.
  ExtValue lowerIntrinsicArgumentAsBox(const Fortran::lower::SomeExpr &expr) {
    mlir::Location loc = getLoc();
    ExtValue exv = genBoxArg(expr);
    mlir::Value box = builder.createBox(loc, exv);
    return fir::BoxValue(
        box, fir::factory::getNonDefaultLowerBounds(builder, loc, exv),
        fir::factory::getNonDeferredLengthParams(exv));
  }

  /// Generate a call to an intrinsic function.
  ExtValue
  genIntrinsicRef(const Fortran::evaluate::ProcedureRef &procRef,
                  const Fortran::evaluate::SpecificIntrinsic &intrinsic,
                  llvm::Optional<mlir::Type> resultType) {
    llvm::SmallVector<ExtValue> operands;

    llvm::StringRef name = intrinsic.name;
    const auto *argLowering =
        Fortran::lower::getIntrinsicArgumentLowering(name);
    for (const auto &[arg, dummy] :
         llvm::zip(procRef.arguments(),
                   intrinsic.characteristics.value().dummyArguments)) {
      auto *expr = Fortran::evaluate::UnwrapExpr<
          Fortran::evaluate::Expr<Fortran::evaluate::SomeType>>(arg);
      if (!expr) {
        // Absent optional.
        operands.emplace_back(Fortran::lower::getAbsentIntrinsicArgument());
        continue;
      }
      if (!argLowering) {
        // No argument lowering instruction, lower by value.
        operands.emplace_back(genval(*expr));
        continue;
      }
      // Ad-hoc argument lowering handling.
      auto lowerAs = Fortran::lower::lowerIntrinsicArgumentAs(
          getLoc(), *argLowering, dummy.name);
      switch (lowerAs) {
      case Fortran::lower::LowerIntrinsicArgAs::Value:
        operands.emplace_back(genval(*expr));
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Addr:
        operands.emplace_back(gen(*expr));
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Box:
        operands.emplace_back(lowerIntrinsicArgumentAsBox(*expr));
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Inquired:
        operands.emplace_back(lowerIntrinsicArgumentAsInquired(*expr));
        continue;
      }
      llvm_unreachable("bad switch");
    }
    // Let the intrinsic library lower the intrinsic procedure call
    return Fortran::lower::genIntrinsicCall(builder, getLoc(), name, resultType,
                                            operands, stmtCtx);
  }

  template <typename A>
  bool isCharacterType(const A &exp) {
    if (auto type = exp.GetType())
      return type->category() == Fortran::common::TypeCategory::Character;
    return false;
  }

  /// helper to detect statement functions
  static bool
  isStatementFunctionCall(const Fortran::evaluate::ProcedureRef &procRef) {
    if (const auto *symbol = procRef.proc().GetSymbol())
      if (const auto *details =
              symbol->detailsIf<Fortran::semantics::SubprogramDetails>())
        return details->stmtFunction().has_value();
    return false;
  }
  /// Generate Statement function calls
  ExtValue genStmtFunctionRef(const Fortran::evaluate::ProcedureRef &procRef) {
    const auto *symbol = procRef.proc().GetSymbol();
    assert(symbol && "expected symbol in ProcedureRef of statement functions");
    const auto &details = symbol->get<Fortran::semantics::SubprogramDetails>();

    // Statement functions have their own scope, we just need to associate
    // the dummy symbols to argument expressions. They are no
    // optional/alternate return arguments. Statement functions cannot be
    // recursive (directly or indirectly) so it is safe to add dummy symbols to
    // the local map here.
    symMap.pushScope();
    for (auto [arg, bind] :
         llvm::zip(details.dummyArgs(), procRef.arguments())) {
      assert(arg && "alternate return in statement function");
      assert(bind && "optional argument in statement function");
      const auto *expr = bind->UnwrapExpr();
      // TODO: assumed type in statement function, that surprisingly seems
      // allowed, probably because nobody thought of restricting this usage.
      // gfortran/ifort compiles this.
      assert(expr && "assumed type used as statement function argument");
      // As per Fortran 2018 C1580, statement function arguments can only be
      // scalars, so just pass the box with the address. The only care is to
      // to use the dummy character explicit length if any instead of the
      // actual argument length (that can be bigger).
      if (const Fortran::semantics::DeclTypeSpec *type = arg->GetType())
        if (type->category() == Fortran::semantics::DeclTypeSpec::Character)
          if (const Fortran::semantics::MaybeIntExpr &lenExpr =
                  type->characterTypeSpec().length().GetExplicit()) {
            mlir::Value len = fir::getBase(genval(*lenExpr));
            // F2018 7.4.4.2 point 5.
            len = Fortran::lower::genMaxWithZero(builder, getLoc(), len);
            symMap.addSymbol(*arg,
                             replaceScalarCharacterLength(gen(*expr), len));
            continue;
          }
      symMap.addSymbol(*arg, gen(*expr));
    }

    // Explicitly map statement function host associated symbols to their
    // parent scope lowered symbol box.
    for (const Fortran::semantics::SymbolRef &sym :
         Fortran::evaluate::CollectSymbols(*details.stmtFunction()))
      if (const auto *details =
              sym->detailsIf<Fortran::semantics::HostAssocDetails>())
        if (!symMap.lookupSymbol(*sym))
          symMap.addSymbol(*sym, gen(details->symbol()));

    auto result = genval(details.stmtFunction().value());
    LLVM_DEBUG(llvm::dbgs() << "stmt-function: " << result << '\n');
    symMap.popScope();
    return result;
  }

  /// Helper to package a Value and its properties into an ExtendedValue.
  static ExtValue toExtendedValue(mlir::Location loc, mlir::Value base,
                                  llvm::ArrayRef<mlir::Value> extents,
                                  llvm::ArrayRef<mlir::Value> lengths) {
    auto type = base.getType();
    if (type.isa<fir::BoxType>())
      return fir::BoxValue(base, /*lbounds=*/{}, lengths, extents);
    type = fir::unwrapRefType(type);
    if (type.isa<fir::BoxType>())
      return fir::MutableBoxValue(base, lengths, /*mutableProperties*/ {});
    if (auto seqTy = type.dyn_cast<fir::SequenceType>()) {
      if (seqTy.getDimension() != extents.size())
        fir::emitFatalError(loc, "incorrect number of extents for array");
      if (seqTy.getEleTy().isa<fir::CharacterType>()) {
        if (lengths.empty())
          fir::emitFatalError(loc, "missing length for character");
        assert(lengths.size() == 1);
        return fir::CharArrayBoxValue(base, lengths[0], extents);
      }
      return fir::ArrayBoxValue(base, extents);
    }
    if (type.isa<fir::CharacterType>()) {
      if (lengths.empty())
        fir::emitFatalError(loc, "missing length for character");
      assert(lengths.size() == 1);
      return fir::CharBoxValue(base, lengths[0]);
    }
    return base;
  }

  // Find the argument that corresponds to the host associations.
  // Verify some assumptions about how the signature was built here.
  [[maybe_unused]] static unsigned findHostAssocTuplePos(mlir::FuncOp fn) {
    // Scan the argument list from last to first as the host associations are
    // appended for now.
    for (unsigned i = fn.getNumArguments(); i > 0; --i)
      if (fn.getArgAttr(i - 1, fir::getHostAssocAttrName())) {
        // Host assoc tuple must be last argument (for now).
        assert(i == fn.getNumArguments() && "tuple must be last");
        return i - 1;
      }
    llvm_unreachable("anyFuncArgsHaveAttr failed");
  }

  /// Create a contiguous temporary array with the same shape,
  /// length parameters and type as mold. It is up to the caller to deallocate
  /// the temporary.
  ExtValue genTempFromMold(const ExtValue &mold, llvm::StringRef tempName) {
    mlir::Type type = fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(mold).getType());
    assert(type && "expected descriptor or memory type");
    auto loc = getLoc();
    auto extents = fir::factory::getExtents(builder, loc, mold);
    auto allocMemTypeParams = fir::getTypeParams(mold);
    mlir::Value charLen;
    mlir::Type elementType = fir::unwrapSequenceType(type);
    if (auto charType = elementType.dyn_cast<fir::CharacterType>()) {
      charLen = allocMemTypeParams.empty()
                    ? fir::factory::readCharLen(builder, loc, mold)
                    : allocMemTypeParams[0];
      if (charType.hasDynamicLen() && allocMemTypeParams.empty())
        allocMemTypeParams.push_back(charLen);
    } else if (fir::hasDynamicSize(elementType)) {
      TODO(loc, "Creating temporary for derived type with length parameters");
    }

    mlir::Value temp = builder.create<fir::AllocMemOp>(
        loc, type, tempName, allocMemTypeParams, extents);
    if (fir::unwrapSequenceType(type).isa<fir::CharacterType>())
      return fir::CharArrayBoxValue{temp, charLen, extents};
    return fir::ArrayBoxValue{temp, extents};
  }

  /// Copy \p source array into \p dest array. Both arrays must be
  /// conforming, but neither array must be contiguous.
  void genArrayCopy(ExtValue dest, ExtValue source) {
    return createSomeArrayAssignment(converter, dest, source, symMap, stmtCtx);
  }

  /// Lower a non-elemental procedure reference and read allocatable and pointer
  /// results into normal values.
  ExtValue genProcedureRef(const Fortran::evaluate::ProcedureRef &procRef,
                           llvm::Optional<mlir::Type> resultType) {
    auto res = genRawProcedureRef(procRef, resultType);
    // In most contexts, pointers and allocatable do not appear as allocatable
    // or pointer variable on the caller side (see 8.5.3 note 1 for
    // allocatables). The few context where this can happen must call
    // genRawProcedureRef directly.
    if (const auto *box = res.getBoxOf<fir::MutableBoxValue>())
      return fir::factory::genMutableBoxRead(builder, getLoc(), *box);
    return res;
  }

  /// Given a call site for which the arguments were already lowered, generate
  /// the call and return the result. This function deals with explicit result
  /// allocation and lowering if needed. It also deals with passing the host
  /// link to internal procedures.
  ExtValue genCallOpAndResult(Fortran::lower::CallerInterface &caller,
                              mlir::FunctionType callSiteType,
                              llvm::Optional<mlir::Type> resultType) {
    auto loc = getLoc();
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;
    // Handle cases where caller must allocate the result or a fir.box for it.
    bool mustPopSymMap = false;
    if (caller.mustMapInterfaceSymbols()) {
      symMap.pushScope();
      mustPopSymMap = true;
      Fortran::lower::mapCallInterfaceSymbols(converter, caller, symMap);
    }

    auto idxTy = builder.getIndexType();
    auto lowerSpecExpr = [&](const auto &expr) -> mlir::Value {
      return builder.createConvert(
          loc, idxTy, fir::getBase(converter.genExprValue(expr, stmtCtx)));
    };
    llvm::SmallVector<mlir::Value> resultLengths;
    auto allocatedResult = [&]() -> llvm::Optional<ExtValue> {
      llvm::SmallVector<mlir::Value> extents;
      llvm::SmallVector<mlir::Value> lengths;
      if (!caller.callerAllocateResult())
        return {};
      auto type = caller.getResultStorageType();
      if (type.isa<fir::SequenceType>())
        caller.walkResultExtents([&](const Fortran::lower::SomeExpr &e) {
          extents.emplace_back(lowerSpecExpr(e));
        });
      caller.walkResultLengths([&](const Fortran::lower::SomeExpr &e) {
        lengths.emplace_back(lowerSpecExpr(e));
      });
      /// Result lengths parameters should not be provided to box storage
      /// allocation and save_results, but they are still useful information to
      /// keep in the ExtendedValue if non-deferred.
      if (!type.isa<fir::BoxType>())
        resultLengths = lengths;
      auto temp =
          builder.createTemporary(loc, type, ".result", extents, resultLengths);
      return toExtendedValue(loc, temp, extents, lengths);
    }();

    if (mustPopSymMap)
      symMap.popScope();

    // Place allocated result or prepare the fir.save_result arguments.
    mlir::Value arrayResultShape;
    if (allocatedResult) {
      if (auto resultArg = caller.getPassedResult()) {
        if (resultArg->passBy == PassBy::AddressAndLength)
          caller.placeAddressAndLengthInput(*resultArg,
                                            fir::getBase(*allocatedResult),
                                            fir::getLen(*allocatedResult));
        else if (resultArg->passBy == PassBy::BaseAddress)
          caller.placeInput(*resultArg, fir::getBase(*allocatedResult));
        else
          fir::emitFatalError(
              loc, "only expect character scalar result to be passed by ref");
      } else {
        assert(caller.mustSaveResult());
        arrayResultShape = allocatedResult->match(
            [&](const fir::CharArrayBoxValue &) {
              return builder.createShape(loc, *allocatedResult);
            },
            [&](const fir::ArrayBoxValue &) {
              return builder.createShape(loc, *allocatedResult);
            },
            [&](const auto &) { return mlir::Value{}; });
      }
    }

    // In older Fortran, procedure argument types are inferred. This may lead
    // different view of what the function signature is in different locations.
    // Casts are inserted as needed below to acomodate this.

    // The mlir::FuncOp type prevails, unless it has a different number of
    // arguments which can happen in legal program if it was passed as a dummy
    // procedure argument earlier with no further type information.
    mlir::Value funcPointer;
    mlir::SymbolRefAttr funcSymbolAttr;
    bool addHostAssociations = false;
    if (const auto *sym = caller.getIfIndirectCallSymbol()) {
      funcPointer = symMap.lookupSymbol(*sym).getAddr();
      if (!funcPointer)
        TODO(loc, "calling a dummy procedure declared in an outer scope");
    } else {
      auto funcOpType = caller.getFuncOp().getType();
      auto symbolAttr = builder.getSymbolRefAttr(caller.getMangledName());
      if (callSiteType.getNumResults() == funcOpType.getNumResults() &&
          callSiteType.getNumInputs() + 1 == funcOpType.getNumInputs() &&
          fir::anyFuncArgsHaveAttr(caller.getFuncOp(),
                                   fir::getHostAssocAttrName())) {
        // The number of arguments is off by one, and we're lowering a function
        // with host associations. Modify call to include host associations
        // argument by appending the value at the end of the operands.
        assert(funcOpType.getInput(findHostAssocTuplePos(caller.getFuncOp())) ==
               converter.hostAssocTupleValue().getType());
        addHostAssociations = true;
      }
      if (!addHostAssociations &&
          (callSiteType.getNumResults() != funcOpType.getNumResults() ||
           callSiteType.getNumInputs() != funcOpType.getNumInputs())) {
        // Deal with argument number mismatch by making a function pointer so
        // that function type cast can be inserted. Do not emit a warning here
        // because this can happen in legal program if the function is not
        // defined here and it was first passed as an argument without any more
        // information.
        funcPointer =
            builder.create<fir::AddrOfOp>(loc, funcOpType, symbolAttr);
      } else if (callSiteType.getResults() != funcOpType.getResults()) {
        // Implicit interface result type mismatch are not standard Fortran, but
        // some compilers are not complaining about it.  The front end is not
        // protecting lowering from this currently. Support this with a
        // discouraging warning.
        LLVM_DEBUG(mlir::emitWarning(
            loc, "a return type mismatch is not standard compliant and may "
                 "lead to undefined behavior."));
        // Cast the actual function to the current caller implicit type because
        // that is the behavior we would get if we could not see the definition.
        funcPointer =
            builder.create<fir::AddrOfOp>(loc, funcOpType, symbolAttr);
      } else {
        funcSymbolAttr = symbolAttr;
      }
    }

    auto funcType = funcPointer ? callSiteType : caller.getFuncOp().getType();
    llvm::SmallVector<mlir::Value> operands;
    // First operand of indirect call is the function pointer. Cast it to
    // required function type for the call to handle procedures that have a
    // compatible interface in Fortran, but that have different signatures in
    // FIR.
    if (funcPointer)
      operands.push_back(builder.createConvert(loc, funcType, funcPointer));

    // Deal with potential mismatches in arguments types. Passing an array to a
    // scalar argument should for instance be tolerated here.
    for (auto [fst, snd] :
         llvm::zip(caller.getInputs(), funcType.getInputs())) {
      auto cast = builder.convertWithSemantics(getLoc(), snd, fst);
      operands.push_back(cast);
    }

    // Add host associations as necessary.
    if (addHostAssociations)
      operands.push_back(converter.hostAssocTupleValue());

    auto call = builder.create<fir::CallOp>(loc, funcType.getResults(),
                                            funcSymbolAttr, operands);

    if (caller.mustSaveResult())
      builder.create<fir::SaveResultOp>(
          loc, call.getResult(0), fir::getBase(allocatedResult.getValue()),
          arrayResultShape, resultLengths);

    if (allocatedResult) {
      allocatedResult->match(
          [&](const fir::MutableBoxValue &box) {
            if (box.isAllocatable()) {
              // 9.7.3.2 point 4. Finalize allocatables.
              auto *bldr = &converter.getFirOpBuilder();
              stmtCtx.attachCleanup([bldr, loc, box]() {
                fir::factory::genFinalization(*bldr, loc, box);
              });
            }
          },
          [](const auto &) {});
      return *allocatedResult;
    }

    if (!resultType.hasValue())
      return mlir::Value{}; // subroutine call
    // For now, Fortran return values are implemented with a single MLIR
    // function return value.
    assert(call.getNumResults() == 1 &&
           "Expected exactly one result in FUNCTION call");
    return call.getResult(0);
  }

  /// Is this a variable wrapped in parentheses ?
  template <typename A>
  bool isParenthesizedVariable(const A &) {
    return false;
  }
  template <typename T>
  bool isParenthesizedVariable(const Fortran::evaluate::Expr<T> &expr) {
    using ExprVariant = decltype(Fortran::evaluate::Expr<T>::u);
    using Parentheses = Fortran::evaluate::Parentheses<T>;
    if constexpr (Fortran::common::HasMember<Parentheses, ExprVariant>) {
      if (const auto *parentheses = std::get_if<Parentheses>(&expr.u))
        return Fortran::evaluate::IsVariable(parentheses->left());
      return false;
    } else {
      return std::visit(
          [&](const auto &x) { return isParenthesizedVariable(x); }, expr.u);
    }
  }

  /// Like genExtAddr, but ensure the address returned is a temporary even if \p
  /// expr is variable inside parentheses.
  ExtValue genTempExtAddr(const Fortran::lower::SomeExpr &expr) {
    // In general, genExtAddr might not create a temp for variable inside
    // parentheses to avoid creating array temporary in sub-expressions. It only
    // ensures the sub-expression is not re-associated with other parts of the
    // expression. In the call semantics, there is a difference between expr and
    // variable (see R1524). For expressions, a variable storage must not be
    // argument associated since it could be modified inside the call, or the
    // variable could also be modified by other means during the call.
    if (!isParenthesizedVariable(expr))
      return genExtAddr(expr);
    if (expr.Rank() > 0)
      return asArray(expr);
    auto loc = getLoc();
    return genExtValue(expr).match(
        [&](const fir::CharBoxValue &boxChar) -> ExtValue {
          return fir::factory::CharacterExprHelper{builder, loc}.createTempFrom(
              boxChar);
        },
        [&](const fir::UnboxedValue &v) -> ExtValue {
          auto type = v.getType();
          mlir::Value value = v;
          if (fir::isa_ref_type(type))
            value = builder.create<fir::LoadOp>(loc, value);
          auto temp = builder.createTemporary(loc, value.getType());
          builder.create<fir::StoreOp>(loc, value, temp);
          return temp;
        },
        [&](const fir::BoxValue &x) -> ExtValue {
          // Derived type scalar that may be polymorphic.
          assert(!x.hasRank() && x.isDerived());
          if (x.isDerivedWithLengthParameters())
            fir::emitFatalError(
                loc, "making temps for derived type with length parameters");
          // TODO: polymorphic aspects should be kept but for now the temp
          // created always has the declared type.
          auto var = fir::getBase(fir::factory::readBoxValue(builder, loc, x));
          auto value = builder.create<fir::LoadOp>(loc, var);
          auto temp = builder.createTemporary(loc, value.getType());
          builder.create<fir::StoreOp>(loc, value, temp);
          return temp;
        },
        [&](const auto &) -> ExtValue {
          fir::emitFatalError(loc, "expr is not a scalar value");
        });
  }

  /// Helper structure to track potential copy-in of non contiguous variable
  /// argument into a contiguous temp. It is used to deallocate the temp that
  /// may have been created as well as to the copy-out from the temp to the
  /// variable after the call.
  struct CopyOutPair {
    ExtValue var;
    ExtValue temp;
    // Flag to indicate if the argument may have been modified by the
    // callee, in which case it must be copied-out to the variable.
    bool argMayBeModifiedByCall;
    // Optional boolean value that, if present and false, prevents
    // the copy-out and temp deallocation.
    llvm::Optional<mlir::Value> restrictCopyAndFreeAtRuntime;
  };
  using CopyOutPairs = llvm::SmallVector<CopyOutPair, 4>;

  /// Helper to read any fir::BoxValue into other fir::ExtendedValue categories
  /// not based on fir.box.
  /// This will lose any non contiguous stride information and dynamic type and
  /// should only be called if \p exv is known to be contiguous or if its base
  /// address will be replaced by a contiguous one. If \p exv is not a
  /// fir::BoxValue, this is a no-op.
  ExtValue readIfBoxValue(const ExtValue &exv) {
    if (const auto *box = exv.getBoxOf<fir::BoxValue>())
      return fir::factory::readBoxValue(builder, getLoc(), *box);
    return exv;
  }

  /// Generate a contiguous temp to pass \p actualArg as argument \p arg. The
  /// creation of the temp and copy-in can be made conditional at runtime by
  /// providing a runtime boolean flag \p restrictCopyAtRuntime (in which case
  /// the temp and copy will only be made if the value is true at runtmime).
  ExtValue genCopyIn(const ExtValue &actualArg,
                     const Fortran::lower::CallerInterface::PassedEntity &arg,
                     CopyOutPairs &copyOutPairs,
                     llvm::Optional<mlir::Value> restrictCopyAtRuntime) {
    if (!restrictCopyAtRuntime) {
      auto temp = genTempFromMold(actualArg, ".copyinout");
      if (arg.mayBeReadByCall())
        genArrayCopy(temp, actualArg);
      copyOutPairs.emplace_back(CopyOutPair{
          actualArg, temp, arg.mayBeModifiedByCall(), restrictCopyAtRuntime});
      return temp;
    }
    // Otherwise, need to be careful to only copy-in if allowed at runtime.
    auto loc = getLoc();
    auto addrType = fir::HeapType::get(
        fir::unwrapPassByRefType(fir::getBase(actualArg).getType()));
    auto addr = builder
                    .genIfOp(loc, {addrType}, *restrictCopyAtRuntime,
                             /*withElseRegion=*/true)
                    .genThen([&]() {
                      auto temp = genTempFromMold(actualArg, ".copyinout");
                      if (arg.mayBeReadByCall())
                        genArrayCopy(temp, actualArg);
                      builder.create<fir::ResultOp>(loc, fir::getBase(temp));
                    })
                    .genElse([&]() {
                      auto nullPtr = builder.createNullConstant(loc, addrType);
                      builder.create<fir::ResultOp>(loc, nullPtr);
                    })
                    .getResults()[0];
    // Associate the temp address with actualArg lengths and extents.
    auto temp = fir::substBase(readIfBoxValue(actualArg), addr);
    copyOutPairs.emplace_back(CopyOutPair{
        actualArg, temp, arg.mayBeModifiedByCall(), restrictCopyAtRuntime});
    return temp;
  }

  /// Generate copy-out if needed and free the temporary for an argument that
  /// has been copied-in into a contiguous temp.
  void genCopyOut(const CopyOutPair &copyOutPair) {
    auto loc = getLoc();
    if (!copyOutPair.restrictCopyAndFreeAtRuntime) {
      if (copyOutPair.argMayBeModifiedByCall)
        genArrayCopy(copyOutPair.var, copyOutPair.temp);
      builder.create<fir::FreeMemOp>(loc, fir::getBase(copyOutPair.temp));
      return;
    }
    builder.genIfThen(loc, *copyOutPair.restrictCopyAndFreeAtRuntime)
        .genThen([&]() {
          if (copyOutPair.argMayBeModifiedByCall)
            genArrayCopy(copyOutPair.var, copyOutPair.temp);
          builder.create<fir::FreeMemOp>(loc, fir::getBase(copyOutPair.temp));
        })
        .end();
  }

  /// Lower a non-elemental procedure reference.
  ExtValue genRawProcedureRef(const Fortran::evaluate::ProcedureRef &procRef,
                              llvm::Optional<mlir::Type> resultType) {
    auto loc = getLoc();
    if (isElementalProcWithArrayArgs(procRef))
      fir::emitFatalError(loc, "trying to lower elemental procedure with array "
                               "arguments as normal procedure");
    if (const auto *intrinsic = procRef.proc().GetSpecificIntrinsic())
      return genIntrinsicRef(procRef, *intrinsic, resultType);

    if (isStatementFunctionCall(procRef))
      return genStmtFunctionRef(procRef);

    Fortran::lower::CallerInterface caller(procRef, converter);
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;

    llvm::SmallVector<fir::MutableBoxValue> mutableModifiedByCall;
    // List of <var, temp> where temp must be copied into var after the call.
    CopyOutPairs copyOutPairs;

    auto callSiteType = caller.genFunctionType();

    // Lower the actual arguments and map the lowered values to the dummy
    // arguments.
    for (const auto &arg : caller.getPassedArguments()) {
      const auto *actual = arg.entity;
      auto argTy = callSiteType.getInput(arg.firArgument);
      if (!actual) {
        // Optional dummy argument for which there is no actual argument.
        caller.placeInput(arg, builder.create<fir::AbsentOp>(loc, argTy));
        continue;
      }
      const auto *expr = actual->UnwrapExpr();
      if (!expr)
        TODO(loc, "assumed type actual argument lowering");

      if (arg.passBy == PassBy::Value) {
        auto argVal = genval(*expr);
        if (!fir::isUnboxedValue(argVal))
          fir::emitFatalError(
              loc, "internal error: passing non trivial value by value");
        caller.placeInput(arg, fir::getBase(argVal));
        continue;
      }

      if (arg.passBy == PassBy::MutableBox) {
        if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
                *expr)) {
          // If expr is NULL(), the mutableBox created must be a deallocated
          // pointer with the dummy argument characteristics (see table 16.5
          // in Fortran 2018 standard).
          // No length parameters are set for the created box because any non
          // deferred type parameters of the dummy will be evaluated on the
          // callee side, and it is illegal to use NULL without a MOLD if any
          // dummy length parameters are assumed.
          auto boxTy = fir::dyn_cast_ptrEleTy(argTy);
          assert(boxTy && boxTy.isa<fir::BoxType>() &&
                 "must be a fir.box type");
          auto boxStorage = builder.createTemporary(loc, boxTy);
          auto nullBox = fir::factory::createUnallocatedBox(
              builder, loc, boxTy, /*nonDeferredParams=*/{});
          builder.create<fir::StoreOp>(loc, nullBox, boxStorage);
          caller.placeInput(arg, boxStorage);
          continue;
        }
        auto mutableBox = genMutableBoxValue(*expr);
        auto irBox = fir::factory::getMutableIRBox(builder, loc, mutableBox);
        caller.placeInput(arg, irBox);
        if (arg.mayBeModifiedByCall())
          mutableModifiedByCall.emplace_back(std::move(mutableBox));
        continue;
      }
      const bool actualArgIsVariable = Fortran::evaluate::IsVariable(*expr);
      if (arg.passBy == PassBy::BaseAddress || arg.passBy == PassBy::BoxChar) {
        const bool actualIsSimplyContiguous =
            !actualArgIsVariable || Fortran::evaluate::IsSimplyContiguous(
                                        *expr, converter.getFoldingContext());
        auto argAddr = [&]() -> ExtValue {
          ExtValue baseAddr;
          if (actualArgIsVariable && arg.isOptional()) {
            if (isAllocatableOrPointer(*expr, converter)) {
              // Fortran 2018 15.5.2.12 point 1: If unallocated/disassociated,
              // it is as if the argument was absent. The main care here is to
              // not do a copy-in/copy-out because the temp address, even though
              // pointing to a null size storage, would not be a nullptr and
              // therefore the argument would not be considered absent on the
              // callee side. Note: if wholeSymbol is optional, it cannot be
              // absent as per 15.5.2.12 point 7. and 8. We rely on this to
              // un-conditionally read the allocatable/pointer descriptor here.
              if (actualIsSimplyContiguous)
                return genBoxArg(*expr);
              fir::MutableBoxValue mutableBox = genMutableBoxValue(*expr);
              auto isAssociated = fir::factory::genIsAllocatedOrAssociatedTest(
                  builder, loc, mutableBox);
              auto actualExv =
                  fir::factory::genMutableBoxRead(builder, loc, mutableBox);
              return genCopyIn(actualExv, arg, copyOutPairs, isAssociated);
            }
            if (const Fortran::semantics::Symbol *wholeSymbol =
                    Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(
                        *expr))
              if (Fortran::semantics::IsOptional(*wholeSymbol)) {
                auto actualArg = gen(*expr);
                auto actualArgBase = fir::getBase(actualArg);
                if (!actualArgBase.getType().isa<fir::BoxType>())
                  return actualArg;
                // Do not read wholeSymbol descriptor that may be a nullptr in
                // case wholeSymbol is absent.
                // Absent descriptor cannot be read. To avoid any issue in
                // copy-in/copy-out, and when retrieving the address/length
                // create an descriptor pointing to a null address here if the
                // fir.box is absent.
                mlir::Value isPresent = builder.create<fir::IsPresentOp>(
                    loc, builder.getI1Type(), actualArgBase);
                auto boxType = actualArgBase.getType();
                auto emptyBox = fir::factory::createUnallocatedBox(
                    builder, loc, boxType, llvm::None);
                auto safeToReadBox = builder.create<mlir::SelectOp>(
                    loc, isPresent, actualArgBase, emptyBox);
                auto safeToReadExv = fir::substBase(actualArg, safeToReadBox);
                if (actualIsSimplyContiguous)
                  return safeToReadExv;
                return genCopyIn(safeToReadExv, arg, copyOutPairs, isPresent);
              }
            // Fall through: The actual argument can safely be
            // copied-in/copied-out without any care if needed.
          }
          if (actualArgIsVariable && expr->Rank() > 0) {
            auto box = genBoxArg(*expr);
            if (!actualIsSimplyContiguous)
              return genCopyIn(box, arg, copyOutPairs,
                               /*restrictCopyAtRuntime=*/llvm::None);
            // Contiguous: just use the box we created above!
            // This gets "unboxed" below, if needed.
            return box;
          }
          // Actual argument is a non optional/non pointer/non allocatable
          // scalar.
          if (actualArgIsVariable)
            return genExtAddr(*expr);
          // Actual argument is not a variable. Make sure a variable address is
          // not passed.
          return genTempExtAddr(*expr);
        }();
        // Scalar and contiguous expressions may be lowered to a fir.box,
        // either to account for potential polymorphism, or because lowering
        // did not account for some contiguity hints.
        // Here, polymorphism does not matter (an entity of the declared type
        // is passed, not one of the dynamic type), and the expr is known to
        // be simply contiguous, so it is safe to unbox it and pass the
        // address without making a copy.
        argAddr = readIfBoxValue(argAddr);

        if (arg.passBy == PassBy::BaseAddress) {
          caller.placeInput(arg, fir::getBase(argAddr));
        } else {
          assert(arg.passBy == PassBy::BoxChar);
          auto helper = fir::factory::CharacterExprHelper{builder, loc};
          auto boxChar = argAddr.match(
              [&](const fir::CharBoxValue &x) { return helper.createEmbox(x); },
              [&](const fir::CharArrayBoxValue &x) {
                return helper.createEmbox(x);
              },
              [&](const auto &) -> mlir::Value {
                fir::emitFatalError(
                    loc, "internal error: actual argument is not a character");
              });
          caller.placeInput(arg, boxChar);
        }
      } else if (arg.passBy == PassBy::Box) {
        // Before lowering to an address, handle the allocatable/pointer actual
        // argument to optional fir.box dummy. It is legal to pass
        // unallocated/disassociated entity to an optional. In this case, an
        // absent fir.box must be created instead of a fir.box with a null value
        // (Fortran 2018 15.5.2.12 point 1).
        if (arg.isOptional() && isAllocatableOrPointer(*expr, converter)) {
          // Note that passing an absent allocatable to a non-allocatable
          // optional dummy argument is illegal (15.5.2.12 point 3 (8)). So
          // nothing has to be done to generate an absent argument in this case,
          // and it is OK to unconditionally read the mutable box here.
          auto mutableBox = genMutableBoxValue(*expr);
          auto isAllocated = fir::factory::genIsAllocatedOrAssociatedTest(
              builder, loc, mutableBox);
          auto absent = builder.create<fir::AbsentOp>(loc, argTy);
          /// For now, assume it is not OK to pass the allocatable/pointer
          /// descriptor to a non pointer/allocatable dummy. That is a strict
          /// interpretation of 18.3.6 point 4 that stipulates the descriptor
          /// has the dummy attributes in BIND(C) contexts.
          auto box = builder.createBox(
              loc, fir::factory::genMutableBoxRead(builder, loc, mutableBox));
          // Need the box types to be exactly similar for the selectOp.
          auto convertedBox = builder.createConvert(loc, argTy, box);
          caller.placeInput(arg, builder.create<mlir::SelectOp>(
                                     loc, isAllocated, convertedBox, absent));
        } else {
          // Make sure a variable address is only passed if the expression is
          // actually a variable.
          auto box = actualArgIsVariable
                         ? builder.createBox(loc, genBoxArg(*expr))
                         : builder.createBox(getLoc(), genTempExtAddr(*expr));
          caller.placeInput(arg, box);
        }
      } else if (arg.passBy == PassBy::AddressAndLength) {
        auto argRef = genExtAddr(*expr);
        caller.placeAddressAndLengthInput(arg, fir::getBase(argRef),
                                          fir::getLen(argRef));
      } else {
        TODO(loc, "pass by value in non elemental function call");
      }
    }

    auto result = genCallOpAndResult(caller, callSiteType, resultType);

    // Sync pointers and allocatables that may have been modified during the
    // call.
    for (const auto &mutableBox : mutableModifiedByCall)
      fir::factory::syncMutableBoxFromIRBox(builder, loc, mutableBox);
    // Handle case where result was passed as argument

    // Copy-out temps that were created for non contiguous variable arguments if
    // needed.
    for (const auto &copyOutPair : copyOutPairs)
      genCopyOut(copyOutPair);

    return result;
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::FunctionRef<Fortran::evaluate::Type<TC, KIND>>
             &funRef) {
    auto retTy = converter.genType(TC, KIND);
    ExtValue result = genProcedureRef(funRef, {retTy});
    if (result.rank() == 0 && fir::isa_ref_type(fir::getBase(result).getType()))
      return genLoad(result);
    return result;
  }

  ExtValue genval(const Fortran::evaluate::ProcedureRef &procRef) {
    llvm::Optional<mlir::Type> resTy;
    if (procRef.hasAlternateReturns())
      resTy = builder.getIndexType();
    return genProcedureRef(procRef, resTy);
  }

  template <typename A>
  bool isScalar(const A &x) {
    return x.Rank() == 0;
  }

  /// Helper to detect Transformational function reference.
  template <typename T>
  bool isTransformationalRef(const T &) {
    return false;
  }
  template <typename T>
  bool isTransformationalRef(const Fortran::evaluate::FunctionRef<T> &funcRef) {
    return !funcRef.IsElemental() && funcRef.Rank();
  }
  template <typename T>
  bool isTransformationalRef(Fortran::evaluate::Expr<T> expr) {
    return std::visit([&](const auto &e) { return isTransformationalRef(e); },
                      expr.u);
  }

  template <typename A>
  ExtValue asArray(const A &x) {
    return Fortran::lower::createSomeArrayTempValue(converter, toEvExpr(x),
                                                    symMap, stmtCtx);
  }

  /// Lower an array value as an argument. This argument can be passed as a box
  /// value, so it may be possible to avoid making a temporary.
  template <typename A>
  ExtValue asArrayArg(const Fortran::evaluate::Expr<A> &x) {
    return std::visit([&](const auto &e) { return asArrayArg(e, x); }, x.u);
  }
  template <typename A, typename B>
  ExtValue asArrayArg(const Fortran::evaluate::Expr<A> &x, const B &y) {
    return std::visit([&](const auto &e) { return asArrayArg(e, y); }, x.u);
  }
  template <typename A, typename B>
  ExtValue asArrayArg(const Fortran::evaluate::Designator<A> &, const B &x) {
    // Designator is being passed as an argument to a procedure. Lower the
    // expression to a boxed value.
    return Fortran::lower::createSomeArrayBox(converter, toEvExpr(x), symMap,
                                              stmtCtx);
  }
  template <typename A, typename B>
  ExtValue asArrayArg(const A &, const B &x) {
    // If the expression to pass as an argument is not a designator, then create
    // an array temp.
    return asArray(x);
  }

  template <typename A>
  ExtValue gen(const Fortran::evaluate::Expr<A> &x) {
    // Whole array symbols or components, and results of transformational
    // functions already have a storage and the scalar expression lowering path
    // is used to not create a new temporary storage.
    if (isScalar(x) ||
        Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(x) ||
        isTransformationalRef(x))
      return std::visit([&](const auto &e) { return genref(e); }, x.u);
    if (useBoxArg)
      return asArrayArg(x);
    return asArray(x);
  }
  template <typename A>
  ExtValue genval(const Fortran::evaluate::Expr<A> &x) {
    if (isScalar(x) || Fortran::evaluate::UnwrapWholeSymbolDataRef(x) ||
        inInitializer)
      return std::visit([&](const auto &e) { return genval(e); }, x.u);
    return asArray(x);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Expr<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Logical, KIND>> &exp) {
    return std::visit([&](const auto &e) { return genval(e); }, exp.u);
  }

  using RefSet =
      std::tuple<Fortran::evaluate::ComplexPart, Fortran::evaluate::Substring,
                 Fortran::evaluate::DataRef, Fortran::evaluate::Component,
                 Fortran::evaluate::ArrayRef, Fortran::evaluate::CoarrayRef,
                 Fortran::semantics::SymbolRef>;
  template <typename A>
  static constexpr bool inRefSet = Fortran::common::HasMember<A, RefSet>;

  template <typename A, typename = std::enable_if_t<inRefSet<A>>>
  ExtValue genref(const A &a) {
    return gen(a);
  }
  template <typename A>
  ExtValue genref(const A &a) {
    auto exv = genval(a);
    auto valBase = fir::getBase(exv);
    // Functions are always referent.
    if (valBase.getType().template isa<mlir::FunctionType>() ||
        fir::conformsWithPassByRef(valBase.getType()))
      return exv;

    // Since `a` is not itself a valid referent, determine its value and
    // create a temporary location at the begining of the function for
    // referencing.
    auto val = valBase;
    if constexpr (!Fortran::common::HasMember<
                      A, Fortran::evaluate::TypelessExpression>) {
      if constexpr (A::Result::category ==
                    Fortran::common::TypeCategory::Logical) {
        // Ensure logicals that may have been lowered to `i1` are cast to the
        // Fortran logical type before being placed in memory.
        auto type = converter.genType(A::Result::category, A::Result::kind);
        val = builder.createConvert(getLoc(), type, valBase);
      }
    }
    auto func = builder.getFunction();
    auto initPos = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&func.front());
    auto mem = builder.create<fir::AllocaOp>(getLoc(), val.getType());
    builder.restoreInsertionPoint(initPos);
    builder.create<fir::StoreOp>(getLoc(), val, mem);
    return fir::substBase(exv, mem.getResult());
  }

  template <typename A, template <typename> typename T,
            typename B = std::decay_t<T<A>>,
            std::enable_if_t<
                std::is_same_v<B, Fortran::evaluate::Expr<A>> ||
                    std::is_same_v<B, Fortran::evaluate::Designator<A>> ||
                    std::is_same_v<B, Fortran::evaluate::FunctionRef<A>>,
                bool> = true>
  ExtValue genref(const T<A> &x) {
    return gen(x);
  }

private:
  mlir::Location location;
  Fortran::lower::AbstractConverter &converter;
  fir::FirOpBuilder &builder;
  Fortran::lower::StatementContext &stmtCtx;
  Fortran::lower::SymMap &symMap;
  bool inInitializer;
  bool useBoxArg = false; // expression lowered as argument
};
} // namespace

// Helper for changing the semantics in a given context. Preserves the current
// semantics which is resumed when the "push" goes out of scope.
#define PushSemantics(PushVal)                                                 \
  [[maybe_unused]] auto pushSemanticsLocalVariable##__LINE__ =                 \
      Fortran::common::ScopedSet(semant, PushVal);

static bool isAdjustedArrayElementType(mlir::Type t) {
  return fir::isa_char(t) || fir::isa_derived(t) || t.isa<fir::SequenceType>();
}
static bool elementTypeWasAdjusted(mlir::Type t) {
  if (auto ty = t.dyn_cast<fir::ReferenceType>())
    return isAdjustedArrayElementType(ty.getEleTy());
  return false;
}
static mlir::Type adjustedArrayElementType(mlir::Type t) {
  return isAdjustedArrayElementType(t) ? fir::ReferenceType::get(t) : t;
}

/// Helper to generate calls to scalar user defined assignment procedures.
static void genScalarUserDefinedAssignmentCall(fir::FirOpBuilder &builder,
                                               mlir::Location loc,
                                               mlir::FuncOp func,
                                               const fir::ExtendedValue &lhs,
                                               const fir::ExtendedValue &rhs) {
  auto prepareUserDefinedArg =
      [](fir::FirOpBuilder &builder, mlir::Location loc,
         const fir::ExtendedValue &value, mlir::Type argType) -> mlir::Value {
    if (argType.isa<fir::BoxCharType>()) {
      const auto *charBox = value.getCharBox();
      assert(charBox && "argument type mismatch in elemental user assignment");
      return fir::factory::CharacterExprHelper{builder, loc}.createEmbox(
          *charBox);
    }
    if (argType.isa<fir::BoxType>()) {
      auto box = builder.createBox(loc, value);
      return builder.createConvert(loc, argType, box);
    }
    // Simple pass by address.
    auto argBaseType = fir::unwrapRefType(argType);
    assert(!fir::hasDynamicSize(argBaseType));
    auto from = fir::getBase(value);
    if (argBaseType != fir::unwrapRefType(from.getType())) {
      // With logicals, it is possible that from is i1 here.
      if (fir::isa_ref_type(from.getType()))
        from = builder.create<fir::LoadOp>(loc, from);
      from = builder.createConvert(loc, argBaseType, from);
    }
    if (!fir::isa_ref_type(from.getType())) {
      auto temp = builder.createTemporary(loc, argBaseType);
      builder.create<fir::StoreOp>(loc, from, temp);
      from = temp;
    }
    return builder.createConvert(loc, argType, from);
  };
  assert(func.getNumArguments() == 2);
  auto lhsType = func.getType().getInput(0);
  auto rhsType = func.getType().getInput(1);
  auto lhsArg = prepareUserDefinedArg(builder, loc, lhs, lhsType);
  auto rhsArg = prepareUserDefinedArg(builder, loc, rhs, rhsType);
  builder.create<fir::CallOp>(loc, func, mlir::ValueRange{lhsArg, rhsArg});
}

/// Convert the result of a fir.array_modify to an ExtendedValue given the
/// related fir.array_load.
static fir::ExtendedValue arrayModifyToExv(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           fir::ArrayLoadOp load,
                                           mlir::Value elementAddr) {
  auto eleTy = fir::unwrapPassByRefType(elementAddr.getType());
  if (fir::isa_char(eleTy)) {
    auto len = fir::factory::CharacterExprHelper{builder, loc}.getLength(
        load.memref());
    if (!len) {
      assert(load.typeparams().size() == 1 && "length must be in array_load");
      len = load.typeparams()[0];
    }
    return fir::CharBoxValue{elementAddr, len};
  }
  return elementAddr;
}

//===----------------------------------------------------------------------===//
//
// Lowering of scalar expressions in an explicit iteration space context.
//
//===----------------------------------------------------------------------===//

// Shared code for creating a copy of a derived type element. This function is
// called from a continuation.
inline static fir::ArrayAmendOp
createDerivedArrayAmend(mlir::Location loc, fir::ArrayLoadOp destLoad,
                        fir::FirOpBuilder &builder, fir::ArrayAccessOp destAcc,
                        const fir::ExtendedValue &elementExv, mlir::Type eleTy,
                        mlir::Value innerArg) {
  if (destLoad.typeparams().empty()) {
    fir::factory::genRecordAssignment(builder, loc, destAcc, elementExv);
  } else {
    auto boxTy = fir::BoxType::get(eleTy);
    auto toBox = builder.create<fir::EmboxOp>(loc, boxTy, destAcc.getResult(),
                                              mlir::Value{}, mlir::Value{},
                                              destLoad.typeparams());
    auto fromBox = builder.create<fir::EmboxOp>(
        loc, boxTy, fir::getBase(elementExv), mlir::Value{}, mlir::Value{},
        destLoad.typeparams());
    fir::factory::genRecordAssignment(builder, loc, fir::BoxValue(toBox),
                                      fir::BoxValue(fromBox));
  }
  return builder.create<fir::ArrayAmendOp>(loc, innerArg.getType(), innerArg,
                                           destAcc);
}

inline static fir::ArrayAmendOp
createCharArrayAmend(mlir::Location loc, fir::FirOpBuilder &builder,
                     fir::ArrayAccessOp dstOp, mlir::Value &dstLen,
                     const fir::ExtendedValue &srcExv, mlir::Value innerArg,
                     llvm::ArrayRef<mlir::Value> bounds) {
  fir::CharBoxValue dstChar(dstOp, dstLen);
  fir::factory::CharacterExprHelper helper{builder, loc};
  if (!bounds.empty()) {
    dstChar = helper.createSubstring(dstChar, bounds);
    fir::factory::genCharacterCopy(fir::getBase(srcExv), fir::getLen(srcExv),
                                   dstChar.getAddr(), dstChar.getLen(), builder,
                                   loc);
    // Update the LEN to the substring's LEN.
    dstLen = dstChar.getLen();
  }
  // For a CHARACTER, we generate the element assignment loops inline.
  helper.createAssign(fir::ExtendedValue{dstChar}, srcExv);
  // Mark this array element as amended.
  auto ty = innerArg.getType();
  auto amend = builder.create<fir::ArrayAmendOp>(loc, ty, innerArg, dstOp);
  return amend;
}

/// Build an ExtendedValue from a fir.array<?x...?xT> without actually setting
/// the actual extents and lengths. This is only to allow their propagation as
/// ExtendedValue without triggering verifier failures when propagating
/// character/arrays as unboxed values. Only the base of the resulting
/// ExtendedValue should be used, it is undefined to use the length or extents
/// of the extended value returned,
inline static fir::ExtendedValue
convertToArrayBoxValue(mlir::Location loc, fir::FirOpBuilder &builder,
                       mlir::Value val, mlir::Value len) {
  auto ty = fir::unwrapRefType(val.getType());
  auto idxTy = builder.getIndexType();
  auto seqTy = ty.cast<fir::SequenceType>();
  auto undef = builder.create<fir::UndefOp>(loc, idxTy);
  llvm::SmallVector<mlir::Value> extents(seqTy.getDimension(), undef);
  if (fir::isa_char(seqTy.getEleTy()))
    return fir::CharArrayBoxValue(val, len ? len : undef, extents);
  return fir::ArrayBoxValue(val, extents);
}

//===----------------------------------------------------------------------===//
//
// Lowering of array expressions.
//
//===----------------------------------------------------------------------===//

namespace {
class ArrayExprLowering {
  using ExtValue = fir::ExtendedValue;

  /// Structure to keep track of lowered array operands in the
  /// array expression. Useful to later deduce the shape of the
  /// array expression.
  struct ArrayOperand {
    /// Array base (can be a fir.box).
    mlir::Value memref;
    /// ShapeOp, ShapeShiftOp or ShiftOp
    mlir::Value shape;
    /// SliceOp
    mlir::Value slice;
  };

  using ImplicitSubscripts = Fortran::lower::details::ImplicitSubscripts;
  using PathComponent = Fortran::lower::PathComponent;

  /// Active iteration space.
  using IterationSpace = Fortran::lower::IterationSpace;
  using IterSpace = const Fortran::lower::IterationSpace &;

  /// Current continuation. Function that will generate IR for a single
  /// iteration of the pending iterative loop structure.
  using CC = Fortran::lower::GenerateElementalArrayFunc;

  /// Projection continuation. Function that will project one iteration space
  /// into another.
  using PC = std::function<IterationSpace(IterSpace)>;
  using ArrayBaseTy =
      std::variant<std::monostate, const Fortran::evaluate::ArrayRef *,
                   const Fortran::evaluate::DataRef *>;
  using ComponentPath = Fortran::lower::ComponentPath;

public:
  //===--------------------------------------------------------------------===//
  // Regular array assignment
  //===--------------------------------------------------------------------===//

  /// Entry point for array assignments. Both the left-hand and right-hand sides
  /// can either be ExtendedValue or evaluate::Expr.
  template <typename TL, typename TR>
  static void lowerArrayAssignment(Fortran::lower::AbstractConverter &converter,
                                   Fortran::lower::SymMap &symMap,
                                   Fortran::lower::StatementContext &stmtCtx,
                                   const TL &lhs, const TR &rhs) {
    ArrayExprLowering ael{converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut};
    ael.lowerArrayAssignment(lhs, rhs);
  }

  template <typename TL, typename TR>
  void lowerArrayAssignment(const TL &lhs, const TR &rhs) {
    auto loc = getLoc();
    /// Here the target subspace is not necessarily contiguous. The ArrayUpdate
    /// continuation is implicitly returned in `ccStoreToDest` and the ArrayLoad
    /// in `destination`.
    PushSemantics(ConstituentSemantics::ProjectedCopyInCopyOut);
    ccStoreToDest = genarr(lhs);
    determineShapeOfDest(lhs);
    semant = ConstituentSemantics::RefTransparent;
    auto exv = lowerArrayExpression(rhs);
    if (explicitSpaceIsActive()) {
      explicitSpace->finalizeContext();
      builder.create<fir::ResultOp>(loc, fir::getBase(exv));
    } else {
      builder.create<fir::ArrayMergeStoreOp>(
          loc, destination, fir::getBase(exv), destination.memref(),
          destination.slice(), destination.typeparams());
    }
  }

  //===--------------------------------------------------------------------===//
  // WHERE array assignment, FORALL assignment, and FORALL+WHERE array
  // assignment
  //===--------------------------------------------------------------------===//

  /// Entry point for array assignment when the iteration space is explicitly
  /// defined (Fortran's FORALL) with or without masks, and/or the implied
  /// iteration space involves masks (Fortran's WHERE). Both contexts (explicit
  /// space and implicit space with masks) may be present.
  static void lowerAnyMaskedArrayAssignment(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
      Fortran::lower::ExplicitIterSpace &explicitSpace,
      Fortran::lower::ImplicitIterSpace &implicitSpace) {
    if (explicitSpace.isActive() && lhs.Rank() == 0) {
      // Scalar assignment expression in a FORALL context.
      ArrayExprLowering ael(converter, stmtCtx, symMap,
                            ConstituentSemantics::RefTransparent,
                            &explicitSpace, &implicitSpace);
      ael.lowerScalarAssignment(lhs, rhs);
      return;
    }
    // Array assignment expression in a FORALL and/or WHERE context.
    ArrayExprLowering ael(converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut, &explicitSpace,
                          &implicitSpace);
    ael.lowerArrayAssignment(lhs, rhs);
  }

  //===--------------------------------------------------------------------===//
  // Array assignment to allocatable array
  //===--------------------------------------------------------------------===//

  /// Entry point for assignment to allocatable array.
  static void lowerAllocatableArrayAssignment(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
      Fortran::lower::ExplicitIterSpace &explicitSpace,
      Fortran::lower::ImplicitIterSpace &implicitSpace) {
    ArrayExprLowering ael(converter, stmtCtx, symMap,
                          ConstituentSemantics::CopyInCopyOut, &explicitSpace,
                          &implicitSpace);
    ael.lowerAllocatableArrayAssignment(lhs, rhs);
  }

  /// Assignment to allocatable array.
  ///
  /// The semantics are reverse that of a "regular" array assignment. The rhs
  /// defines the iteration space of the computation and the lhs is
  /// resized/reallocated to fit if necessary.
  void lowerAllocatableArrayAssignment(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs) {
    // With assignment to allocatable, we want to lower the rhs first and use
    // its shape to determine if we need to reallocate, etc.
    auto loc = getLoc();
    // FIXME: If the lhs is in an explicit iteration space, the assignment may
    // be to an array of allocatable arrays rather than a single allocatable
    // array.
    auto mutableBox = createMutableBox(loc, converter, lhs, symMap);
    auto resultTy = converter.genType(rhs);
    auto rhsCC = [&]() {
      PushSemantics(ConstituentSemantics::RefTransparent);
      return genarr(rhs);
    }();
    if (!arrayOperands.empty())
      destShape = getShape(arrayOperands[0]);

    llvm::SmallVector<mlir::Value> lengthParams;
    // Currently no safe way to gather length from rhs (at least for
    // character, it cannot be taken from array_loads since it may be
    // changed by concatenations).
    if ((mutableBox.isCharacter() && !mutableBox.hasNonDeferredLenParams()) ||
        mutableBox.isDerivedWithLengthParameters())
      TODO(loc, "gather rhs length parameters in assignment to allocatable");

    // The allocatable must take lower bounds from the expr if reallocated.
    // An expr has lbounds only if it is an array symbol or component.
    llvm::SmallVector<mlir::Value> lbounds;
    // takeLboundsIfRealloc is only true iff the rhs is a single dataref.
    const bool takeLboundsIfRealloc =
        Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(rhs);
    if (takeLboundsIfRealloc && !arrayOperands.empty()) {
      assert(arrayOperands.size() == 1 &&
             "lbounds can only come from one array");
      auto lbs = fir::factory::getOrigins(arrayOperands[0].shape);
      lbounds.append(lbs.begin(), lbs.end());
    }
    auto realloc = fir::factory::genReallocIfNeeded(builder, loc, mutableBox,
                                                    destShape, lengthParams);
    // Create ArrayLoad for the mutable box and save it into `destination`.
    PushSemantics(ConstituentSemantics::ProjectedCopyInCopyOut);
    ccStoreToDest = genarr(realloc.newValue);
    // If the rhs is scalar, get shape from the allocatable ArrayLoad.
    if (destShape.empty())
      destShape = getShape(destination);
    // Finish lowering the loop nest.
    assert(destination && "destination must have been set");
    auto exv = lowerArrayExpression(rhsCC, resultTy);
    if (explicitSpaceIsActive()) {
      explicitSpace->finalizeContext();
      builder.create<fir::ResultOp>(loc, fir::getBase(exv));
    } else {
      builder.create<fir::ArrayMergeStoreOp>(
          loc, destination, fir::getBase(exv), destination.memref(),
          destination.slice(), destination.typeparams());
    }
    fir::factory::finalizeRealloc(builder, loc, mutableBox, lbounds,
                                  takeLboundsIfRealloc, realloc);
  }

  /// Entry point for when an array expression appears in a context where the
  /// result must be boxed. (BoxValue semantics.)
  static ExtValue lowerBoxedArrayExpression(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
    ArrayExprLowering ael{converter, stmtCtx, symMap,
                          ConstituentSemantics::BoxValue};
    return ael.lowerBoxedArrayExpr(expr);
  }

  ExtValue lowerBoxedArrayExpr(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &exp) {
    return std::visit(
        [&](const auto &e) {
          auto f = genarr(e);
          auto exv = f(IterationSpace{});
          if (fir::getBase(exv).getType().template isa<fir::BoxType>())
            return exv;
          fir::emitFatalError(getLoc(), "array must be emboxed");
        },
        exp.u);
  }

  /// Entry point into lowering an expression with rank. This entry point is for
  /// lowering a rhs expression, for example. (RefTransparent semantics.)
  static ExtValue
  lowerNewArrayExpression(Fortran::lower::AbstractConverter &converter,
                          Fortran::lower::SymMap &symMap,
                          Fortran::lower::StatementContext &stmtCtx,
                          const Fortran::lower::SomeExpr &expr) {
    ArrayExprLowering ael{converter, stmtCtx, symMap};
    ael.determineShapeOfDest(expr);
    auto loopRes = ael.lowerArrayExpression(expr);
    auto dest = ael.destination;
    auto tempRes = dest.memref();
    auto &builder = converter.getFirOpBuilder();
    auto loc = converter.getCurrentLocation();
    builder.create<fir::ArrayMergeStoreOp>(loc, dest, fir::getBase(loopRes),
                                           tempRes, dest.slice(),
                                           dest.typeparams());

    auto arrTy =
        fir::dyn_cast_ptrEleTy(tempRes.getType()).cast<fir::SequenceType>();
    if (auto charTy =
            arrTy.getEleTy().template dyn_cast<fir::CharacterType>()) {
      if (fir::characterWithDynamicLen(charTy))
        TODO(loc, "CHARACTER does not have constant LEN");
      auto len = builder.createIntegerConstant(
          loc, builder.getCharacterLengthType(), charTy.getLen());
      return fir::CharArrayBoxValue(tempRes, len, dest.getExtents());
    }
    return fir::ArrayBoxValue(tempRes, dest.getExtents());
  }

  static void lowerLazyArrayExpression(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
      mlir::Value raggedHeader) {
    ArrayExprLowering ael(converter, stmtCtx, symMap);
    ael.lowerLazyArrayExpression(expr, raggedHeader);
  }

  /// Lower the expression \p expr into a buffer that is created on demand. The
  /// variable containing the pointer to the buffer is \p var and the variable
  /// containing the shape of the buffer is \p shapeBuffer.
  void lowerLazyArrayExpression(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
      mlir::Value header) {
    auto loc = getLoc();
    auto hdrTy = fir::factory::getRaggedArrayHeaderType(builder);
    auto i32Ty = builder.getIntegerType(32);

    // Once the loop extents have been computed, which may require being inside
    // some explicit loops, lazily allocate the expression on the heap. The
    // following continuation creates the buffer as needed.
    ccPrelude = [=](llvm::ArrayRef<mlir::Value> shape) {
      auto i64Ty = builder.getIntegerType(64);
      auto byteSize = builder.createIntegerConstant(loc, i64Ty, 1);
      fir::runtime::genRaggedArrayAllocate(
          loc, builder, header, /*asHeaders=*/false, byteSize, shape);
    };

    // Create a dummy array_load before the loop. We're storing to a lazy
    // temporary, so there will be no conflict and no copy-in. TODO: skip this
    // as there isn't any necessity for it.
    ccLoadDest = [=](llvm::ArrayRef<mlir::Value> shape) -> fir::ArrayLoadOp {
      auto one = builder.createIntegerConstant(loc, i32Ty, 1);
      auto var = builder.create<fir::CoordinateOp>(
          loc, builder.getRefType(hdrTy.getType(1)), header, one);
      auto load = builder.create<fir::LoadOp>(loc, var);
      auto eleTy = fir::unwrapSequenceType(fir::unwrapRefType(load.getType()));
      auto seqTy = fir::SequenceType::get(eleTy, shape.size());
      auto castTo = builder.createConvert(loc, fir::HeapType::get(seqTy), load);
      auto shapeOp = builder.genShape(loc, shape);
      return builder.create<fir::ArrayLoadOp>(
          loc, seqTy, castTo, shapeOp, /*slice=*/mlir::Value{}, llvm::None);
    };
    // Custom lowering of the element store to deal with the extra indirection
    // to the lazy allocated buffer.
    ccStoreToDest = [=](IterSpace iters) {
      auto one = builder.createIntegerConstant(loc, i32Ty, 1);
      auto var = builder.create<fir::CoordinateOp>(
          loc, builder.getRefType(hdrTy.getType(1)), header, one);
      auto load = builder.create<fir::LoadOp>(loc, var);
      auto eleTy = fir::unwrapSequenceType(fir::unwrapRefType(load.getType()));
      auto seqTy = fir::SequenceType::get(eleTy, iters.iterVec().size());
      auto toTy = fir::HeapType::get(seqTy);
      auto castTo = builder.createConvert(loc, toTy, load);
      auto shape = builder.genShape(loc, genIterationShape());
      auto indices = fir::factory::originateIndices(
          loc, builder, castTo.getType(), shape, iters.iterVec());
      auto eleAddr = builder.create<fir::ArrayCoorOp>(
          loc, builder.getRefType(eleTy), castTo, shape,
          /*slice=*/mlir::Value{}, indices, destination.typeparams());
      auto eleVal = builder.createConvert(loc, eleTy, iters.getElement());
      builder.create<fir::StoreOp>(loc, eleVal, eleAddr);
      return iters.innerArgument();
    };

    // Lower the array expression now.
    [[maybe_unused]] auto loopRes = lowerArrayExpression(expr);
    assert(fir::getBase(loopRes));
  }

  static void
  lowerElementalUserAssignment(Fortran::lower::AbstractConverter &converter,
                               Fortran::lower::SymMap &symMap,
                               Fortran::lower::StatementContext &stmtCtx,
                               Fortran::lower::ExplicitIterSpace &explicitSpace,
                               Fortran::lower::ImplicitIterSpace &implicitSpace,
                               const Fortran::evaluate::ProcedureRef &procRef) {
    ArrayExprLowering ael(converter, stmtCtx, symMap,
                          ConstituentSemantics::CustomCopyInCopyOut,
                          &explicitSpace, &implicitSpace);
    assert(procRef.arguments().size() == 2);
    const auto *lhs = procRef.arguments()[0].value().UnwrapExpr();
    const auto *rhs = procRef.arguments()[1].value().UnwrapExpr();
    assert(lhs && rhs &&
           "user defined assignment arguments must be expressions");
    auto func = Fortran::lower::CallerInterface(procRef, converter).getFuncOp();
    ael.lowerElementalUserAssignment(func, *lhs, *rhs);
  }

  void lowerElementalUserAssignment(
      mlir::FuncOp userAssignment,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs) {
    auto loc = getLoc();
    PushSemantics(ConstituentSemantics::CustomCopyInCopyOut);
    auto genArrayModify = genarr(lhs);
    ccStoreToDest = [=](IterSpace iters) -> ExtValue {
      auto modifiedArray = genArrayModify(iters);
      auto arrayModify = mlir::dyn_cast_or_null<fir::ArrayModifyOp>(
          fir::getBase(modifiedArray).getDefiningOp());
      assert(arrayModify && "must be created by ArrayModifyOp");
      auto lhs =
          arrayModifyToExv(builder, loc, destination, arrayModify.getResult(0));
      genScalarUserDefinedAssignmentCall(builder, loc, userAssignment, lhs,
                                         iters.elementExv());
      return modifiedArray;
    };
    determineShapeOfDest(lhs);
    semant = ConstituentSemantics::RefTransparent;
    auto exv = lowerArrayExpression(rhs);
    if (explicitSpaceIsActive()) {
      explicitSpace->finalizeContext();
      builder.create<fir::ResultOp>(loc, fir::getBase(exv));
    } else {
      builder.create<fir::ArrayMergeStoreOp>(
          loc, destination, fir::getBase(exv), destination.memref(),
          destination.slice(), destination.typeparams());
    }
  }

  /// Lower an elemental subroutine call with at least one array argument.
  /// An elemental subroutine is an exception and does not have copy-in/copy-out
  /// semantics. See 15.8.3.
  /// Do NOT use this for user defined assignments.
  static void
  lowerElementalSubroutine(Fortran::lower::AbstractConverter &converter,
                           Fortran::lower::SymMap &symMap,
                           Fortran::lower::StatementContext &stmtCtx,
                           const Fortran::lower::SomeExpr &call) {
    ArrayExprLowering ael(converter, stmtCtx, symMap,
                          ConstituentSemantics::RefTransparent);
    ael.lowerElementalSubroutine(call);
  }

  // TODO: See the comment in genarr(const Fortran::lower::Parentheses<T>&).
  // This is skipping generation of copy-in/copy-out code for analysis that is
  // required when arguments are in parentheses.
  void lowerElementalSubroutine(const Fortran::lower::SomeExpr &call) {
    auto f = genarr(call);
    auto shape = genIterationShape();
    auto [iterSpace, insPt] = genImplicitLoops(shape, /*innerArg=*/{});
    f(iterSpace);
    builder.restoreInsertionPoint(insPt);
  }

  template <typename A, typename B>
  ExtValue lowerScalarAssignment(const A &lhs, const B &rhs) {
    // 1) Lower the rhs expression with array_fetch op(s).
    IterationSpace iters;
    iters.setElement(genarr(rhs)(iters));
    auto elementalExv = iters.elementExv();
    // 2) Lower the lhs expression to an array_update.
    semant = ConstituentSemantics::ProjectedCopyInCopyOut;
    auto lexv = genarr(lhs)(iters);
    // 3) Finalize the inner context.
    explicitSpace->finalizeContext();
    // 4) Thread the array value updated forward. Note: the lhs might be
    // ill-formed (performing scalar assignment in an array context),
    // in which case there is no array to thread.
    auto createResult = [&](auto op) {
      auto oldInnerArg = op.sequence();
      auto offset = explicitSpace->argPosition(oldInnerArg);
      explicitSpace->setInnerArg(offset, fir::getBase(lexv));
      builder.create<fir::ResultOp>(getLoc(), fir::getBase(lexv));
    };
    if (auto updateOp = mlir::dyn_cast<fir::ArrayUpdateOp>(
            fir::getBase(lexv).getDefiningOp()))
      createResult(updateOp);
    else if (auto amend = mlir::dyn_cast<fir::ArrayAmendOp>(
                 fir::getBase(lexv).getDefiningOp()))
      createResult(amend);
    else if (auto modifyOp = mlir::dyn_cast<fir::ArrayModifyOp>(
                 fir::getBase(lexv).getDefiningOp()))
      createResult(modifyOp);
    return lexv;
  }

  static ExtValue lowerScalarUserAssignment(
      Fortran::lower::AbstractConverter &converter,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      Fortran::lower::ExplicitIterSpace &explicitIterSpace,
      mlir::FuncOp userAssignmentFunction, const Fortran::lower::SomeExpr &lhs,
      const Fortran::lower::SomeExpr &rhs) {
    Fortran::lower::ImplicitIterSpace implicit;
    ArrayExprLowering ael(converter, stmtCtx, symMap,
                          ConstituentSemantics::RefTransparent,
                          &explicitIterSpace, &implicit);
    return ael.lowerScalarUserAssignment(userAssignmentFunction, lhs, rhs);
  }

  ExtValue lowerScalarUserAssignment(mlir::FuncOp userAssignment,
                                     const Fortran::lower::SomeExpr &lhs,
                                     const Fortran::lower::SomeExpr &rhs) {
    auto loc = getLoc();
    if (rhs.Rank() > 0)
      TODO(loc, "user-defined elemental assigment from expression with rank");
    // 1) Lower the rhs expression with array_fetch op(s).
    IterationSpace iters;
    iters.setElement(genarr(rhs)(iters));
    auto elementalExv = iters.elementExv();
    // 2) Lower the lhs expression to an array_modify.
    semant = ConstituentSemantics::CustomCopyInCopyOut;
    auto lexv = genarr(lhs)(iters);
    bool isIllFormedLHS = false;
    // 3) Insert the call
    if (auto modifyOp = mlir::dyn_cast<fir::ArrayModifyOp>(
            fir::getBase(lexv).getDefiningOp())) {
      auto oldInnerArg = modifyOp.sequence();
      auto offset = explicitSpace->argPosition(oldInnerArg);
      explicitSpace->setInnerArg(offset, fir::getBase(lexv));
      auto exv = arrayModifyToExv(builder, loc,
                                  explicitSpace->getLhsLoad(0).getValue(),
                                  modifyOp.getResult(0));
      genScalarUserDefinedAssignmentCall(builder, loc, userAssignment, exv,
                                         elementalExv);
    } else {
      // LHS is ill formed, it is a scalar with no references to FORALL
      // subscripts, so there is actually no array assignment here. The user
      // code is probably bad, but still insert user assignment call since it
      // was not rejected by semantics (a warning was emitted).
      isIllFormedLHS = true;
      genScalarUserDefinedAssignmentCall(builder, getLoc(), userAssignment,
                                         lexv, elementalExv);
    }
    // 4) Finalize the inner context.
    explicitSpace->finalizeContext();
    // 5). Thread the array value updated forward.
    if (!isIllFormedLHS)
      builder.create<fir::ResultOp>(getLoc(), fir::getBase(lexv));
    return lexv;
  }

private:
  void determineShapeOfDest(const fir::ExtendedValue &lhs) {
    destShape = fir::factory::getExtents(builder, getLoc(), lhs);
  }

  void determineShapeOfDest(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs) {
    if (!destShape.empty())
      return;
    if (explicitSpaceIsActive() && determineShapeWithSlice(lhs))
      return;
    if (auto shape =
            Fortran::evaluate::GetShape(converter.getFoldingContext(), lhs))
      convertFEShape(shape.value(), destShape);
  }

  bool genShapeFromDataRef(const Fortran::semantics::Symbol &x) {
    return false;
  }
  bool genShapeFromDataRef(const Fortran::evaluate::CoarrayRef &) {
    TODO(getLoc(), "coarray ref");
    return false;
  }
  bool genShapeFromDataRef(const Fortran::evaluate::Component &x) {
    return x.base().Rank() > 0 ? genShapeFromDataRef(x.base()) : false;
  }
  bool genShapeFromDataRef(const Fortran::evaluate::ArrayRef &x) {
    if (x.Rank() == 0)
      return false;
    if (x.base().Rank() > 0)
      if (genShapeFromDataRef(x.base()))
        return true;
    // x has rank and x.base did not produce a shape.
    auto exv = x.base().IsSymbol() ? asScalarRef(x.base().GetFirstSymbol())
                                   : asScalarRef(x.base().GetComponent());
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto definedShape = fir::factory::getExtents(builder, loc, exv);
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    for (auto ss : llvm::enumerate(x.subscript())) {
      std::visit(Fortran::common::visitors{
                     [&](const Fortran::evaluate::Triplet &trip) {
                       // For a subscript of triple notation, we compute the
                       // range of this dimension of the iteration space.
                       auto lo = [&]() {
                         if (auto optLo = trip.lower())
                           return fir::getBase(asScalar(*optLo));
                         return getLBound(exv, ss.index(), one);
                       }();
                       auto hi = [&]() {
                         if (auto optHi = trip.upper())
                           return fir::getBase(asScalar(*optHi));
                         return getUBound(exv, ss.index(), one);
                       }();
                       auto step = builder.createConvert(
                           loc, idxTy, fir::getBase(asScalar(trip.stride())));
                       auto extent = builder.genExtentFromTriplet(loc, lo, hi,
                                                                  step, idxTy);
                       destShape.push_back(extent);
                     },
                     [&](auto) {}},
                 ss.value().u);
    }
    return true;
  }
  bool genShapeFromDataRef(const Fortran::evaluate::NamedEntity &x) {
    if (x.IsSymbol())
      return genShapeFromDataRef(x.GetFirstSymbol());
    return genShapeFromDataRef(x.GetComponent());
  }
  bool genShapeFromDataRef(const Fortran::evaluate::DataRef &x) {
    return std::visit([&](const auto &v) { return genShapeFromDataRef(v); },
                      x.u);
  }

  /// When in an explicit space, the ranked component must be evaluated to
  /// determine the actual number of iterations when slicing triples are
  /// present. Lower these expressions here.
  bool determineShapeWithSlice(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs) {
    LLVM_DEBUG(Fortran::lower::DumpEvaluateExpr::dump(
        llvm::dbgs() << "determine shape of:\n", lhs));
    // FIXME: We may not want to use ExtractDataRef here since it doesn't deal
    // with substrings, etc.
    auto dref = Fortran::evaluate::ExtractDataRef(lhs);
    return dref.has_value() ? genShapeFromDataRef(*dref) : false;
  }

  /// Returns true iff the Ev::Shape is constant.
  static bool evalShapeIsConstant(const Fortran::evaluate::Shape &shape) {
    for (const auto &s : shape)
      if (!s || !Fortran::evaluate::IsConstantExpr(*s))
        return false;
    return true;
  }

  /// Convert an Ev::Shape to IR values.
  void convertFEShape(const Fortran::evaluate::Shape &shape,
                      llvm::SmallVectorImpl<mlir::Value> &result) {
    if (evalShapeIsConstant(shape)) {
      auto idxTy = builder.getIndexType();
      auto loc = getLoc();
      for (const auto &s : shape)
        result.emplace_back(builder.createConvert(
            loc, idxTy, convertOptExtentExpr(converter, stmtCtx, s)));
    }
  }

  /// CHARACTER and derived type elements are treated as memory references. The
  /// numeric types are treated as values.
  static mlir::Type adjustedArraySubtype(mlir::Type ty,
                                         mlir::ValueRange indices) {
    auto pathTy = fir::applyPathToType(ty, indices);
    assert(pathTy && "indices failed to apply to type");
    return adjustedArrayElementType(pathTy);
  }

  ExtValue lowerArrayExpression(
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &exp) {
    auto resTy = converter.genType(exp);
    return std::visit(
        [&](const auto &e) { return lowerArrayExpression(genarr(e), resTy); },
        exp.u);
  }
  ExtValue lowerArrayExpression(const ExtValue &exv) {
    assert(!explicitSpace);
    auto resTy = fir::unwrapPassByRefType(fir::getBase(exv).getType());
    return lowerArrayExpression(genarr(exv), resTy);
  }

  void populateBounds(llvm::SmallVectorImpl<mlir::Value> &bounds,
                      const Fortran::evaluate::Substring *substring) {
    if (!substring)
      return;
    bounds.push_back(fir::getBase(asScalar(substring->lower())));
    if (auto upper = substring->upper())
      bounds.push_back(fir::getBase(asScalar(*upper)));
  }

  /// Default store to destination implementation.
  /// This implements the default case, which is to assign the value in
  /// `iters.element` into the destination array, `iters.innerArgument`. Handles
  /// by value and by reference assignment.
  CC defaultStoreToDestination(const Fortran::evaluate::Substring *substring) {
    return [=](IterSpace iterSpace) -> ExtValue {
      auto loc = getLoc();
      auto innerArg = iterSpace.innerArgument();
      auto exv = iterSpace.elementExv();
      auto arrTy = innerArg.getType();
      auto eleTy = fir::applyPathToType(arrTy, iterSpace.iterVec());
      if (isAdjustedArrayElementType(eleTy)) {
        // The elemental update is in the memref domain. Under this semantics,
        // we must always copy the computed new element from its location in
        // memory into the destination array.
        auto resRefTy = builder.getRefType(eleTy);
        // Get a reference to the array element to be amended.
        auto arrayOp = builder.create<fir::ArrayAccessOp>(
            loc, resRefTy, innerArg, iterSpace.iterVec(),
            destination.typeparams());
        if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
          llvm::SmallVector<mlir::Value> substringBounds;
          populateBounds(substringBounds, substring);
          auto dstLen = fir::factory::genLenOfCharacter(
              builder, loc, destination, iterSpace.iterVec(), substringBounds);
          auto amend = createCharArrayAmend(loc, builder, arrayOp, dstLen, exv,
                                            innerArg, substringBounds);
          return abstractArrayExtValue(amend, dstLen);
        }
        if (fir::isa_derived(eleTy)) {
          auto amend = createDerivedArrayAmend(loc, destination, builder,
                                               arrayOp, exv, eleTy, innerArg);
          return abstractArrayExtValue(amend /*FIXME: typeparams?*/);
        }
        assert(eleTy.isa<fir::SequenceType>() && "must be an array");
        TODO(loc, "array (as element) assignment");
      }
      // By value semantics. The element is being assigned by value.
      auto ele = builder.createConvert(loc, eleTy, fir::getBase(exv));
      auto update = builder.create<fir::ArrayUpdateOp>(
          loc, arrTy, innerArg, ele, iterSpace.iterVec(),
          destination.typeparams());
      return abstractArrayExtValue(update);
    };
  }

  /// For an elemental array expression.
  ///   1. Lower the scalars and array loads.
  ///   2. Create the iteration space.
  ///   3. Create the element-by-element computation in the loop.
  ///   4. Return the resulting array value.
  /// If no destination was set in the array context, a temporary of
  /// \p resultTy will be created to hold the evaluated expression.
  /// Otherwise, \p resultTy is ignored and the expression is evaluated
  /// in the destination. \p f is a continuation built from an
  /// evaluate::Expr or an ExtendedValue.
  ExtValue lowerArrayExpression(CC f, mlir::Type resultTy) {
    auto loc = getLoc();
    auto [iterSpace, insPt] = genIterSpace(resultTy);
    auto exv = f(iterSpace);
    iterSpace.setElement(std::move(exv));
    auto lambda = ccStoreToDest.hasValue()
                      ? ccStoreToDest.getValue()
                      : defaultStoreToDestination(/*substring=*/nullptr);
    auto updVal = fir::getBase(lambda(iterSpace));
    finalizeElementCtx();
    builder.create<fir::ResultOp>(loc, updVal);
    builder.restoreInsertionPoint(insPt);
    return abstractArrayExtValue(iterSpace.outerResult());
  }

  /// Compute the shape of a slice.
  llvm::SmallVector<mlir::Value> computeSliceShape(mlir::Value slice) {
    llvm::SmallVector<mlir::Value> slicedShape;
    auto slOp = mlir::cast<fir::SliceOp>(slice.getDefiningOp());
    auto triples = slOp.triples();
    auto idxTy = builder.getIndexType();
    auto loc = getLoc();
    for (unsigned i = 0, end = triples.size(); i < end; i += 3) {
      if (!mlir::isa_and_nonnull<fir::UndefOp>(
              triples[i + 1].getDefiningOp())) {
        // (..., lb:ub:step, ...) case:  extent = max((ub-lb+step)/step, 0)
        // See Fortran 2018 9.5.3.3.2 section for more details.
        auto res = builder.genExtentFromTriplet(loc, triples[i], triples[i + 1],
                                                triples[i + 2], idxTy);
        slicedShape.emplace_back(res);
      } else {
        // do nothing. `..., i, ...` case, so dimension is dropped.
      }
    }
    return slicedShape;
  }

  /// Get the shape from an ArrayOperand. The shape of the array is adjusted if
  /// the array was sliced.
  llvm::SmallVector<mlir::Value> getShape(ArrayOperand array) {
    if (array.slice)
      return computeSliceShape(array.slice);
    if (array.memref.getType().isa<fir::BoxType>())
      return fir::factory::readExtents(builder, getLoc(),
                                       fir::BoxValue{array.memref});
    auto extents = fir::factory::getExtents(array.shape);
    return {extents.begin(), extents.end()};
  }

  /// Get the shape from an ArrayLoad.
  llvm::SmallVector<mlir::Value> getShape(fir::ArrayLoadOp arrayLoad) {
    return getShape(
        ArrayOperand{arrayLoad.memref(), arrayLoad.shape(), arrayLoad.slice()});
  }

  /// Generate the shape of the iteration space over the array expression. The
  /// iteration space may be implicit, explicit, or both. If it is implied it is
  /// based on the destination and operand array loads, or an optional
  /// Fortran::evaluate::Shape from the front end. If the shape is explicit,
  /// this returns any implicit shape component, if it exists.
  llvm::SmallVector<mlir::Value> genIterationShape() {
    // Use the precomputed destination shape.
    if (!destShape.empty())
      return destShape;
    // Otherwise, use the destination's shape.
    if (destination)
      return getShape(destination);
    // Otherwise, use the first ArrayLoad operand shape.
    if (!arrayOperands.empty())
      return getShape(arrayOperands[0]);
    fir::emitFatalError(getLoc(),
                        "failed to compute the array expression shape");
  }

  bool explicitSpaceIsActive() const {
    return explicitSpace && explicitSpace->isActive();
  }

  bool implicitSpaceHasMasks() const {
    return implicitSpace && !implicitSpace->empty();
  }

  CC genMaskAccess(mlir::Value tmp, mlir::Value shape) {
    auto loc = getLoc();
    return [=, builder = &converter.getFirOpBuilder()](IterSpace iters) {
      auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(tmp.getType());
      auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
      auto eleRefTy = builder->getRefType(eleTy);
      auto i1Ty = builder->getI1Type();
      // Adjust indices for any shift of the origin of the array.
      auto indices = fir::factory::originateIndices(
          loc, *builder, tmp.getType(), shape, iters.iterVec());
      auto addr = builder->create<fir::ArrayCoorOp>(
          loc, eleRefTy, tmp, shape, /*slice=*/mlir::Value{}, indices,
          /*typeParams=*/llvm::None);
      auto load = builder->create<fir::LoadOp>(loc, addr);
      return builder->createConvert(loc, i1Ty, load);
    };
  }

  /// Construct the incremental instantiations of the ragged array structure.
  /// Rebind the lazy buffer variable, etc. as we go.
  template <bool withAllocation = false>
  mlir::Value prepareRaggedArrays(Fortran::lower::FrontEndExpr expr) {
    assert(explicitSpaceIsActive());
    auto loc = getLoc();
    auto raggedTy = fir::factory::getRaggedArrayHeaderType(builder);
    auto loopStack = explicitSpace->getLoopStack();
    const auto depth = loopStack.size();
    auto i64Ty = builder.getIntegerType(64);
    [[maybe_unused]] auto byteSize =
        builder.createIntegerConstant(loc, i64Ty, 1);
    auto header = implicitSpace->lookupMaskHeader(expr);
    for (std::remove_const_t<decltype(depth)> i = 0; i < depth; ++i) {
      auto insPt = builder.saveInsertionPoint();
      if (i < depth - 1)
        builder.setInsertionPoint(loopStack[i + 1][0]);

      // Compute and gather the extents.
      llvm::SmallVector<mlir::Value> extents;
      for (auto doLoop : loopStack[i])
        extents.push_back(builder.genExtentFromTriplet(loc, doLoop.lowerBound(),
                                                       doLoop.upperBound(),
                                                       doLoop.step(), i64Ty));
      if constexpr (withAllocation) {
        fir::runtime::genRaggedArrayAllocate(
            loc, builder, header, /*asHeader=*/true, byteSize, extents);
      }

      // Compute the dynamic position into the header.
      llvm::SmallVector<mlir::Value> offsets;
      for (auto doLoop : loopStack[i]) {
        auto m = builder.create<mlir::arith::SubIOp>(
            loc, doLoop.getInductionVar(), doLoop.lowerBound());
        auto n = builder.create<mlir::arith::DivSIOp>(loc, m, doLoop.step());
        auto one = builder.createIntegerConstant(loc, n.getType(), 1);
        offsets.push_back(builder.create<mlir::arith::AddIOp>(loc, n, one));
      }
      auto i32Ty = builder.getIntegerType(32);
      auto uno = builder.createIntegerConstant(loc, i32Ty, 1);
      auto coorTy = builder.getRefType(raggedTy.getType(1));
      auto hdOff = builder.create<fir::CoordinateOp>(loc, coorTy, header, uno);
      auto toTy = fir::SequenceType::get(raggedTy, offsets.size());
      auto toRefTy = builder.getRefType(toTy);
      auto ldHdr = builder.create<fir::LoadOp>(loc, hdOff);
      auto hdArr = builder.createConvert(loc, toRefTy, ldHdr);
      auto shapeOp = builder.genShape(loc, extents);
      header = builder.create<fir::ArrayCoorOp>(
          loc, builder.getRefType(raggedTy), hdArr, shapeOp,
          /*slice=*/mlir::Value{}, offsets,
          /*typeparams=*/mlir::ValueRange{});
      auto hdrVar = builder.create<fir::CoordinateOp>(loc, coorTy, header, uno);
      auto inVar = builder.create<fir::LoadOp>(loc, hdrVar);
      auto two = builder.createIntegerConstant(loc, i32Ty, 2);
      auto coorTy2 = builder.getRefType(raggedTy.getType(2));
      auto hdrSh = builder.create<fir::CoordinateOp>(loc, coorTy2, header, two);
      auto shapePtr = builder.create<fir::LoadOp>(loc, hdrSh);
      // Replace the binding.
      implicitSpace->rebind(expr, genMaskAccess(inVar, shapePtr));
      if (i < depth - 1)
        builder.restoreInsertionPoint(insPt);
    }
    return header;
  }

  /// Lower mask expressions with implied iteration spaces from the variants of
  /// WHERE syntax. Since it is legal for mask expressions to have side-effects
  /// and modify values that will be used for the lhs, rhs, or both of
  /// subsequent assignments, the mask must be evaluated before the assignment
  /// is processed.
  /// Mask expressions are array expressions too.
  void genMasks() {
    // Lower the mask expressions, if any.
    if (implicitSpaceHasMasks()) {
      auto loc = getLoc();
      // Mask expressions are array expressions too.
      for (const auto *e : implicitSpace->getExprs())
        if (e && !implicitSpace->isLowered(e)) {
          if (auto var = implicitSpace->lookupMaskVariable(e)) {
            // Allocate the mask buffer lazily.
            assert(explicitSpaceIsActive());
            auto header = prepareRaggedArrays</*withAllocations=*/true>(e);
            Fortran::lower::createLazyArrayTempValue(converter, *e, header,
                                                     symMap, stmtCtx);
            // Close the explicit loops.
            builder.create<fir::ResultOp>(loc, explicitSpace->getInnerArgs());
            builder.setInsertionPointAfter(explicitSpace->getOuterLoop());
            // Open a new copy of the explicit loop nest.
            explicitSpace->genLoopNest();
            continue;
          }
          auto tmp = Fortran::lower::createSomeArrayTempValue(converter, *e,
                                                              symMap, stmtCtx);
          auto shape = builder.createShape(loc, tmp);
          implicitSpace->bind(e, genMaskAccess(fir::getBase(tmp), shape));
        }

      // Set buffer from the header.
      for (const auto *e : implicitSpace->getExprs()) {
        if (!e)
          continue;
        if (implicitSpace->lookupMaskVariable(e)) {
          // Index into the ragged buffer to retrieve cached results.
          const auto rank = e->Rank();
          assert(destShape.empty() ||
                 static_cast<std::size_t>(rank) == destShape.size());
          auto header = prepareRaggedArrays(e);
          auto raggedTy = fir::factory::getRaggedArrayHeaderType(builder);
          auto i32Ty = builder.getIntegerType(32);
          auto one = builder.createIntegerConstant(loc, i32Ty, 1);
          auto coor1 = builder.create<fir::CoordinateOp>(
              loc, builder.getRefType(raggedTy.getType(1)), header, one);
          auto db = builder.create<fir::LoadOp>(loc, coor1);
          auto eleTy =
              fir::unwrapSequenceType(fir::unwrapRefType(db.getType()));
          auto buffTy = builder.getRefType(fir::SequenceType::get(eleTy, rank));
          // Address of ragged buffer data.
          auto buff = builder.createConvert(loc, buffTy, db);

          auto two = builder.createIntegerConstant(loc, i32Ty, 2);
          auto coor2 = builder.create<fir::CoordinateOp>(
              loc, builder.getRefType(raggedTy.getType(2)), header, two);
          auto shBuff = builder.create<fir::LoadOp>(loc, coor2);
          auto i64Ty = builder.getIntegerType(64);
          auto idxTy = builder.getIndexType();
          llvm::SmallVector<mlir::Value> extents;
          for (std::remove_const_t<decltype(rank)> i = 0; i < rank; ++i) {
            auto off = builder.createIntegerConstant(loc, i32Ty, i);
            auto coor = builder.create<fir::CoordinateOp>(
                loc, builder.getRefType(i64Ty), shBuff, off);
            auto ldExt = builder.create<fir::LoadOp>(loc, coor);
            extents.push_back(builder.createConvert(loc, idxTy, ldExt));
          }
          if (destShape.empty())
            destShape = extents;
          // Construct shape of buffer.
          auto shapeOp = builder.genShape(loc, extents);

          // Replace binding with the local result.
          implicitSpace->rebind(e, genMaskAccess(buff, shapeOp));
        }
      }
    }
  }

  // FIXME: should take multiple inner arguments.
  std::pair<IterationSpace, mlir::OpBuilder::InsertPoint>
  genImplicitLoops(mlir::ValueRange shape, mlir::Value innerArg) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    auto zero = builder.createIntegerConstant(loc, idxTy, 0);
    llvm::SmallVector<mlir::Value> loopUppers;

    // Convert any implied shape to closed interval form. The fir.do_loop will
    // run from 0 to `extent - 1` inclusive.
    for (auto extent : shape)
      loopUppers.push_back(
          builder.create<mlir::arith::SubIOp>(loc, extent, one));

    // Iteration space is created with outermost columns, innermost rows
    llvm::SmallVector<fir::DoLoopOp> loops;

    const auto loopDepth = loopUppers.size();
    llvm::SmallVector<mlir::Value> ivars;

    for (auto i : llvm::enumerate(llvm::reverse(loopUppers))) {
      if (i.index() > 0) {
        assert(!loops.empty());
        builder.setInsertionPointToStart(loops.back().getBody());
      }
      fir::DoLoopOp loop;
      if (innerArg) {
        loop = builder.create<fir::DoLoopOp>(
            loc, zero, i.value(), one, isUnordered(),
            /*finalCount=*/false, mlir::ValueRange{innerArg});
        innerArg = loop.getRegionIterArgs().front();
        if (explicitSpaceIsActive())
          explicitSpace->setInnerArg(0, innerArg);
      } else {
        loop = builder.create<fir::DoLoopOp>(loc, zero, i.value(), one,
                                             isUnordered(),
                                             /*finalCount=*/false);
      }
      ivars.push_back(loop.getInductionVar());
      loops.push_back(loop);
    }

    if (innerArg)
      for (std::remove_const_t<decltype(loopDepth)> i = 0; i + 1 < loopDepth;
           ++i) {
        builder.setInsertionPointToEnd(loops[i].getBody());
        builder.create<fir::ResultOp>(loc, loops[i + 1].getResult(0));
      }

    // Move insertion point to the start of the innermost loop in the nest.
    builder.setInsertionPointToStart(loops.back().getBody());
    // Set `afterLoopNest` to just after the entire loop nest.
    auto currPt = builder.saveInsertionPoint();
    builder.setInsertionPointAfter(loops[0]);
    auto afterLoopNest = builder.saveInsertionPoint();
    builder.restoreInsertionPoint(currPt);

    // Put the implicit loop variables in row to column order to match FIR's
    // Ops. (The loops were constructed from outermost column to innermost
    // row.)
    mlir::Value outerRes = loops[0].getResult(0);
    return {IterationSpace(innerArg, outerRes, llvm::reverse(ivars)),
            afterLoopNest};
  }

  /// Build the iteration space into which the array expression will be
  /// lowered. The resultType is used to create a temporary, if needed.
  std::pair<IterationSpace, mlir::OpBuilder::InsertPoint>
  genIterSpace(mlir::Type resultType) {
    auto loc = getLoc();
    auto shape = genIterationShape();
    if (!destination) {
      // Allocate storage for the result if it is not already provided.
      destination = createAndLoadSomeArrayTemp(resultType, shape);
    }

    // Generate the lazy mask allocation, if one was given.
    if (ccPrelude.hasValue())
      ccPrelude.getValue()(shape);

    // Now handle the implicit loops.
    auto inner = explicitSpaceIsActive() ? explicitSpace->getInnerArgs().front()
                                         : destination.getResult();
    auto [iters, afterLoopNest] = genImplicitLoops(shape, inner);
    auto innerArg = iters.innerArgument();

    // Generate the mask conditional structure, if there are masks. Unlike the
    // explicit masks, which are interleaved, these mask expression appear in
    // the innermost loop.
    if (implicitSpaceHasMasks()) {
      // Recover the cached condition from the mask buffer.
      auto genCond = [&](Fortran::lower::FrontEndExpr e, IterSpace iters) {
        return implicitSpace->getBoundClosure(e)(iters);
      };

      // Handle the negated conditions in topological order of the WHERE
      // clauses. See 10.2.3.2p4 as to why this control structure is produced.
      for (auto maskExprs : implicitSpace->getMasks()) {
        const auto size = maskExprs.size() - 1;
        auto genFalseBlock = [&](const auto *e, auto &&cond) {
          auto ifOp = builder.create<fir::IfOp>(
              loc, mlir::TypeRange{innerArg.getType()}, fir::getBase(cond),
              /*withElseRegion=*/true);
          builder.create<fir::ResultOp>(loc, ifOp.getResult(0));
          builder.setInsertionPointToStart(&ifOp.thenRegion().front());
          builder.create<fir::ResultOp>(loc, innerArg);
          builder.setInsertionPointToStart(&ifOp.elseRegion().front());
        };
        auto genTrueBlock = [&](const auto *e, auto &&cond) {
          auto ifOp = builder.create<fir::IfOp>(
              loc, mlir::TypeRange{innerArg.getType()}, fir::getBase(cond),
              /*withElseRegion=*/true);
          builder.create<fir::ResultOp>(loc, ifOp.getResult(0));
          builder.setInsertionPointToStart(&ifOp.elseRegion().front());
          builder.create<fir::ResultOp>(loc, innerArg);
          builder.setInsertionPointToStart(&ifOp.thenRegion().front());
        };
        for (std::remove_const_t<decltype(size)> i = 0; i < size; ++i)
          if (const auto *e = maskExprs[i])
            genFalseBlock(e, genCond(e, iters));

        // The last condition is either non-negated or unconditionally negated.
        if (const auto *e = maskExprs[size])
          genTrueBlock(e, genCond(e, iters));
      }
    }

    // We're ready to lower the body (an assignment statement) for this context
    // of loop nests at this point.
    return {iters, afterLoopNest};
  }

  fir::ArrayLoadOp
  createAndLoadSomeArrayTemp(mlir::Type type,
                             llvm::ArrayRef<mlir::Value> shape) {
    if (ccLoadDest.hasValue())
      return ccLoadDest.getValue()(shape);
    auto seqTy = type.dyn_cast<fir::SequenceType>();
    assert(seqTy && "must be an array");
    auto loc = getLoc();
    // TODO: Need to thread the length parameters here. For character, they may
    // differ from the operands length (e.g concatenation). So the array loads
    // type parameters are not enough.
    if (auto charTy = seqTy.getEleTy().dyn_cast<fir::CharacterType>())
      if (charTy.hasDynamicLen())
        TODO(loc, "character array expression temp with dynamic length");
    if (auto recTy = seqTy.getEleTy().dyn_cast<fir::RecordType>())
      if (recTy.getNumLenParams() > 0)
        TODO(loc, "derived type array expression temp with length parameters");
    mlir::Value temp = seqTy.hasConstantShape()
                           ? builder.create<fir::AllocMemOp>(loc, type)
                           : builder.create<fir::AllocMemOp>(
                                 loc, type, ".array.expr", llvm::None, shape);
    auto *bldr = &converter.getFirOpBuilder();
    stmtCtx.attachCleanup(
        [bldr, loc, temp]() { bldr->create<fir::FreeMemOp>(loc, temp); });
    auto shapeOp = genShapeOp(shape);
    return builder.create<fir::ArrayLoadOp>(loc, seqTy, temp, shapeOp,
                                            /*slice=*/mlir::Value{},
                                            llvm::None);
  }

  static fir::ShapeOp genShapeOp(mlir::Location loc, fir::FirOpBuilder &builder,
                                 llvm::ArrayRef<mlir::Value> shape) {
    auto idxTy = builder.getIndexType();
    llvm::SmallVector<mlir::Value> idxShape;
    for (auto s : shape)
      idxShape.push_back(builder.createConvert(loc, idxTy, s));
    auto shapeTy = fir::ShapeType::get(builder.getContext(), idxShape.size());
    return builder.create<fir::ShapeOp>(loc, shapeTy, idxShape);
  }

  fir::ShapeOp genShapeOp(llvm::ArrayRef<mlir::Value> shape) {
    return genShapeOp(getLoc(), builder, shape);
  }

  //===--------------------------------------------------------------------===//
  // Expression traversal and lowering.
  //===--------------------------------------------------------------------===//

  // Lower the expression in a scalar context.
  template <typename A>
  ExtValue asScalar(const A &x) {
    return ScalarExprLowering{getLoc(), converter, symMap, stmtCtx}.genval(x);
  }
  template <typename A>
  ExtValue asScalarArray(const A &x) {
    assert(explicitSpace);
    return genarr(x)(IterationSpace{});
  }

  /// Lower the expression in a scalar context to a memory reference.
  template <typename A>
  ExtValue asScalarRef(const A &x) {
    return ScalarExprLowering{getLoc(), converter, symMap, stmtCtx}.gen(x);
  }
  template <typename A>
  ExtValue asScalarArrayRef(const A &x) {
    assert(explicitSpace);
    PushSemantics(ConstituentSemantics::RefOpaque);
    return genarr(x)(IterationSpace{});
  }
  template <typename A>
  ExtValue asScalarArrayRef(const A &x, ComponentPath &components) {
    assert(explicitSpace);
    PushSemantics(ConstituentSemantics::RefOpaque);
    return genarr(x, components)(IterationSpace{});
  }

  // An expression with non-zero rank is an array expression.
  template <typename A>
  bool isArray(const A &x) const {
    return x.Rank() != 0;
  }

  /// Some temporaries are allocated on an element-by-element basis during the
  /// array expression evaluation. Collect the cleanups here so the resources
  /// can be freed before the next loop iteration, avoiding memory leaks. etc.
  Fortran::lower::StatementContext &getElementCtx() {
    if (!elementCtx)
      elementCtx = new Fortran::lower::StatementContext;
    return *elementCtx;
  }

  /// If there were temporaries created for this element evaluation, finalize
  /// and deallocate the resources now. This should be done just prior the the
  /// fir::ResultOp at the end of the innermost loop.
  void finalizeElementCtx() {
    if (elementCtx) {
      elementCtx->finalize();
      elementCtx = nullptr;
    }
  }

  // A procedure reference to a Fortran elemental intrinsic procedure.
  CC genElementalIntrinsicProcRef(
      const Fortran::evaluate::ProcedureRef &procRef,
      llvm::Optional<mlir::Type> retTy,
      const Fortran::evaluate::SpecificIntrinsic &intrinsic) {
    llvm::SmallVector<CC> operands;
    llvm::StringRef name = intrinsic.name;
    const auto *argLowering =
        Fortran::lower::getIntrinsicArgumentLowering(name);
    auto loc = getLoc();
    for (const auto &[arg, dummy] :
         llvm::zip(procRef.arguments(),
                   intrinsic.characteristics.value().dummyArguments)) {
      auto *expr = Fortran::evaluate::UnwrapExpr<
          Fortran::evaluate::Expr<Fortran::evaluate::SomeType>>(arg);
      if (!expr) {
        // Absent optional.
        operands.emplace_back([=](IterSpace) { return mlir::Value{}; });
      } else if (!argLowering) {
        // No argument lowering instruction, lower by value.
        PushSemantics(ConstituentSemantics::RefTransparent);
        auto lambda = genarr(*expr);
        operands.emplace_back([=](IterSpace iters) { return lambda(iters); });
      } else {
        // Ad-hoc argument lowering handling.
        switch (Fortran::lower::lowerIntrinsicArgumentAs(getLoc(), *argLowering,
                                                         dummy.name)) {
        case Fortran::lower::LowerIntrinsicArgAs::Value: {
          PushSemantics(ConstituentSemantics::RefTransparent);
          auto lambda = genarr(*expr);
          operands.emplace_back([=](IterSpace iters) { return lambda(iters); });
        } break;
        case Fortran::lower::LowerIntrinsicArgAs::Addr: {
          // Note: assume does not have Fortran VALUE attribute semantics.
          PushSemantics(ConstituentSemantics::RefOpaque);
          auto lambda = genarr(*expr);
          operands.emplace_back([=](IterSpace iters) { return lambda(iters); });
        } break;
        case Fortran::lower::LowerIntrinsicArgAs::Box: {
          PushSemantics(ConstituentSemantics::RefOpaque);
          auto lambda = genarr(*expr);
          operands.emplace_back([=](IterSpace iters) {
            return builder.createBox(loc, lambda(iters));
          });
        } break;
        case Fortran::lower::LowerIntrinsicArgAs::Inquired:
          TODO(loc, "intrinsic function with inquired argument");
          break;
        }
      }
    }

    // Let the intrinsic library lower the intrinsic procedure call
    return [=](IterSpace iters) {
      llvm::SmallVector<ExtValue> args;
      for (const auto &cc : operands)
        args.push_back(cc(iters));
      return Fortran::lower::genIntrinsicCall(builder, loc, name, retTy, args,
                                              getElementCtx());
    };
  }

  // A procedure reference to a user-defined elemental procedure.
  CC genElementalUserDefinedProcRef(
      const Fortran::evaluate::ProcedureRef &procRef,
      llvm::Optional<mlir::Type> retTy) {
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;

    // 10.1.4 p5. Impure elemental procedures must be called in element order.
    if (const auto *procSym = procRef.proc().GetSymbol())
      if (!Fortran::semantics::IsPureProcedure(*procSym))
        setUnordered(false);

    Fortran::lower::CallerInterface caller(procRef, converter);
    llvm::SmallVector<CC> operands;
    operands.reserve(caller.getPassedArguments().size());
    auto loc = getLoc();
    auto callSiteType = caller.genFunctionType();
    for (const auto &arg : caller.getPassedArguments()) {
      // 15.8.3 p1. Elemental procedure with intent(out)/intent(inout)
      // arguments must be called in element order.
      if (arg.mayBeModifiedByCall())
        setUnordered(false);
      const auto *actual = arg.entity;
      auto argTy = callSiteType.getInput(arg.firArgument);
      if (!actual) {
        // Optional dummy argument for which there is no actual argument.
        auto absent = builder.create<fir::AbsentOp>(loc, argTy);
        operands.emplace_back([=](IterSpace) { return absent; });
        continue;
      }
      const auto *expr = actual->UnwrapExpr();
      if (!expr)
        TODO(loc, "assumed type actual argument lowering");

      LLVM_DEBUG(expr->AsFortran(llvm::dbgs()
                                 << "argument: " << arg.firArgument << " = [")
                 << "]\n");
      switch (arg.passBy) {
      case PassBy::Value: {
        // True pass-by-value semantics.
        PushSemantics(ConstituentSemantics::RefTransparent);
        operands.emplace_back(genarr(*expr));
      } break;
      case PassBy::BaseAddressValueAttribute: {
        // VALUE attribute or pass-by-reference to a copy semantics. (byval*)
        if (isArray(*expr)) {
          PushSemantics(ConstituentSemantics::ByValueArg);
          operands.emplace_back(genarr(*expr));
        } else {
          // Store scalar value in a temp to fulfill VALUE attribute.
          auto val = fir::getBase(asScalar(*expr));
          auto temp = builder.createTemporary(
              loc, val.getType(),
              llvm::ArrayRef<mlir::NamedAttribute>{
                  Fortran::lower::getAdaptToByRefAttr(builder)});
          builder.create<fir::StoreOp>(loc, val, temp);
          operands.emplace_back(
              [=](IterSpace iters) -> ExtValue { return temp; });
        }
      } break;
      case PassBy::BaseAddress: {
        if (isArray(*expr)) {
          PushSemantics(ConstituentSemantics::RefOpaque);
          operands.emplace_back(genarr(*expr));
        } else {
          auto exv = asScalarRef(*expr);
          operands.emplace_back([=](IterSpace iters) { return exv; });
        }
      } break;
      case PassBy::CharBoxValueAttribute: {
        if (isArray(*expr)) {
          PushSemantics(ConstituentSemantics::RefOpaque);
          auto lambda = genarr(*expr);
          operands.emplace_back([=](IterSpace iters) {
            return fir::factory::CharacterExprHelper{builder, loc}
                .createTempFrom(lambda(iters));
          });
        } else {
          fir::factory::CharacterExprHelper helper(builder, loc);
          auto argVal = helper.createTempFrom(asScalarRef(*expr));
          operands.emplace_back(
              [=](IterSpace iters) -> ExtValue { return argVal; });
        }
      } break;
      case PassBy::BoxChar: {
        PushSemantics(ConstituentSemantics::RefOpaque);
        operands.emplace_back(genarr(*expr));
      } break;
      case PassBy::AddressAndLength:
        // PassBy::AddressAndLength is only used for character results. Results
        // are not handled here.
        fir::emitFatalError(
            loc, "unexpected PassBy::AddressAndLength in elemental call");
        break;
      case PassBy::Box:
      case PassBy::MutableBox:
        // See C15100 and C15101
        fir::emitFatalError(loc, "cannot be POINTER, ALLOCATABLE");
      }
    }

    if (caller.getIfIndirectCallSymbol())
      fir::emitFatalError(loc, "cannot be indirect call");

    // The lambda is mutable so that `caller` copy can be modified inside it.
    return
        [=, caller = std::move(caller)](IterSpace iters) mutable -> ExtValue {
          for (const auto &[cc, argIface] :
               llvm::zip(operands, caller.getPassedArguments())) {
            auto exv = cc(iters);
            auto arg = exv.match(
                [&](const fir::CharBoxValue &cb) -> mlir::Value {
                  return fir::factory::CharacterExprHelper{builder, loc}
                      .createEmbox(cb);
                },
                [&](const auto &) { return fir::getBase(exv); });
            caller.placeInput(argIface, arg);
          }
          return ScalarExprLowering{loc, converter, symMap, stmtCtx}
              .genCallOpAndResult(caller, callSiteType, retTy);
        };
  }

  /// Generate a procedure reference. This code is shared for both functions and
  /// subroutines, the difference being reflected by `retTy`.
  CC genProcRef(const Fortran::evaluate::ProcedureRef &procRef,
                llvm::Optional<mlir::Type> retTy) {
    auto loc = getLoc();
    if (procRef.IsElemental()) {
      if (const auto *intrin = procRef.proc().GetSpecificIntrinsic()) {
        // All elemental intrinsic functions are pure and cannot modify their
        // arguments. The only elemental subroutine, MVBITS has an Intent(inout)
        // argument. So for this last one, loops must be in element order
        // according to 15.8.3 p1.
        if (!retTy)
          setUnordered(false);

        // Elemental intrinsic call.
        // The intrinsic procedure is called once per element of the array.
        return genElementalIntrinsicProcRef(procRef, retTy, *intrin);
      }
      if (ScalarExprLowering::isStatementFunctionCall(procRef))
        fir::emitFatalError(loc, "statement function cannot be elemental");

      // Elemental call.
      // The procedure is called once per element of the array argument(s).
      return genElementalUserDefinedProcRef(procRef, retTy);
    }

    // Transformational call.
    // The procedure is called once and produces a value of rank > 0.
    if (const auto *intrinsic = procRef.proc().GetSpecificIntrinsic()) {
      if (explicitSpaceIsActive() && procRef.Rank() == 0) {
        // Elide any implicit loop iters.
        return [=, &procRef](IterSpace) {
          return ScalarExprLowering{loc, converter, symMap, stmtCtx}
              .genIntrinsicRef(procRef, *intrinsic, retTy);
        };
      }
      return genarr(
          ScalarExprLowering{loc, converter, symMap, stmtCtx}.genIntrinsicRef(
              procRef, *intrinsic, retTy));
    }

    if (explicitSpaceIsActive() && procRef.Rank() == 0) {
      // Elide any implicit loop iters.
      return [=, &procRef](IterSpace) {
        return ScalarExprLowering{loc, converter, symMap, stmtCtx}
            .genProcedureRef(procRef, retTy);
      };
    }
    // In the default case, the call can be hoisted out of the loop nest. Apply
    // the iterations to the result, which may be an array value.
    return genarr(
        ScalarExprLowering{loc, converter, symMap, stmtCtx}.genProcedureRef(
            procRef, retTy));
  }

  CC genarr(const Fortran::evaluate::ProcedureDesignator &) {
    TODO(getLoc(), "procedure designator");
  }
  CC genarr(const Fortran::evaluate::ProcedureRef &x) {
    if (x.hasAlternateReturns())
      fir::emitFatalError(getLoc(),
                          "array procedure reference with alt-return");
    return genProcRef(x, llvm::None);
  }
  template <typename A>
  CC genScalarAndForwardValue(const A &x) {
    auto result = asScalar(x);
    return [=](IterSpace) { return result; };
  }
  template <typename A, typename = std::enable_if_t<Fortran::common::HasMember<
                            A, Fortran::evaluate::TypelessExpression>>>
  CC genarr(const A &x) {
    return genScalarAndForwardValue(x);
  }

  template <typename A>
  CC genarr(const Fortran::evaluate::Expr<A> &x) {
    LLVM_DEBUG(Fortran::lower::DumpEvaluateExpr::dump(llvm::dbgs(), x));
    if (isArray(x) || explicitSpaceIsActive() ||
        isElementalProcWithArrayArgs(x))
      return std::visit([&](const auto &e) { return genarr(e); }, x.u);
    return genScalarAndForwardValue(x);
  }

  // Converting a value of memory bound type requires creating a temp and
  // copying the value.
  static ExtValue convertAdjustedType(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Type toType,
                                      const ExtValue &exv) {
    return exv.match(
        [&](const fir::CharBoxValue &cb) -> ExtValue {
          auto len = cb.getLen();
          auto mem =
              builder.create<fir::AllocaOp>(loc, toType, mlir::ValueRange{len});
          fir::CharBoxValue result(mem, len);
          fir::factory::CharacterExprHelper{builder, loc}.createAssign(
              ExtValue{result}, exv);
          return result;
        },
        [&](const auto &) -> ExtValue {
          fir::emitFatalError(loc, "convert on adjusted extended value");
        });
  }
  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  CC genarr(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>,
                                             TC2> &x) {
    auto loc = getLoc();
    auto lambda = genarr(x.left());
    auto ty = converter.genType(TC1, KIND);
    return [=](IterSpace iters) -> ExtValue {
      auto exv = lambda(iters);
      auto val = fir::getBase(exv);
      auto valTy = val.getType();
      if (elementTypeWasAdjusted(valTy) &&
          !(fir::isa_ref_type(valTy) && fir::isa_integer(ty)))
        return convertAdjustedType(builder, loc, ty, exv);
      return builder.createConvert(loc, ty, val);
    };
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::ComplexComponent<KIND> &x) {
    auto loc = getLoc();
    auto lambda = genarr(x.left());
    auto isImagPart = x.isImaginaryPart;
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lambda(iters));
      return fir::factory::Complex{builder, loc}.extractComplexPart(lhs,
                                                                    isImagPart);
    };
  }

  template <typename T>
  CC genarr(const Fortran::evaluate::Parentheses<T> &x) {
    auto loc = getLoc();
    if (isReferentiallyOpaque()) {
      // Context is a call argument in, for example, an elemental procedure
      // call. TODO: all array arguments should use array_load, array_access,
      // array_amend, and INTENT(OUT), INTENT(INOUT) arguments should have
      // array_merge_store ops.
      TODO(loc, "parentheses on argument in elemental call");
    }
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      auto val = f(iters);
      auto base = fir::getBase(val);
      auto newBase =
          builder.create<fir::NoReassocOp>(loc, base.getType(), base);
      return fir::substBase(val, newBase);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Integer, KIND>> &x) {
    auto loc = getLoc();
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      auto val = fir::getBase(f(iters));
      auto ty = converter.genType(Fortran::common::TypeCategory::Integer, KIND);
      auto zero = builder.createIntegerConstant(loc, ty, 0);
      return builder.create<mlir::arith::SubIOp>(loc, zero, val);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Real, KIND>> &x) {
    auto loc = getLoc();
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      return builder.create<mlir::arith::NegFOp>(loc, fir::getBase(f(iters)));
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Complex, KIND>> &x) {
    auto loc = getLoc();
    auto f = genarr(x.left());
    return [=](IterSpace iters) -> ExtValue {
      return builder.create<fir::NegcOp>(loc, fir::getBase(f(iters)));
    };
  }

  //===--------------------------------------------------------------------===//
  // Binary elemental ops
  //===--------------------------------------------------------------------===//

  template <typename OP, typename A>
  CC createBinaryOp(const A &evEx) {
    auto loc = getLoc();
    auto lambda = genarr(evEx.left());
    auto rf = genarr(evEx.right());
    return [=](IterSpace iters) -> ExtValue {
      auto left = fir::getBase(lambda(iters));
      auto right = fir::getBase(rf(iters));
      return builder.create<OP>(loc, left, right);
    };
  }

#undef GENBIN
#define GENBIN(GenBinEvOp, GenBinTyCat, GenBinFirOp)                           \
  template <int KIND>                                                          \
  CC genarr(const Fortran::evaluate::GenBinEvOp<Fortran::evaluate::Type<       \
                Fortran::common::TypeCategory::GenBinTyCat, KIND>> &x) {       \
    return createBinaryOp<GenBinFirOp>(x);                                     \
  }

  GENBIN(Add, Integer, mlir::arith::AddIOp)
  GENBIN(Add, Real, mlir::arith::AddFOp)
  GENBIN(Add, Complex, fir::AddcOp)
  GENBIN(Subtract, Integer, mlir::arith::SubIOp)
  GENBIN(Subtract, Real, mlir::arith::SubFOp)
  GENBIN(Subtract, Complex, fir::SubcOp)
  GENBIN(Multiply, Integer, mlir::arith::MulIOp)
  GENBIN(Multiply, Real, mlir::arith::MulFOp)
  GENBIN(Multiply, Complex, fir::MulcOp)
  GENBIN(Divide, Integer, mlir::arith::DivSIOp)
  GENBIN(Divide, Real, mlir::arith::DivFOp)
  GENBIN(Divide, Complex, fir::DivcOp)

  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &x) {
    auto loc = getLoc();
    auto ty = converter.genType(TC, KIND);
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lf(iters));
      auto rhs = fir::getBase(rf(iters));
      return Fortran::lower::genPow(builder, loc, ty, lhs, rhs);
    };
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>> &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    switch (x.ordering) {
    case Fortran::evaluate::Ordering::Greater:
      return [=](IterSpace iters) -> ExtValue {
        auto lhs = fir::getBase(lf(iters));
        auto rhs = fir::getBase(rf(iters));
        return Fortran::lower::genMax(builder, loc,
                                      llvm::ArrayRef<mlir::Value>{lhs, rhs});
      };
    case Fortran::evaluate::Ordering::Less:
      return [=](IterSpace iters) -> ExtValue {
        auto lhs = fir::getBase(lf(iters));
        auto rhs = fir::getBase(rf(iters));
        return Fortran::lower::genMin(builder, loc,
                                      llvm::ArrayRef<mlir::Value>{lhs, rhs});
      };
    case Fortran::evaluate::Ordering::Equal:
      llvm_unreachable("Equal is not a valid ordering in this context");
    }
    llvm_unreachable("unknown ordering");
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  CC genarr(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &x) {
    auto loc = getLoc();
    auto ty = converter.genType(TC, KIND);
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) {
      auto lhs = fir::getBase(lf(iters));
      auto rhs = fir::getBase(rf(iters));
      return Fortran::lower::genPow(builder, loc, ty, lhs, rhs);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::ComplexConstructor<KIND> &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lf(iters));
      auto rhs = fir::getBase(rf(iters));
      return fir::factory::Complex{builder, loc}.createComplex(KIND, lhs, rhs);
    };
  }

  /// Fortran's concatenation operator `//`.
  template <int KIND>
  CC genarr(const Fortran::evaluate::Concat<KIND> &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = lf(iters);
      auto rhs = rf(iters);
      auto *lchr = lhs.getCharBox();
      auto *rchr = rhs.getCharBox();
      if (lchr && rchr) {
        return fir::factory::CharacterExprHelper{builder, loc}
            .createConcatenate(*lchr, *rchr);
      }
      TODO(loc, "concat on unexpected extended values");
      return mlir::Value{};
    };
  }

  template <int KIND>
  CC genarr(const Fortran::evaluate::SetLength<KIND> &x) {
    auto lf = genarr(x.left());
    auto rhs = fir::getBase(asScalar(x.right()));
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lf(iters));
      return fir::CharBoxValue{lhs, rhs};
    };
  }

  template <typename A>
  CC genarr(const Fortran::evaluate::Constant<A> &x) {
    if (/*explicitSpaceIsActive() &&*/ x.Rank() == 0)
      return genScalarAndForwardValue(x);
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto arrTy = converter.genType(toEvExpr(x));
    std::string globalName = Fortran::lower::mangle::mangleArrayLiteral(x);
    auto global = builder.getNamedGlobal(globalName);
    if (!global) {
      global = builder.createGlobalConstant(
          loc, arrTy, globalName,
          [&](fir::FirOpBuilder &builder) {
            Fortran::lower::StatementContext stmtCtx;
            auto result = Fortran::lower::createSomeInitializerExpression(
                loc, converter, toEvExpr(x), symMap, stmtCtx);
            auto castTo =
                builder.createConvert(loc, arrTy, fir::getBase(result));
            builder.create<fir::HasValueOp>(loc, castTo);
          },
          builder.createInternalLinkage());
    }
    auto addr = builder.create<fir::AddrOfOp>(getLoc(), global.resultType(),
                                              global.getSymbol());
    auto seqTy = global.getType().cast<fir::SequenceType>();
    llvm::SmallVector<mlir::Value> extents;
    for (auto extent : seqTy.getShape())
      extents.push_back(builder.createIntegerConstant(loc, idxTy, extent));
    if (auto charTy = seqTy.getEleTy().dyn_cast<fir::CharacterType>()) {
      auto len = builder.createIntegerConstant(loc, builder.getI64Type(),
                                               charTy.getLen());
      return genarr(fir::CharArrayBoxValue{addr, len, extents});
    }
    return genarr(fir::ArrayBoxValue{addr, extents});
  }

  //===--------------------------------------------------------------------===//
  // A vector subscript expression may be wrapped with a cast to INTEGER*8.
  // Get rid of it here so the vector can be loaded. Add it back when
  // generating the elemental evaluation (inside the loop nest).

  static Fortran::evaluate::Expr<Fortran::evaluate::SomeType>
  ignoreEvConvert(const Fortran::evaluate::Expr<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, 8>> &x) {
    return std::visit([&](const auto &v) { return ignoreEvConvert(v); }, x.u);
  }
  template <Fortran::common::TypeCategory FROM>
  static Fortran::evaluate::Expr<Fortran::evaluate::SomeType> ignoreEvConvert(
      const Fortran::evaluate::Convert<
          Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, 8>,
          FROM> &x) {
    return toEvExpr(x.left());
  }
  template <typename A>
  static Fortran::evaluate::Expr<Fortran::evaluate::SomeType>
  ignoreEvConvert(const A &x) {
    return toEvExpr(x);
  }

  //===--------------------------------------------------------------------===//
  // Get the `Se::Symbol*` for the subscript expression, `x`. This symbol can
  // be used to determine the lbound, ubound of the vector.

  template <typename A>
  static const Fortran::semantics::Symbol *
  extractSubscriptSymbol(const Fortran::evaluate::Expr<A> &x) {
    return std::visit([&](const auto &v) { return extractSubscriptSymbol(v); },
                      x.u);
  }
  template <typename A>
  static const Fortran::semantics::Symbol *
  extractSubscriptSymbol(const Fortran::evaluate::Designator<A> &x) {
    return Fortran::evaluate::UnwrapWholeSymbolDataRef(x);
  }
  template <typename A>
  static const Fortran::semantics::Symbol *extractSubscriptSymbol(const A &x) {
    return nullptr;
  }

  //===--------------------------------------------------------------------===//

  /// Get the declared lower bound value of the array `x` in dimension `dim`.
  /// The argument `one` must be an ssa-value for the constant 1.
  mlir::Value getLBound(const ExtValue &x, unsigned dim, mlir::Value one) {
    return fir::factory::readLowerBound(builder, getLoc(), x, dim, one);
  }

  /// Get the declared upper bound value of the array `x` in dimension `dim`.
  /// The argument `one` must be an ssa-value for the constant 1.
  mlir::Value getUBound(const ExtValue &x, unsigned dim, mlir::Value one) {
    auto loc = getLoc();
    auto lb = getLBound(x, dim, one);
    auto extent = fir::factory::readExtent(builder, loc, x, dim);
    auto add = builder.create<mlir::arith::AddIOp>(loc, lb, extent);
    return builder.create<mlir::arith::SubIOp>(loc, add, one);
  }

  /// Return the extent of the boxed array `x` in dimesion `dim`.
  mlir::Value getExtent(const ExtValue &x, unsigned dim) {
    return fir::factory::readExtent(builder, getLoc(), x, dim);
  }

  template <typename A>
  ExtValue genArrayBase(const A &base) {
    ScalarExprLowering sel{getLoc(), converter, symMap, stmtCtx};
    return base.IsSymbol() ? sel.gen(base.GetFirstSymbol())
                           : sel.gen(base.GetComponent());
  }

  template <typename A>
  bool hasEvArrayRef(const A &x) {
    struct HasEvArrayRefHelper
        : public Fortran::evaluate::AnyTraverse<HasEvArrayRefHelper> {
      HasEvArrayRefHelper()
          : Fortran::evaluate::AnyTraverse<HasEvArrayRefHelper>(*this) {}
      using Fortran::evaluate::AnyTraverse<HasEvArrayRefHelper>::operator();
      bool operator()(const Fortran::evaluate::ArrayRef &) const {
        return true;
      }
    } helper;
    return helper(x);
  }

  CC genVectorSubscriptArrayFetch(const Fortran::lower::SomeExpr &expr,
                                  std::size_t dim) {
    PushSemantics(ConstituentSemantics::RefTransparent);
    auto saved = Fortran::common::ScopedSet(explicitSpace, nullptr);
    auto savedDestShape = destShape;
    destShape.clear();
    auto result = genarr(expr);
    if (destShape.empty())
      TODO(getLoc(), "expected vector to have an extent");
    assert(destShape.size() == 1 && "vector has rank > 1");
    if (destShape[0] != savedDestShape[dim]) {
      // Not the same, so choose the smaller value.
      auto loc = getLoc();
      auto cmp = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sgt, destShape[0],
          savedDestShape[dim]);
      auto sel = builder.create<mlir::SelectOp>(loc, cmp, savedDestShape[dim],
                                                destShape[0]);
      savedDestShape[dim] = sel;
      destShape = savedDestShape;
    }
    return result;
  }

  /// Generate an access by vector subscript using the index in the iteration
  /// vector at `dim`.
  mlir::Value genAccessByVector(mlir::Location loc, CC genArrFetch,
                                IterSpace iters, std::size_t dim) {
    IterationSpace vecIters(iters,
                            llvm::ArrayRef<mlir::Value>{iters.iterValue(dim)});
    auto fetch = genArrFetch(vecIters);
    auto idxTy = builder.getIndexType();
    return builder.createConvert(loc, idxTy, fir::getBase(fetch));
  }

  /// When we have an array reference, the expressions specified in each
  /// dimension may be slice operations (e.g. `i:j:k`), vectors, or simple
  /// (loop-invarianet) scalar expressions. This returns the base entity, the
  /// resulting type, and a continuation to adjust the default iteration space.
  void genSliceIndices(ComponentPath &cmptData, const ExtValue &arrayExv,
                       const Fortran::evaluate::ArrayRef &x) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    auto &trips = cmptData.trips;
    LLVM_DEBUG(llvm::dbgs() << "array: " << arrayExv << '\n');
    auto &pc = cmptData.pc;
    const bool useTripsForSlice = !explicitSpaceIsActive();
    const bool createDestShape = destShape.empty();
    bool useSlice = false;
    std::size_t shapeIndex = 0;
    for (auto sub : llvm::enumerate(x.subscript())) {
      const auto subsIndex = sub.index();
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::evaluate::Triplet &t) {
                mlir::Value lowerBound;
                if (auto optLo = t.lower())
                  lowerBound = fir::getBase(asScalar(*optLo));
                else
                  lowerBound = getLBound(arrayExv, subsIndex, one);
                lowerBound = builder.createConvert(loc, idxTy, lowerBound);
                auto stride = fir::getBase(asScalar(t.stride()));
                stride = builder.createConvert(loc, idxTy, stride);
                if (useTripsForSlice || createDestShape) {
                  // Generate a slice operation for the triplet. The first and
                  // second position of the triplet may be omitted, and the
                  // declared lbound and/or ubound expression values,
                  // respectively, should be used instead.
                  trips.push_back(lowerBound);
                  mlir::Value upperBound;
                  if (auto optUp = t.upper())
                    upperBound = fir::getBase(asScalar(*optUp));
                  else
                    upperBound = getUBound(arrayExv, subsIndex, one);
                  upperBound = builder.createConvert(loc, idxTy, upperBound);
                  trips.push_back(upperBound);
                  trips.push_back(stride);
                  if (createDestShape) {
                    auto extent = builder.genExtentFromTriplet(
                        loc, lowerBound, upperBound, stride, idxTy);
                    destShape.push_back(extent);
                  }
                  useSlice = true;
                }
                if (!useTripsForSlice) {
                  auto currentPC = pc;
                  pc = [=](IterSpace iters) {
                    IterationSpace newIters = currentPC(iters);
                    auto impliedIter = newIters.iterValue(subsIndex);
                    // FIXME: must use the lower bound of this component.
                    // auto arrLowerBound = getLBound(arrayExv, subsIndex, one);
                    auto arrLowerBound = one;
                    auto initial = builder.create<mlir::arith::SubIOp>(
                        loc, lowerBound, arrLowerBound);
                    auto prod = builder.create<mlir::arith::MulIOp>(
                        loc, impliedIter, stride);
                    auto result =
                        builder.create<mlir::arith::AddIOp>(loc, initial, prod);
                    newIters.setIndexValue(subsIndex, result);
                    return newIters;
                  };
                }
                shapeIndex++;
              },
              [&](const Fortran::evaluate::IndirectSubscriptIntegerExpr &ie) {
                const auto &e = ie.value(); // dereference
                if (isArray(e)) {
                  // This is a vector subscript. Use the index values as read
                  // from a vector to determine the temporary array value.
                  // Note: 9.5.3.3.3(3) specifies undefined behavior for
                  // multiple updates to any specific array element through a
                  // vector subscript with replicated values.
                  assert(!isBoxValue() &&
                         "fir.box cannot be created with vector subscripts");
                  auto arrExpr = ignoreEvConvert(e);
                  if (createDestShape) {
                    destShape.push_back(fir::getExtentAtDimension(
                        arrayExv, builder, loc, subsIndex));
                  }
                  auto genArrFetch =
                      genVectorSubscriptArrayFetch(arrExpr, shapeIndex);
                  auto currentPC = pc;
                  pc = [=](IterSpace iters) {
                    IterationSpace newIters = currentPC(iters);
                    auto val = genAccessByVector(loc, genArrFetch, newIters,
                                                 subsIndex);
                    // Value read from vector subscript array and normalized
                    // using the base array's lower bound value.
                    mlir::Value lb = fir::factory::readLowerBound(
                        builder, loc, arrayExv, subsIndex, one);
                    auto origin = builder.create<mlir::arith::SubIOp>(
                        loc, idxTy, val, lb);
                    newIters.setIndexValue(subsIndex, origin);
                    return newIters;
                  };
                  if (useTripsForSlice) {
                    LLVM_ATTRIBUTE_UNUSED auto vectorSubscriptShape =
                        getShape(arrayOperands.back());
                    auto undef = builder.create<fir::UndefOp>(loc, idxTy);
                    trips.push_back(undef);
                    trips.push_back(undef);
                    trips.push_back(undef);
                  }
                  shapeIndex++;
                } else {
                  // This is a regular scalar subscript.
                  if (useTripsForSlice) {
                    // A regular scalar index, which does not yield an array
                    // section. Use a degenerate slice operation
                    // `(e:undef:undef)` in this dimension as a placeholder.
                    // This does not necessarily change the rank of the original
                    // array, so the iteration space must also be extended to
                    // include this expression in this dimension to adjust to
                    // the array's declared rank.
                    auto v = fir::getBase(asScalar(e));
                    trips.push_back(v);
                    auto undef = builder.create<fir::UndefOp>(loc, idxTy);
                    trips.push_back(undef);
                    trips.push_back(undef);
                    auto currentPC = pc;
                    // Cast `e` to index type.
                    auto iv = builder.createConvert(loc, idxTy, v);
                    // Normalize `e` by subtracting the declared lbound.
                    auto lb = fir::factory::readLowerBound(
                        builder, loc, arrayExv, subsIndex, one);
                    mlir::Value ivAdj =
                        builder.create<mlir::arith::SubIOp>(loc, idxTy, iv, lb);
                    // Add lbound adjusted value of `e` to the iteration vector
                    // (except when creating a box because the iteration vector
                    // is empty).
                    if (!isBoxValue())
                      pc = [=](IterSpace iters) {
                        IterationSpace newIters = currentPC(iters);
                        newIters.insertIndexValue(subsIndex, ivAdj);
                        return newIters;
                      };
                  } else {
                    auto currentPC = pc;
                    auto newValue = fir::getBase(asScalarArray(e));
                    mlir::Value result =
                        builder.createConvert(loc, idxTy, newValue);
                    mlir::Value lb = fir::factory::readLowerBound(
                        builder, loc, arrayExv, subsIndex, one);
                    result = builder.create<mlir::arith::SubIOp>(loc, idxTy,
                                                                 result, lb);
                    pc = [=](IterSpace iters) {
                      IterationSpace newIters = currentPC(iters);
                      newIters.insertIndexValue(subsIndex, result);
                      return newIters;
                    };
                  }
                }
              }},
          sub.value().u);
    }
    if (!useSlice)
      trips.clear();
  }

  static mlir::Type unwrapBoxEleTy(mlir::Type ty) {
    if (auto boxTy = ty.dyn_cast<fir::BoxType>())
      return fir::unwrapRefType(boxTy.getEleTy());
    return ty;
  }

  llvm::SmallVector<mlir::Value> getShape(mlir::Type ty) {
    llvm::SmallVector<mlir::Value> result;
    ty = unwrapBoxEleTy(ty);
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    for (auto extent : ty.cast<fir::SequenceType>().getShape()) {
      auto v = extent == fir::SequenceType::getUnknownExtent()
                   ? builder.create<fir::UndefOp>(loc, idxTy).getResult()
                   : builder.createIntegerConstant(loc, idxTy, extent);
      result.push_back(v);
    }
    return result;
  }
  llvm::SmallVector<mlir::Value>
  getShape(const Fortran::semantics::SymbolRef &x) {
    if (x.get().Rank() == 0)
      return {};
    return getFrontEndShape(x);
  }
  template <typename A>
  llvm::SmallVector<mlir::Value> getShape(const A &x) {
    if (x.Rank() == 0)
      return {};
    return getFrontEndShape(x);
  }
  template <typename A>
  llvm::SmallVector<mlir::Value> getFrontEndShape(const A &x) {
    if (auto optShape = Fortran::evaluate::GetShape(x)) {
      llvm::SmallVector<mlir::Value> result;
      convertFEShape(*optShape, result);
      return result;
    }
    return {};
  }

  CC genarr(const Fortran::semantics::SymbolRef &sym,
            ComponentPath &components) {
    return genarr(sym.get(), components);
  }

  ExtValue abstractArrayExtValue(mlir::Value val, mlir::Value len = {}) {
    return convertToArrayBoxValue(getLoc(), builder, val, len);
  }

  CC genarr(const ExtValue &extMemref) {
    ComponentPath dummy(/*isImplicit=*/true);
    return genarr(extMemref, dummy);
  }

  /// Base case of generating an array reference,
  CC genarr(const ExtValue &extMemref, ComponentPath &components) {
    auto loc = getLoc();
    auto memref = fir::getBase(extMemref);
    auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(memref.getType());
    assert(arrTy.isa<fir::SequenceType>() && "memory ref must be an array");
    auto shape = builder.createShape(loc, extMemref);
    if (destShape.empty()) {
      if (shape) {
        auto extents = fir::factory::getExtents(shape);
        destShape.assign(extents.begin(), extents.end());
      } else {
        auto extents = fir::factory::getExtents(builder, loc, extMemref);
        destShape.assign(extents.begin(), extents.end());
      }
    }
    mlir::Value slice;
    if (components.isSlice()) {
      if (isBoxValue() && components.substring) {
        // Append the substring operator to emboxing Op as it will become an
        // interior adjustment (add offset, adjust LEN) to the CHARACTER value
        // being referenced in the descriptor.
        llvm::SmallVector<mlir::Value> substringBounds;
        populateBounds(substringBounds, components.substring);
        // Convert to (offset, size)
        auto iTy = substringBounds[0].getType();
        if (substringBounds.size() != 2) {
          auto charTy = fir::factory::CharacterExprHelper::getCharType(arrTy);
          if (charTy.hasConstantLen()) {
            auto idxTy = builder.getIndexType();
            auto charLen = charTy.getLen();
            auto lenValue = builder.createIntegerConstant(loc, idxTy, charLen);
            substringBounds.push_back(lenValue);
          } else {
            auto typeparams = fir::getTypeParams(extMemref);
            substringBounds.push_back(typeparams.back());
          }
        }
        // Convert the lower bound to 0-based substring.
        auto one =
            builder.createIntegerConstant(loc, substringBounds[0].getType(), 1);
        substringBounds[0] =
            builder.create<mlir::arith::SubIOp>(loc, substringBounds[0], one);
        // Convert the upper bound to a length.
        auto cast = builder.createConvert(loc, iTy, substringBounds[1]);
        auto zero = builder.createIntegerConstant(loc, iTy, 0);
        auto size =
            builder.create<mlir::arith::SubIOp>(loc, cast, substringBounds[0]);
        auto cmp = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sgt, size, zero);
        // size = MAX(upper - (lower - 1), 0)
        substringBounds[1] =
            builder.create<mlir::SelectOp>(loc, cmp, size, zero);
        slice = builder.create<fir::SliceOp>(loc, components.trips,
                                             components.suffixComponents,
                                             substringBounds);
      } else {
        slice = builder.createSlice(loc, extMemref, components.trips,
                                    components.suffixComponents);
      }
      if (components.hasComponents()) {
        auto seqTy = arrTy.cast<fir::SequenceType>();
        auto eleTy =
            fir::applyPathToType(seqTy.getEleTy(), components.suffixComponents);
        if (!eleTy)
          fir::emitFatalError(loc, "slicing path is ill-formed");
        if (auto realTy = eleTy.dyn_cast<fir::RealType>())
          eleTy = Fortran::lower::convertReal(realTy.getContext(),
                                              realTy.getFKind());

        // create the type of the projected array.
        arrTy = fir::SequenceType::get(seqTy.getShape(), eleTy);
        LLVM_DEBUG(llvm::dbgs()
                   << "type of array projection from component slicing: "
                   << eleTy << ", " << arrTy << '\n');
      }
    }
    arrayOperands.push_back(ArrayOperand{memref, shape, slice});
    if (isBoxValue()) {
      // Semantics are a reference to a boxed array.
      // This case just requires that an embox operation be created to box the
      // value. The value of the box is forwarded in the continuation.
      auto reduceTy = reduceRank(arrTy, slice);
      auto boxTy = fir::BoxType::get(reduceTy);
      if (components.substring) {
        // Adjust char length to substring size.
        auto charTy = fir::factory::CharacterExprHelper::getCharType(reduceTy);
        auto seqTy = reduceTy.cast<fir::SequenceType>();
        // TODO: Use a constant for fir.char LEN if we can compute it.
        boxTy = fir::BoxType::get(
            fir::SequenceType::get(fir::CharacterType::getUnknownLen(
                                       builder.getContext(), charTy.getFKind()),
                                   seqTy.getDimension()));
      }
      mlir::Value embox =
          memref.getType().isa<fir::BoxType>()
              ? builder.create<fir::ReboxOp>(loc, boxTy, memref, shape, slice)
                    .getResult()
              : builder
                    .create<fir::EmboxOp>(loc, boxTy, memref, shape, slice,
                                          fir::getTypeParams(extMemref))
                    .getResult();
      return [=](IterSpace) -> ExtValue { return fir::BoxValue(embox); };
    }
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    if (isReferentiallyOpaque()) {
      // Semantics are an opaque reference to an array.
      // This case forwards a continuation that will generate the address
      // arithmetic to the array element. This does not have copy-in/copy-out
      // semantics. No attempt to copy the array value will be made during the
      // interpretation of the Fortran statement.
      auto refEleTy = builder.getRefType(eleTy);
      return [=](IterSpace iters) -> ExtValue {
        // ArrayCoorOp does not expect zero based indices.
        auto indices = fir::factory::originateIndices(
            loc, builder, memref.getType(), shape, iters.iterVec());
        mlir::Value coor = builder.create<fir::ArrayCoorOp>(
            loc, refEleTy, memref, shape, slice, indices,
            fir::getTypeParams(extMemref));
        if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
          llvm::SmallVector<mlir::Value> substringBounds;
          populateBounds(substringBounds, components.substring);
          if (!substringBounds.empty()) {
            mlir::Value dstLen = fir::factory::genLenOfCharacter(
                builder, loc, arrTy.cast<fir::SequenceType>(), memref,
                fir::getTypeParams(extMemref), iters.iterVec(),
                substringBounds);
            fir::CharBoxValue dstChar(coor, dstLen);
            return fir::factory::CharacterExprHelper{builder, loc}
                .createSubstring(dstChar, substringBounds);
          }
        }
        return fir::factory::arraySectionElementToExtendedValue(
            builder, loc, extMemref, coor, slice);
      };
    }
    auto arrLoad = builder.create<fir::ArrayLoadOp>(
        loc, arrTy, memref, shape, slice, fir::getTypeParams(extMemref));
    auto arrLd = arrLoad.getResult();
    if (isProjectedCopyInCopyOut()) {
      // Semantics are projected copy-in copy-out.
      // The backing store of the destination of an array expression may be
      // partially modified. These updates are recorded in FIR by forwarding a
      // continuation that generates an `array_update` Op. The destination is
      // always loaded at the beginning of the statement and merged at the
      // end.
      destination = arrLoad;
      auto lambda = ccStoreToDest.hasValue()
                        ? ccStoreToDest.getValue()
                        : defaultStoreToDestination(components.substring);
      return [=](IterSpace iters) -> ExtValue { return lambda(iters); };
    }
    if (isCustomCopyInCopyOut()) {
      // Create an array_modify to get the LHS element address and indicate
      // the assignment, the actual assignment must be implemented in
      // ccStoreToDest.
      destination = arrLoad;
      return [=](IterSpace iters) -> ExtValue {
        auto innerArg = iters.innerArgument();
        auto resTy = innerArg.getType();
        auto eleTy = fir::applyPathToType(resTy, iters.iterVec());
        auto refEleTy =
            fir::isa_ref_type(eleTy) ? eleTy : builder.getRefType(eleTy);
        auto arrModify = builder.create<fir::ArrayModifyOp>(
            loc, mlir::TypeRange{refEleTy, resTy}, innerArg, iters.iterVec(),
            destination.typeparams());
        return abstractArrayExtValue(arrModify.getResult(1));
      };
    }
    if (isCopyInCopyOut()) {
      // Semantics are copy-in copy-out.
      // The continuation simply forwards the result of the `array_load` Op,
      // which is the value of the array as it was when loaded. All data
      // references with rank > 0 in an array expression typically have
      // copy-in copy-out semantics.
      return [=](IterSpace) -> ExtValue { return arrLd; };
    }
    auto arrLdTypeParams = arrLoad.typeparams();
    if (isValueAttribute()) {
      // Semantics are value attribute.
      // Here the continuation will `array_fetch` a value from an array and
      // then store that value in a temporary. One can thus imitate pass by
      // value even when the call is pass by reference.
      return [=](IterSpace iters) -> ExtValue {
        mlir::Value base;
        auto eleTy = fir::applyPathToType(arrTy, iters.iterVec());
        if (isAdjustedArrayElementType(eleTy)) {
          auto eleRefTy = builder.getRefType(eleTy);
          base = builder.create<fir::ArrayAccessOp>(
              loc, eleRefTy, arrLd, iters.iterVec(), arrLdTypeParams);
        } else {
          base = builder.create<fir::ArrayFetchOp>(
              loc, eleTy, arrLd, iters.iterVec(), arrLdTypeParams);
        }
        auto temp = builder.createTemporary(
            loc, base.getType(),
            llvm::ArrayRef<mlir::NamedAttribute>{
                Fortran::lower::getAdaptToByRefAttr(builder)});
        builder.create<fir::StoreOp>(loc, base, temp);
        return fir::factory::arraySectionElementToExtendedValue(
            builder, loc, extMemref, temp, slice);
      };
    }
    // In the default case, the array reference forwards an `array_fetch` or
    // `array_access` Op in the continuation.
    return [=](IterSpace iters) -> ExtValue {
      auto eleTy = fir::applyPathToType(arrTy, iters.iterVec());
      if (isAdjustedArrayElementType(eleTy)) {
        auto eleRefTy = builder.getRefType(eleTy);
        mlir::Value arrayOp = builder.create<fir::ArrayAccessOp>(
            loc, eleRefTy, arrLd, iters.iterVec(), arrLdTypeParams);
        if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
          llvm::SmallVector<mlir::Value> substringBounds;
          populateBounds(substringBounds, components.substring);
          if (!substringBounds.empty()) {
            mlir::Value dstLen = fir::factory::genLenOfCharacter(
                builder, loc, arrLoad, iters.iterVec(), substringBounds);
            fir::CharBoxValue dstChar(arrayOp, dstLen);
            return fir::factory::CharacterExprHelper{builder, loc}
                .createSubstring(dstChar, substringBounds);
          }
        }
        return fir::factory::arraySectionElementToExtendedValue(
            builder, loc, extMemref, arrayOp, slice);
      }
      auto arrFetch = builder.create<fir::ArrayFetchOp>(
          loc, eleTy, arrLd, iters.iterVec(), arrLdTypeParams);
      return fir::factory::arraySectionElementToExtendedValue(
          builder, loc, extMemref, arrFetch, slice);
    };
  }

  /// Reduce the rank of a array to be boxed based on the slice's operands.
  static mlir::Type reduceRank(mlir::Type arrTy, mlir::Value slice) {
    if (slice) {
      auto slOp = mlir::dyn_cast<fir::SliceOp>(slice.getDefiningOp());
      assert(slOp && "expected slice op");
      auto seqTy = arrTy.dyn_cast<fir::SequenceType>();
      assert(seqTy && "expected array type");
      auto triples = slOp.triples();
      fir::SequenceType::Shape shape;
      // reduce the rank for each invariant dimension
      for (unsigned i = 1, end = triples.size(); i < end; i += 3)
        if (!mlir::isa_and_nonnull<fir::UndefOp>(triples[i].getDefiningOp()))
          shape.push_back(fir::SequenceType::getUnknownExtent());
      return fir::SequenceType::get(shape, seqTy.getEleTy());
    }
    // not sliced, so no change in rank
    return arrTy;
  }

  /// Example: <code>array%RE</code>
  CC genarr(const Fortran::evaluate::ComplexPart &x,
            ComponentPath &components) {
    components.reversePath.push_back(&x);
    return genarr(x.complex(), components);
  }

  template <typename A>
  CC genSlicePath(const A &x, ComponentPath &components) {
    return genarr(x, components);
  }

  CC genarr(const Fortran::evaluate::StaticDataObject::Pointer &,
            ComponentPath &components) {
    fir::emitFatalError(getLoc(), "substring of static array object");
  }

  /// Substrings (see 9.4.1)
  CC genarr(const Fortran::evaluate::Substring &x, ComponentPath &components) {
    components.substring = &x;
    return std::visit([&](const auto &v) { return genarr(v, components); },
                      x.parent());
  }

  template <typename T>
  CC genarr(const Fortran::evaluate::FunctionRef<T> &funRef) {
    // Note that it's possible that the function being called returns either an
    // array or a scalar.  In the first case, use the element type of the array.
    return genProcRef(
        funRef, fir::unwrapSequenceType(converter.genType(toEvExpr(funRef))));
  }

  //===--------------------------------------------------------------------===//
  // Array construction
  //===--------------------------------------------------------------------===//

  // Lower the expr cases in an ac-value-list.
  template <typename A>
  std::pair<ExtValue, bool>
  genArrayCtorInitializer(const Fortran::evaluate::Expr<A> &x, mlir::Type,
                          mlir::Value, mlir::Value, mlir::Value,
                          Fortran::lower::StatementContext &stmtCtx) {
    if (isArray(x))
      return {lowerNewArrayExpression(converter, symMap, stmtCtx, toEvExpr(x)),
              /*needCopy=*/true};
    return {asScalar(x), /*needCopy=*/true};
  }

  /// Target agnostic computation of the size of an element in the array.
  /// Returns the size in bytes with type `index` or a null Value if the element
  /// size is not constant.
  mlir::Value computeElementSize(const ExtValue &exv, mlir::Type eleTy,
                                 mlir::Type resTy) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto multiplier = builder.createIntegerConstant(loc, idxTy, 1);
    if (fir::hasDynamicSize(eleTy)) {
      if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
        // Array of char with dynamic length parameter. Downcast to an array
        // of singleton char, and scale by the len type parameter from
        // `exv`.
        exv.match(
            [&](const fir::CharBoxValue &cb) { multiplier = cb.getLen(); },
            [&](const fir::CharArrayBoxValue &cb) { multiplier = cb.getLen(); },
            [&](const fir::BoxValue &box) {
              multiplier = fir::factory::CharacterExprHelper(builder, loc)
                               .readLengthFromBox(box.getAddr());
            },
            [&](const fir::MutableBoxValue &box) {
              multiplier = fir::factory::CharacterExprHelper(builder, loc)
                               .readLengthFromBox(box.getAddr());
            },
            [&](const auto &) {
              fir::emitFatalError(loc,
                                  "array constructor element has unknown size");
            });
        auto newEleTy = fir::CharacterType::getSingleton(eleTy.getContext(),
                                                         charTy.getFKind());
        if (auto seqTy = resTy.dyn_cast<fir::SequenceType>()) {
          assert(eleTy == seqTy.getEleTy());
          resTy = fir::SequenceType::get(seqTy.getShape(), newEleTy);
        }
        eleTy = newEleTy;
      } else {
        TODO(loc, "dynamic sized type");
      }
    }
    auto eleRefTy = builder.getRefType(eleTy);
    auto resRefTy = builder.getRefType(resTy);
    auto nullPtr = builder.createNullConstant(loc, resRefTy);
    auto offset = builder.create<fir::CoordinateOp>(
        loc, eleRefTy, nullPtr, mlir::ValueRange{multiplier});
    return builder.createConvert(loc, idxTy, offset);
  }

  /// Get the function signature of the LLVM memcpy intrinsic.
  mlir::FunctionType memcpyType() {
    return fir::factory::getLlvmMemcpy(builder).getType();
  }

  /// Create a call to the LLVM memcpy intrinsic.
  void createCallMemcpy(llvm::ArrayRef<mlir::Value> args) {
    auto loc = getLoc();
    auto memcpyFunc = fir::factory::getLlvmMemcpy(builder);
    auto funcSymAttr = builder.getSymbolRefAttr(memcpyFunc.getName());
    auto funcTy = memcpyFunc.getType();
    builder.create<fir::CallOp>(loc, funcTy.getResults(), funcSymAttr, args);
  }

  // Construct code to check for a buffer overrun and realloc the buffer when
  // space is depleted. This is done between each item in the ac-value-list.
  mlir::Value growBuffer(mlir::Value mem, mlir::Value needed,
                         mlir::Value bufferSize, mlir::Value buffSize,
                         mlir::Value eleSz) {
    auto loc = getLoc();
    auto reallocFunc = fir::factory::getRealloc(builder);
    auto cond = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::sle, bufferSize, needed);
    auto ifOp = builder.create<fir::IfOp>(loc, mem.getType(), cond,
                                          /*withElseRegion=*/true);
    auto insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&ifOp.thenRegion().front());
    // Not enough space, resize the buffer.
    auto idxTy = builder.getIndexType();
    auto two = builder.createIntegerConstant(loc, idxTy, 2);
    auto newSz = builder.create<mlir::arith::MulIOp>(loc, needed, two);
    builder.create<fir::StoreOp>(loc, newSz, buffSize);
    mlir::Value byteSz = builder.create<mlir::arith::MulIOp>(loc, newSz, eleSz);
    auto funcSymAttr = builder.getSymbolRefAttr(reallocFunc.getName());
    auto funcTy = reallocFunc.getType();
    auto newMem = builder.create<fir::CallOp>(
        loc, funcTy.getResults(), funcSymAttr,
        llvm::ArrayRef<mlir::Value>{
            builder.createConvert(loc, funcTy.getInputs()[0], mem),
            builder.createConvert(loc, funcTy.getInputs()[1], byteSz)});
    auto castNewMem =
        builder.createConvert(loc, mem.getType(), newMem.getResult(0));
    builder.create<fir::ResultOp>(loc, castNewMem);
    builder.setInsertionPointToStart(&ifOp.elseRegion().front());
    // Otherwise, just forward the buffer.
    builder.create<fir::ResultOp>(loc, mem);
    builder.restoreInsertionPoint(insPt);
    return ifOp.getResult(0);
  }

  /// Copy the next value (or vector of values) into the array being
  /// constructed.
  mlir::Value copyNextArrayCtorSection(const ExtValue &exv, mlir::Value buffPos,
                                       mlir::Value buffSize, mlir::Value mem,
                                       mlir::Value eleSz, mlir::Type eleTy,
                                       mlir::Type eleRefTy, mlir::Type resTy) {
    auto loc = getLoc();
    auto off = builder.create<fir::LoadOp>(loc, buffPos);
    auto limit = builder.create<fir::LoadOp>(loc, buffSize);
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);

    if (fir::isRecordWithAllocatableMember(eleTy))
      TODO(loc, "deep copy on allocatable members");

    if (!eleSz) {
      // Compute the element size at runtime.
      assert(fir::hasDynamicSize(eleTy));
      if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
        auto charBytes =
            builder.getKindMap().getCharacterBitsize(charTy.getFKind()) / 8;
        auto bytes = builder.createIntegerConstant(loc, idxTy, charBytes);
        auto length = fir::getLen(exv);
        if (!length)
          fir::emitFatalError(loc, "result is not boxed character");
        eleSz = builder.create<mlir::arith::MulIOp>(loc, bytes, length);
      } else {
        TODO(loc, "PDT size");
        // Will call the PDT's size function with the type parameters.
      }
    }

    // Compute the coordinate using `fir.coordinate_of`, or, if the type has
    // dynamic size, generating the pointer arithmetic.
    auto computeCoordinate = [&](mlir::Value buff, mlir::Value off) {
      auto refTy = eleRefTy;
      if (fir::hasDynamicSize(eleTy)) {
        if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
          // Scale a simple pointer using dynamic length and offset values.
          auto chTy = fir::CharacterType::getSingleton(charTy.getContext(),
                                                       charTy.getFKind());
          refTy = builder.getRefType(chTy);
          auto toTy = builder.getRefType(builder.getVarLenSeqTy(chTy));
          buff = builder.createConvert(loc, toTy, buff);
          off = builder.create<mlir::arith::MulIOp>(loc, off, eleSz);
        } else {
          TODO(loc, "PDT offset");
        }
      }
      auto coor = builder.create<fir::CoordinateOp>(loc, refTy, buff,
                                                    mlir::ValueRange{off});
      return builder.createConvert(loc, eleRefTy, coor);
    };

    // Lambda to lower an abstract array box value.
    auto doAbstractArray = [&](const auto &v) {
      // Compute the array size.
      auto arrSz = one;
      for (auto ext : v.getExtents())
        arrSz = builder.create<mlir::arith::MulIOp>(loc, arrSz, ext);

      // Grow the buffer as needed.
      auto endOff = builder.create<mlir::arith::AddIOp>(loc, off, arrSz);
      mem = growBuffer(mem, endOff, limit, buffSize, eleSz);

      // Copy the elements to the buffer.
      mlir::Value byteSz =
          builder.create<mlir::arith::MulIOp>(loc, arrSz, eleSz);
      auto buff = builder.createConvert(loc, fir::HeapType::get(resTy), mem);
      auto buffi = computeCoordinate(buff, off);
      auto args = fir::runtime::createArguments(
          builder, loc, memcpyType(), buffi, v.getAddr(), byteSz,
          /*volatile=*/builder.createBool(loc, false));
      createCallMemcpy(args);

      // Save the incremented buffer position.
      builder.create<fir::StoreOp>(loc, endOff, buffPos);
    };

    // Copy the value.
    exv.match(
        [&](const mlir::Value &v) {
          // Increment the buffer position.
          auto plusOne = builder.create<mlir::arith::AddIOp>(loc, off, one);

          // Grow the buffer as needed.
          mem = growBuffer(mem, plusOne, limit, buffSize, eleSz);

          // Store the element in the buffer.
          auto buff =
              builder.createConvert(loc, fir::HeapType::get(resTy), mem);
          auto buffi = builder.create<fir::CoordinateOp>(loc, eleRefTy, buff,
                                                         mlir::ValueRange{off});
          fir::factory::genScalarAssignment(builder, loc, buffi, v);
          builder.create<fir::StoreOp>(loc, plusOne, buffPos);
        },
        [&](const fir::CharBoxValue &v) {
          // Increment the buffer position.
          auto plusOne = builder.create<mlir::arith::AddIOp>(loc, off, one);

          // Grow the buffer as needed.
          mem = growBuffer(mem, plusOne, limit, buffSize, eleSz);

          // Store the element in the buffer.
          auto buff =
              builder.createConvert(loc, fir::HeapType::get(resTy), mem);
          auto buffi = computeCoordinate(buff, off);
          auto args = fir::runtime::createArguments(
              builder, loc, memcpyType(), buffi, v.getAddr(), eleSz,
              /*volatile=*/builder.createBool(loc, false));
          createCallMemcpy(args);

          builder.create<fir::StoreOp>(loc, plusOne, buffPos);
        },
        [&](const fir::ArrayBoxValue &v) { doAbstractArray(v); },
        [&](const fir::CharArrayBoxValue &v) { doAbstractArray(v); },
        [&](const auto &) {
          TODO(loc, "unhandled array constructor expression");
        });
    return mem;
  }

  // Lower an ac-implied-do in an ac-value-list.
  template <typename A>
  std::pair<ExtValue, bool>
  genArrayCtorInitializer(const Fortran::evaluate::ImpliedDo<A> &x,
                          mlir::Type resTy, mlir::Value mem,
                          mlir::Value buffPos, mlir::Value buffSize,
                          Fortran::lower::StatementContext &) {
    auto loc = getLoc();
    auto idxTy = builder.getIndexType();
    auto lo =
        builder.createConvert(loc, idxTy, fir::getBase(asScalar(x.lower())));
    auto up =
        builder.createConvert(loc, idxTy, fir::getBase(asScalar(x.upper())));
    auto step =
        builder.createConvert(loc, idxTy, fir::getBase(asScalar(x.stride())));
    auto seqTy = resTy.template cast<fir::SequenceType>();
    auto eleTy = fir::unwrapSequenceType(seqTy);
    auto loop =
        builder.create<fir::DoLoopOp>(loc, lo, up, step, /*unordered=*/false,
                                      /*finalCount=*/false, mem);
    // create a new binding for x.name(), to ac-do-variable, to the iteration
    // value.
    symMap.pushImpliedDoBinding(toStringRef(x.name()), loop.getInductionVar());
    auto insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(loop.getBody());
    // Thread mem inside the loop via loop argument.
    mem = loop.getRegionIterArgs()[0];

    auto eleRefTy = builder.getRefType(eleTy);

    // Cleanups for temps in loop body. Any temps created in the loop body
    // need to be freed before the end of the loop.
    Fortran::lower::StatementContext loopCtx;
    llvm::Optional<mlir::Value> charLen;
    for (const Fortran::evaluate::ArrayConstructorValue<A> &acv : x.values()) {
      auto [exv, copyNeeded] = std::visit(
          [&](const auto &v) {
            return genArrayCtorInitializer(v, resTy, mem, buffPos, buffSize,
                                           loopCtx);
          },
          acv.u);
      auto eleSz = computeElementSize(exv, eleTy, resTy);
      mem = copyNeeded ? copyNextArrayCtorSection(exv, buffPos, buffSize, mem,
                                                  eleSz, eleTy, eleRefTy, resTy)
                       : fir::getBase(exv);
      if (fir::isa_char(seqTy.getEleTy()) && !charLen.hasValue()) {
        charLen = builder.createTemporary(loc, builder.getI64Type());
        auto castLen =
            builder.createConvert(loc, builder.getI64Type(), fir::getLen(exv));
        builder.create<fir::StoreOp>(loc, castLen, charLen.getValue());
      }
    }
    loopCtx.finalize();

    builder.create<fir::ResultOp>(loc, mem);
    builder.restoreInsertionPoint(insPt);
    mem = loop.getResult(0);
    symMap.popImpliedDoBinding();
    llvm::SmallVector<mlir::Value> extents = {
        builder.create<fir::LoadOp>(loc, buffPos).getResult()};

    // Convert to extended value.
    if (fir::isa_char(seqTy.getEleTy())) {
      auto len = builder.create<fir::LoadOp>(loc, charLen.getValue());
      return {fir::CharArrayBoxValue{mem, len, extents}, /*needCopy=*/false};
    }
    return {fir::ArrayBoxValue{mem, extents}, /*needCopy=*/false};
  }

  // To simplify the handling and interaction between the various cases, array
  // constructors are always lowered to the incremental construction code
  // pattern, even if the extent of the array value is constant. After the
  // MemToReg pass and constant folding, the optimizer should be able to
  // determine that all the buffer overrun tests are false when the
  // incremental construction wasn't actually required.
  template <typename A>
  CC genarr(const Fortran::evaluate::ArrayConstructor<A> &x) {
    auto loc = getLoc();
    auto evExpr = toEvExpr(x);
    auto resTy = translateSomeExprToFIRType(converter, evExpr);
    auto idxTy = builder.getIndexType();
    auto seqTy = resTy.template cast<fir::SequenceType>();
    auto eleTy = fir::unwrapSequenceType(resTy);
    auto buffSize = builder.createTemporary(loc, idxTy, ".buff.size");
    auto zero = builder.createIntegerConstant(loc, idxTy, 0);
    auto buffPos = builder.createTemporary(loc, idxTy, ".buff.pos");
    builder.create<fir::StoreOp>(loc, zero, buffPos);
    // Allocate space for the array to be constructed.
    mlir::Value mem;
    if (fir::hasDynamicSize(resTy)) {
      if (fir::hasDynamicSize(eleTy)) {
        // The size of each element may depend on a general expression. Defer
        // creating the buffer until after the expression is evaluated.
        mem = builder.createNullConstant(loc, builder.getRefType(eleTy));
        builder.create<fir::StoreOp>(loc, zero, buffSize);
      } else {
        auto initBuffSz =
            builder.createIntegerConstant(loc, idxTy, clInitialBufferSize);
        mem = builder.create<fir::AllocMemOp>(
            loc, eleTy, /*typeparams=*/llvm::None, initBuffSz);
        builder.create<fir::StoreOp>(loc, initBuffSz, buffSize);
      }
    } else {
      mem = builder.create<fir::AllocMemOp>(loc, resTy);
      int64_t buffSz = 1;
      for (auto extent : seqTy.getShape())
        buffSz *= extent;
      auto initBuffSz = builder.createIntegerConstant(loc, idxTy, buffSz);
      builder.create<fir::StoreOp>(loc, initBuffSz, buffSize);
    }
    // Compute size of element
    auto eleRefTy = builder.getRefType(eleTy);

    // Populate the buffer with the elements, growing as necessary.
    llvm::Optional<mlir::Value> charLen;
    for (const auto &expr : x) {
      auto [exv, copyNeeded] = std::visit(
          [&](const auto &e) {
            return genArrayCtorInitializer(e, resTy, mem, buffPos, buffSize,
                                           stmtCtx);
          },
          expr.u);
      auto eleSz = computeElementSize(exv, eleTy, resTy);
      mem = copyNeeded ? copyNextArrayCtorSection(exv, buffPos, buffSize, mem,
                                                  eleSz, eleTy, eleRefTy, resTy)
                       : fir::getBase(exv);
      if (fir::isa_char(seqTy.getEleTy()) && !charLen.hasValue()) {
        charLen = builder.createTemporary(loc, builder.getI64Type());
        auto castLen =
            builder.createConvert(loc, builder.getI64Type(), fir::getLen(exv));
        builder.create<fir::StoreOp>(loc, castLen, charLen.getValue());
      }
    }
    mem = builder.createConvert(loc, fir::HeapType::get(resTy), mem);
    llvm::SmallVector<mlir::Value> extents = {
        builder.create<fir::LoadOp>(loc, buffPos)};

    // Cleanup the temporary.
    auto *bldr = &converter.getFirOpBuilder();
    stmtCtx.attachCleanup(
        [bldr, loc, mem]() { bldr->create<fir::FreeMemOp>(loc, mem); });

    // Return the continuation.
    if (fir::isa_char(seqTy.getEleTy())) {
      auto len = builder.create<fir::LoadOp>(loc, charLen.getValue());
      return genarr(fir::CharArrayBoxValue{mem, len, extents});
    }
    return genarr(fir::ArrayBoxValue{mem, extents});
  }

  CC genarr(const Fortran::evaluate::ImpliedDoIndex &) {
    fir::emitFatalError(getLoc(), "implied do index cannot have rank > 0");
  }
  CC genarr(const Fortran::evaluate::TypeParamInquiry &x) {
    TODO(getLoc(), "array expr type parameter inquiry");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }
  CC genarr(const Fortran::evaluate::DescriptorInquiry &x) {
    TODO(getLoc(), "array expr descriptor inquiry");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }
  CC genarr(const Fortran::evaluate::StructureConstructor &x) {
    TODO(getLoc(), "structure constructor");
    return [](IterSpace iters) -> ExtValue { return mlir::Value{}; };
  }

  //===--------------------------------------------------------------------===//
  // LOCICAL operators (.NOT., .AND., .EQV., etc.)
  //===--------------------------------------------------------------------===//

  template <int KIND>
  CC genarr(const Fortran::evaluate::Not<KIND> &x) {
    auto loc = getLoc();
    auto i1Ty = builder.getI1Type();
    auto lambda = genarr(x.left());
    auto truth = builder.createBool(loc, true);
    return [=](IterSpace iters) -> ExtValue {
      auto logical = fir::getBase(lambda(iters));
      auto val = builder.createConvert(loc, i1Ty, logical);
      return builder.create<mlir::arith::XOrIOp>(loc, val, truth);
    };
  }
  template <typename OP, typename A>
  CC createBinaryBoolOp(const A &x) {
    auto loc = getLoc();
    auto i1Ty = builder.getI1Type();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto left = fir::getBase(lf(iters));
      auto right = fir::getBase(rf(iters));
      auto lhs = builder.createConvert(loc, i1Ty, left);
      auto rhs = builder.createConvert(loc, i1Ty, right);
      return builder.create<OP>(loc, lhs, rhs);
    };
  }
  template <typename OP, typename A>
  CC createCompareBoolOp(mlir::arith::CmpIPredicate pred, const A &x) {
    auto loc = getLoc();
    auto i1Ty = builder.getI1Type();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto left = fir::getBase(lf(iters));
      auto right = fir::getBase(rf(iters));
      auto lhs = builder.createConvert(loc, i1Ty, left);
      auto rhs = builder.createConvert(loc, i1Ty, right);
      return builder.create<OP>(loc, pred, lhs, rhs);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::LogicalOperation<KIND> &x) {
    switch (x.logicalOperator) {
    case Fortran::evaluate::LogicalOperator::And:
      return createBinaryBoolOp<mlir::arith::AndIOp>(x);
    case Fortran::evaluate::LogicalOperator::Or:
      return createBinaryBoolOp<mlir::arith::OrIOp>(x);
    case Fortran::evaluate::LogicalOperator::Eqv:
      return createCompareBoolOp<mlir::arith::CmpIOp>(
          mlir::arith::CmpIPredicate::eq, x);
    case Fortran::evaluate::LogicalOperator::Neqv:
      return createCompareBoolOp<mlir::arith::CmpIOp>(
          mlir::arith::CmpIPredicate::ne, x);
    case Fortran::evaluate::LogicalOperator::Not:
      llvm_unreachable(".NOT. handled elsewhere");
    }
    llvm_unreachable("unhandled case");
  }

  //===--------------------------------------------------------------------===//
  // Relational operators (<, <=, ==, etc.)
  //===--------------------------------------------------------------------===//

  template <typename OP, typename PRED, typename A>
  CC createCompareOp(PRED pred, const A &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = fir::getBase(lf(iters));
      auto rhs = fir::getBase(rf(iters));
      return builder.create<OP>(loc, pred, lhs, rhs);
    };
  }
  template <typename A>
  CC createCompareCharOp(mlir::arith::CmpIPredicate pred, const A &x) {
    auto loc = getLoc();
    auto lf = genarr(x.left());
    auto rf = genarr(x.right());
    return [=](IterSpace iters) -> ExtValue {
      auto lhs = lf(iters);
      auto rhs = rf(iters);
      return fir::runtime::genCharCompare(builder, loc, pred, lhs, rhs);
    };
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Integer, KIND>> &x) {
    return createCompareOp<mlir::arith::CmpIOp>(translateRelational(x.opr), x);
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Character, KIND>> &x) {
    return createCompareCharOp(translateRelational(x.opr), x);
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Real, KIND>> &x) {
    return createCompareOp<mlir::arith::CmpFOp>(translateFloatRelational(x.opr),
                                                x);
  }
  template <int KIND>
  CC genarr(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                Fortran::common::TypeCategory::Complex, KIND>> &x) {
    return createCompareOp<fir::CmpcOp>(translateFloatRelational(x.opr), x);
  }
  CC genarr(
      const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &r) {
    return std::visit([&](const auto &x) { return genarr(x); }, r.u);
  }

  template <typename A>
  CC genarr(const Fortran::evaluate::Designator<A> &des) {
    ComponentPath components(des.Rank() > 0);
    return std::visit([&](const auto &x) { return genarr(x, components); },
                      des.u);
  }

  /// Is the path component rank > 0?
  static bool ranked(const PathComponent &x) {
    return std::visit(Fortran::common::visitors{
                          [](const ImplicitSubscripts &) { return false; },
                          [](const auto *v) { return v->Rank() > 0; }},
                      x);
  }

  //===-------------------------------------------------------------------===//
  // Array data references in an explicit iteration space.
  //
  // Use the base array that was loaded before the loop nest.
  //===-------------------------------------------------------------------===//

  /// Lower the path (`revPath`, in reverse) to be appended to an array_fetch or
  /// array_update op. \p ty is the initial type of the array
  /// (reference). Returns the type of the element after application of the
  /// path in \p components.
  ///
  /// TODO: This needs to deal with array's with initial bounds other than 1.
  /// TODO: Thread type parameters correctly.
  mlir::Type lowerPath(const ExtValue &arrayExv, ComponentPath &components) {
    mlir::Location loc = getLoc();
    mlir::Type ty = fir::getBase(arrayExv).getType();
    auto &revPath = components.reversePath;
    ty = fir::unwrapPassByRefType(ty);
    bool prefix = true;
    auto addComponent = [&](mlir::Value v) {
      if (prefix)
        components.prefixComponents.push_back(v);
      else
        components.suffixComponents.push_back(v);
    };
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    bool atBase = true;
    auto saveSemant = semant;
    if (isProjectedCopyInCopyOut())
      semant = ConstituentSemantics::RefTransparent;
    for (const auto &v : llvm::reverse(revPath)) {
      std::visit(
          Fortran::common::visitors{
              [&](const ImplicitSubscripts &) {
                prefix = false;
                ty = fir::unwrapSequenceType(ty);
              },
              [&](const Fortran::evaluate::ComplexPart *x) {
                assert(!prefix && "complex part must be at end");
                auto offset = builder.createIntegerConstant(
                    loc, builder.getI32Type(),
                    x->part() == Fortran::evaluate::ComplexPart::Part::RE ? 0
                                                                          : 1);
                components.suffixComponents.push_back(offset);
                ty = fir::applyPathToType(ty, mlir::ValueRange{offset});
              },
              [&](const Fortran::evaluate::ArrayRef *x) {
                if (Fortran::lower::isRankedArrayAccess(*x)) {
                  genSliceIndices(components, arrayExv, *x);
                } else {
                  // Array access where the expressions are scalar and cannot
                  // depend upon the implied iteration space.
                  unsigned ssIndex = 0u;
                  for (const auto &ss : x->subscript()) {
                    std::visit(
                        Fortran::common::visitors{
                            [&](const Fortran::evaluate::
                                    IndirectSubscriptIntegerExpr &ie) {
                              const auto &e = ie.value();
                              if (isArray(e))
                                fir::emitFatalError(
                                    loc,
                                    "multiple components along single path "
                                    "generating array subexpressions");
                              // Lower scalar index expression, append it to
                              // subs.
                              auto subscriptVal = fir::getBase(asScalar(e));
                              // arrayExv is the base array. It needs to reflect
                              // the current array component instead.
                              // FIXME: must use lower bound of this component,
                              // not just the constant 1.
                              mlir::Value lb =
                                  atBase ? fir::factory::readLowerBound(
                                               builder, loc, arrayExv, ssIndex,
                                               one)
                                         : one;
                              mlir::Value val = builder.createConvert(
                                  loc, idxTy, subscriptVal);
                              mlir::Value ivAdj =
                                  builder.create<mlir::arith::SubIOp>(
                                      loc, idxTy, val, lb);
                              addComponent(
                                  builder.createConvert(loc, idxTy, ivAdj));
                            },
                            [&](const auto &) {
                              fir::emitFatalError(
                                  loc, "multiple components along single path "
                                       "generating array subexpressions");
                            }},
                        ss.u);
                    ssIndex++;
                  }
                }
                ty = fir::unwrapSequenceType(ty);
              },
              [&](const Fortran::evaluate::Component *x) {
                auto fieldTy = fir::FieldType::get(builder.getContext());
                auto name = toStringRef(x->GetLastSymbol().name());
                auto recTy = ty.cast<fir::RecordType>();
                ty = recTy.getType(name);
                auto fld = builder.create<fir::FieldIndexOp>(
                    loc, fieldTy, name, recTy, fir::getTypeParams(arrayExv));
                addComponent(fld);
              }},
          v);
      atBase = false;
    }
    semant = saveSemant;
    ty = fir::unwrapSequenceType(ty);
    components.applied = true;
    return ty;
  }

  llvm::SmallVector<mlir::Value> genSubstringBounds(ComponentPath &components) {
    llvm::SmallVector<mlir::Value> result;
    if (components.substring)
      populateBounds(result, components.substring);
    return result;
  }

  CC applyPathToArrayLoad(fir::ArrayLoadOp load, ComponentPath &components) {
    auto loc = getLoc();
    auto revPath = components.reversePath;
    auto arrayExv = arrayLoadExtValue(builder, loc, load, {}, load);
    auto eleTy = lowerPath(arrayExv, components);
    auto currentPC = components.pc;
    auto pc = [=, prefix = components.prefixComponents,
               suffix = components.suffixComponents](IterSpace iters) {
      IterationSpace newIters = currentPC(iters);
      // Add path prefix and suffix.
      IterationSpace addIters(newIters, prefix, suffix);
      return addIters;
    };
    components.pc = [=](IterSpace iters) { return iters; };
    auto substringBounds = genSubstringBounds(components);
    if (isProjectedCopyInCopyOut()) {
      destination = load;
      auto lambda = [=, esp = this->explicitSpace](IterSpace iters) mutable {
        auto innerArg = esp->findArgumentOfLoad(load);
        if (isAdjustedArrayElementType(eleTy)) {
          auto eleRefTy = builder.getRefType(eleTy);
          auto arrayOp = builder.create<fir::ArrayAccessOp>(
              loc, eleRefTy, innerArg, iters.iterVec(), load.typeparams());
          if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
            auto dstLen = fir::factory::genLenOfCharacter(
                builder, loc, load, iters.iterVec(), substringBounds);
            auto amend = createCharArrayAmend(loc, builder, arrayOp, dstLen,
                                              iters.elementExv(), innerArg,
                                              substringBounds);
            return arrayLoadExtValue(builder, loc, load, iters.iterVec(), amend,
                                     dstLen);
          } else if (fir::isa_derived(eleTy)) {
            auto amend =
                createDerivedArrayAmend(loc, load, builder, arrayOp,
                                        iters.elementExv(), eleTy, innerArg);
            return arrayLoadExtValue(builder, loc, load, iters.iterVec(),
                                     amend);
          }
          assert(eleTy.isa<fir::SequenceType>());
          TODO(loc, "array (as element) assignment");
        }
        auto castedElement =
            builder.createConvert(loc, eleTy, iters.getElement());
        auto update = builder.create<fir::ArrayUpdateOp>(
            loc, innerArg.getType(), innerArg, castedElement, iters.iterVec(),
            load.typeparams());
        return arrayLoadExtValue(builder, loc, load, iters.iterVec(), update);
      };
      return [=](IterSpace iters) mutable { return lambda(pc(iters)); };
    }
    if (isCustomCopyInCopyOut()) {
      // Create an array_modify to get the LHS element address and indicate
      // the assignment, and create the call to the user defined assignment.
      destination = load;
      auto lambda = [=](IterSpace iters) mutable {
        auto innerArg = explicitSpace->findArgumentOfLoad(load);
        auto refEleTy =
            fir::isa_ref_type(eleTy) ? eleTy : builder.getRefType(eleTy);
        auto arrModify = builder.create<fir::ArrayModifyOp>(
            loc, mlir::TypeRange{refEleTy, innerArg.getType()}, innerArg,
            iters.iterVec(), load.typeparams());
        return arrayLoadExtValue(builder, loc, load, iters.iterVec(),
                                 arrModify.getResult(1));
      };
      return [=](IterSpace iters) mutable { return lambda(pc(iters)); };
    }
    auto lambda = [=, semant = this->semant](IterSpace iters) mutable {
      if (semant == ConstituentSemantics::RefOpaque ||
          isAdjustedArrayElementType(eleTy)) {
        auto resTy = builder.getRefType(eleTy);
        // Use array element reference semantics.
        auto access = builder.create<fir::ArrayAccessOp>(
            loc, resTy, load, iters.iterVec(), load.typeparams());
        mlir::Value newBase = access;
        if (fir::isa_char(eleTy)) {
          auto dstLen = fir::factory::genLenOfCharacter(
              builder, loc, load, iters.iterVec(), substringBounds);
          if (!substringBounds.empty()) {
            fir::CharBoxValue charDst{access, dstLen};
            fir::factory::CharacterExprHelper helper{builder, loc};
            charDst = helper.createSubstring(charDst, substringBounds);
            newBase = charDst.getAddr();
          }
          return arrayLoadExtValue(builder, loc, load, iters.iterVec(), newBase,
                                   dstLen);
        }
        return arrayLoadExtValue(builder, loc, load, iters.iterVec(), newBase);
      }
      auto fetch = builder.create<fir::ArrayFetchOp>(
          loc, eleTy, load, iters.iterVec(), load.typeparams());
      return arrayLoadExtValue(builder, loc, load, iters.iterVec(), fetch);
    };
    return [=](IterSpace iters) mutable {
      auto newIters = pc(iters);
      return lambda(newIters);
    };
  }

  template <typename A>
  CC genImplicitArrayAccess(const A &x, ComponentPath &components) {
    components.reversePath.push_back(ImplicitSubscripts{});
    auto exv = asScalarRef(x);
    lowerPath(exv, components);
    auto lambda = genarr(exv, components);
    return [=](IterSpace iters) { return lambda(components.pc(iters)); };
  }
  CC genImplicitArrayAccess(const Fortran::evaluate::NamedEntity &x,
                            ComponentPath &components) {
    if (x.IsSymbol())
      return genImplicitArrayAccess(x.GetFirstSymbol(), components);
    return genImplicitArrayAccess(x.GetComponent(), components);
  }

  template <typename A>
  CC genAsScalar(const A &x) {
    auto loc = getLoc();
    if (isProjectedCopyInCopyOut()) {
      return [=, &x, builder = &converter.getFirOpBuilder()](
                 IterSpace iters) -> ExtValue {
        auto exv = asScalarRef(x);
        auto val = fir::getBase(exv);
        auto eleTy = fir::unwrapRefType(val.getType());
        if (isAdjustedArrayElementType(eleTy)) {
          if (fir::isa_char(eleTy)) {
            fir::factory::CharacterExprHelper{*builder, loc}.createAssign(
                exv, iters.elementExv());
          } else if (fir::isa_derived(eleTy)) {
            TODO(loc, "assignment of derived type");
          } else {
            fir::emitFatalError(loc, "array type not expected in scalar");
          }
        } else {
          builder->create<fir::StoreOp>(loc, iters.getElement(), val);
        }
        return exv;
      };
    }
    return [=, &x](IterSpace) { return asScalar(x); };
  }

  CC genarr(const Fortran::semantics::Symbol &x, ComponentPath &components) {
    if (explicitSpaceIsActive()) {
      if (x.Rank() > 0)
        components.reversePath.push_back(ImplicitSubscripts{});
      if (auto load = explicitSpace->findBinding(&x))
        return applyPathToArrayLoad(load, components);
    } else {
      return genImplicitArrayAccess(x, components);
    }
    if (pathIsEmpty(components))
      return genAsScalar(x);
    auto loc = getLoc();
    return [=](IterSpace) -> ExtValue {
      fir::emitFatalError(loc, "reached symbol with path");
    };
  }

  /// Lower a component path with or without rank.
  /// Example: <code>array%baz%qux%waldo</code>
  CC genarr(const Fortran::evaluate::Component &x, ComponentPath &components) {
    if (explicitSpaceIsActive()) {
      if (x.base().Rank() == 0 && x.Rank() > 0)
        components.reversePath.push_back(ImplicitSubscripts{});
      if (auto load = explicitSpace->findBinding(&x))
        return applyPathToArrayLoad(load, components);
    } else {
      if (x.base().Rank() == 0)
        return genImplicitArrayAccess(x, components);
    }
    bool atEnd = pathIsEmpty(components);
    components.reversePath.push_back(&x);
    auto result = genarr(x.base(), components);
    if (components.applied)
      return result;
    if (atEnd)
      return genAsScalar(x);
    auto loc = getLoc();
    return [=](IterSpace) -> ExtValue {
      fir::emitFatalError(loc, "reached component with path");
    };
  }

  /// Array reference with subscripts. If this has rank > 0, this is a form
  /// of an array section (slice).
  ///
  /// There are two "slicing" primitives that may be applied on a dimension by
  /// dimension basis: (1) triple notation and (2) vector addressing. Since
  /// dimensions can be selectively sliced, some dimensions may contain
  /// regular scalar expressions and those dimensions do not participate in
  /// the array expression evaluation.
  CC genarr(const Fortran::evaluate::ArrayRef &x, ComponentPath &components) {
    if (explicitSpaceIsActive()) {
      if (Fortran::lower::isRankedArrayAccess(x))
        components.reversePath.push_back(ImplicitSubscripts{});
      if (auto load = explicitSpace->findBinding(&x)) {
        components.reversePath.push_back(&x);
        return applyPathToArrayLoad(load, components);
      }
    } else {
      if (Fortran::lower::isRankedArrayAccess(x)) {
        components.reversePath.push_back(&x);
        return genImplicitArrayAccess(x.base(), components);
      }
    }
    bool atEnd = pathIsEmpty(components);
    components.reversePath.push_back(&x);
    auto result = genarr(x.base(), components);
    if (components.applied)
      return result;
    auto loc = getLoc();
    if (atEnd) {
      if (x.Rank() == 0)
        return genAsScalar(x);
      fir::emitFatalError(loc, "expected scalar");
    }
    return [=](IterSpace) -> ExtValue {
      fir::emitFatalError(loc, "reached arrayref with path");
    };
  }

  CC genarr(const Fortran::evaluate::CoarrayRef &x, ComponentPath &components) {
    TODO(getLoc(), "coarray reference");
  }

  CC genarr(const Fortran::evaluate::NamedEntity &x,
            ComponentPath &components) {
    return x.IsSymbol() ? genarr(x.GetFirstSymbol(), components)
                        : genarr(x.GetComponent(), components);
  }

  CC genarr(const Fortran::evaluate::DataRef &x, ComponentPath &components) {
    return std::visit([&](const auto &v) { return genarr(v, components); },
                      x.u);
  }

  bool pathIsEmpty(const ComponentPath &components) {
    return components.reversePath.empty();
  }

  explicit ArrayExprLowering(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::StatementContext &stmtCtx,
                             Fortran::lower::SymMap &symMap)
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap} {}

  explicit ArrayExprLowering(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::StatementContext &stmtCtx,
                             Fortran::lower::SymMap &symMap,
                             ConstituentSemantics sem)
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap}, semant{sem} {}

  explicit ArrayExprLowering(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::StatementContext &stmtCtx,
                             Fortran::lower::SymMap &symMap,
                             ConstituentSemantics sem,
                             Fortran::lower::ExplicitIterSpace *expSpace,
                             Fortran::lower::ImplicitIterSpace *impSpace)
      : converter{converter}, builder{converter.getFirOpBuilder()},
        stmtCtx{stmtCtx}, symMap{symMap},
        explicitSpace(expSpace->isActive() ? expSpace : nullptr),
        implicitSpace(impSpace->empty() ? nullptr : impSpace), semant{sem} {
    // Generate any mask expressions, as necessary. This is the compute step
    // that creates the effective masks. See 10.2.3.2 in particular.
    genMasks();
  }

  mlir::Location getLoc() { return converter.getCurrentLocation(); }

  /// Array appears in a lhs context such that it is assigned after the rhs is
  /// fully evaluated.
  inline bool isCopyInCopyOut() {
    return semant == ConstituentSemantics::CopyInCopyOut;
  }

  /// Array appears in a lhs (or temp) context such that a projected,
  /// discontiguous subspace of the array is assigned after the rhs is fully
  /// evaluated. That is, the rhs array value is merged into a section of the
  /// lhs array.
  inline bool isProjectedCopyInCopyOut() {
    return semant == ConstituentSemantics::ProjectedCopyInCopyOut;
  }

  // ???: Do we still need this?
  inline bool isCustomCopyInCopyOut() {
    return semant == ConstituentSemantics::CustomCopyInCopyOut;
  }

  /// Array appears in a context where it must be boxed.
  inline bool isBoxValue() { return semant == ConstituentSemantics::BoxValue; }

  /// Array appears in a context where differences in the memory reference can
  /// be observable in the computational results. For example, an array
  /// element is passed to an impure procedure.
  inline bool isReferentiallyOpaque() {
    return semant == ConstituentSemantics::RefOpaque;
  }

  /// Array appears in a context where it is passed as a VALUE argument.
  inline bool isValueAttribute() {
    return semant == ConstituentSemantics::ByValueArg;
  }

  /// Can the loops over the expression be unordered?
  inline bool isUnordered() const { return unordered; }

  void setUnordered(bool b) { unordered = b; }

  Fortran::lower::AbstractConverter &converter;
  fir::FirOpBuilder &builder;
  Fortran::lower::StatementContext &stmtCtx;
  Fortran::lower::StatementContext *elementCtx = nullptr;
  Fortran::lower::SymMap &symMap;
  /// The continuation to generate code to update the destination.
  llvm::Optional<CC> ccStoreToDest;
  llvm::Optional<std::function<void(llvm::ArrayRef<mlir::Value>)>> ccPrelude;
  llvm::Optional<std::function<fir::ArrayLoadOp(llvm::ArrayRef<mlir::Value>)>>
      ccLoadDest;
  /// The destination is the loaded array into which the results will be
  /// merged.
  fir::ArrayLoadOp destination;
  /// The shape of the destination.
  llvm::SmallVector<mlir::Value> destShape;
  /// List of arrays in the expression that have been loaded.
  llvm::SmallVector<ArrayOperand> arrayOperands;
  /// If there is a user-defined iteration space, explicitShape will hold the
  /// information from the front end.
  Fortran::lower::ExplicitIterSpace *explicitSpace = nullptr;
  Fortran::lower::ImplicitIterSpace *implicitSpace = nullptr;
  ConstituentSemantics semant = ConstituentSemantics::RefTransparent;
  // Can the array expression be evaluated in any order ?
  // Will be set to false if any of the expression parts prevent this.
  bool unordered = true;
};
} // namespace

fir::ExtendedValue Fortran::lower::createSomeExtendedExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "expr: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx}.genval(expr);
}

fir::ExtendedValue Fortran::lower::createSomeInitializerExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "expr: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx,
                            /*initializer=*/true}
      .genval(expr);
}

fir::ExtendedValue Fortran::lower::createSomeExtendedAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "address: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx}.gen(expr);
}

void Fortran::lower::createSomeArrayAssignment(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(lhs.AsFortran(llvm::dbgs() << "onto array: ") << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ") << '\n';);
  ArrayExprLowering::lowerArrayAssignment(converter, symMap, stmtCtx, lhs, rhs);
}

void Fortran::lower::createSomeArrayAssignment(
    Fortran::lower::AbstractConverter &converter, const fir::ExtendedValue &lhs,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(llvm::dbgs() << "onto array: " << lhs << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ") << '\n';);
  ArrayExprLowering::lowerArrayAssignment(converter, symMap, stmtCtx, lhs, rhs);
}
void Fortran::lower::createSomeArrayAssignment(
    Fortran::lower::AbstractConverter &converter, const fir::ExtendedValue &lhs,
    const fir::ExtendedValue &rhs, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(llvm::dbgs() << "onto array: " << lhs << '\n';
             llvm::dbgs() << "assign expression: " << rhs << '\n';);
  ArrayExprLowering::lowerArrayAssignment(converter, symMap, stmtCtx, lhs, rhs);
}

void Fortran::lower::createAnyMaskedArrayAssignment(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
    Fortran::lower::ExplicitIterSpace &explicitSpace,
    Fortran::lower::ImplicitIterSpace &implicitSpace,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(lhs.AsFortran(llvm::dbgs() << "onto array: ") << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ")
             << " given the explicit iteration space:\n"
             << explicitSpace << "\n and implied mask conditions:\n"
             << implicitSpace << '\n';);
  ArrayExprLowering::lowerAnyMaskedArrayAssignment(
      converter, symMap, stmtCtx, lhs, rhs, explicitSpace, implicitSpace);
}

void Fortran::lower::createAllocatableArrayAssignment(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &lhs,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &rhs,
    Fortran::lower::ExplicitIterSpace &explicitSpace,
    Fortran::lower::ImplicitIterSpace &implicitSpace,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(lhs.AsFortran(llvm::dbgs() << "defining array: ") << '\n';
             rhs.AsFortran(llvm::dbgs() << "assign expression: ")
             << " given the explicit iteration space:\n"
             << explicitSpace << "\n and implied mask conditions:\n"
             << implicitSpace << '\n';);
  ArrayExprLowering::lowerAllocatableArrayAssignment(
      converter, symMap, stmtCtx, lhs, rhs, explicitSpace, implicitSpace);
}

fir::ExtendedValue Fortran::lower::createSomeArrayTempValue(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "array value: ") << '\n');
  return ArrayExprLowering::lowerNewArrayExpression(converter, symMap, stmtCtx,
                                                    expr);
}

void Fortran::lower::createLazyArrayTempValue(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, mlir::Value raggedHeader,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "array value: ") << '\n');
  ArrayExprLowering::lowerLazyArrayExpression(converter, symMap, stmtCtx, expr,
                                              raggedHeader);
}

fir::ExtendedValue Fortran::lower::createSomeArrayBox(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "box designator: ") << '\n');
  return ArrayExprLowering::lowerBoxedArrayExpression(converter, symMap,
                                                      stmtCtx, expr);
}

fir::MutableBoxValue Fortran::lower::createMutableBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap) {
  // MutableBox lowering StatementContext does not need to be propagated
  // to the caller because the result value is a variable, not a temporary
  // expression. The StatementContext clean-up can occur before using the
  // resulting MutableBoxValue. Variables of all other types are handled in the
  // bridge.
  Fortran::lower::StatementContext dummyStmtCtx;
  return ScalarExprLowering{loc, converter, symMap, dummyStmtCtx}
      .genMutableBoxValue(expr);
}

mlir::Value Fortran::lower::createSubroutineCall(
    AbstractConverter &converter, const evaluate::ProcedureRef &call,
    ExplicitIterSpace &explicitIterSpace, ImplicitIterSpace &implicitIterSpace,
    SymMap &symMap, StatementContext &stmtCtx, bool isUserDefAssignment) {
  auto loc = converter.getCurrentLocation();

  if (isUserDefAssignment) {
    assert(call.arguments().size() == 2);
    const auto *lhs = call.arguments()[0].value().UnwrapExpr();
    const auto *rhs = call.arguments()[1].value().UnwrapExpr();
    assert(lhs && rhs &&
           "user defined assignment arguments must be expressions");
    if (call.IsElemental() && lhs->Rank() > 0) {
      // Elemental user defined assignment has special requirements to deal with
      // LHS/RHS overlaps. See 10.2.1.5 p2.
      ArrayExprLowering::lowerElementalUserAssignment(
          converter, symMap, stmtCtx, explicitIterSpace, implicitIterSpace,
          call);
    } else if (explicitIterSpace.isActive() && lhs->Rank() == 0) {
      // Scalar defined assignment (elemental or not) in a FORALL context.
      auto func = Fortran::lower::CallerInterface(call, converter).getFuncOp();
      ArrayExprLowering::lowerScalarUserAssignment(
          converter, symMap, stmtCtx, explicitIterSpace, func, *lhs, *rhs);
    } else if (explicitIterSpace.isActive()) {
      // TODO: need to array fetch/modify sub-arrays ?
      TODO(loc, "non elemental user defined array assignment inside FORALL");
    } else {
      if (!implicitIterSpace.empty())
        fir::emitFatalError(
            loc,
            "C1032: user defined assignment inside WHERE must be elemental");
      // Non elemental user defined assignment outside of FORALL and WHERE.
      // FIXME: The non elemental user defined assignment case with array
      // arguments must be take into account potential overlap. So far the front
      // end does not add parentheses around the RHS argument in the call as it
      // should according to 15.4.3.4.3 p2.
      Fortran::semantics::SomeExpr expr{call};
      Fortran::lower::createSomeExtendedExpression(loc, converter, expr, symMap,
                                                   stmtCtx);
    }
    return {};
  }

  assert(implicitIterSpace.empty() && !explicitIterSpace.isActive() &&
         "subroutine calls are not allowed inside WHERE and FORALL");

  if (isElementalProcWithArrayArgs(call)) {
    Fortran::semantics::SomeExpr expr{call};
    ArrayExprLowering::lowerElementalSubroutine(converter, symMap, stmtCtx,
                                                expr);
    return {};
  }
  // Simple subroutine call, with potential alternate return.
  Fortran::semantics::SomeExpr expr{call};
  auto res = Fortran::lower::createSomeExtendedExpression(loc, converter, expr,
                                                          symMap, stmtCtx);
  return fir::getBase(res);
}

template <typename A>
fir::ArrayLoadOp genArrayLoad(mlir::Location loc,
                              Fortran::lower::AbstractConverter &converter,
                              fir::FirOpBuilder &builder, const A *x,
                              Fortran::lower::SymMap &symMap,
                              Fortran::lower::StatementContext &stmtCtx) {
  auto exv = ScalarExprLowering{loc, converter, symMap, stmtCtx}.gen(*x);
  auto addr = fir::getBase(exv);
  auto shapeOp = builder.createShape(loc, exv);
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(addr.getType());
  return builder.create<fir::ArrayLoadOp>(loc, arrTy, addr, shapeOp,
                                          /*slice=*/mlir::Value{},
                                          fir::getTypeParams(exv));
}
template <>
fir::ArrayLoadOp
genArrayLoad(mlir::Location loc, Fortran::lower::AbstractConverter &converter,
             fir::FirOpBuilder &builder, const Fortran::evaluate::ArrayRef *x,
             Fortran::lower::SymMap &symMap,
             Fortran::lower::StatementContext &stmtCtx) {
  if (x->base().IsSymbol())
    return genArrayLoad(loc, converter, builder, &x->base().GetLastSymbol(),
                        symMap, stmtCtx);
  return genArrayLoad(loc, converter, builder, &x->base().GetComponent(),
                      symMap, stmtCtx);
}

void Fortran::lower::createArrayLoads(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::ExplicitIterSpace &esp, Fortran::lower::SymMap &symMap) {
  auto counter = esp.getCounter();
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto &stmtCtx = esp.stmtContext();
  // Gen the fir.array_load ops.
  auto genLoad = [&](const auto *x) -> fir::ArrayLoadOp {
    return genArrayLoad(loc, converter, builder, x, symMap, stmtCtx);
  };
  if (esp.lhsBases[counter].hasValue()) {
    auto &base = esp.lhsBases[counter].getValue();
    auto load = std::visit(genLoad, base);
    esp.initialArgs.push_back(load);
    esp.resetInnerArgs();
    esp.bindLoad(base, load);
  }
  for (auto &base : esp.rhsBases[counter])
    esp.bindLoad(base, std::visit(genLoad, base));
}

void Fortran::lower::createArrayMergeStores(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::ExplicitIterSpace &esp) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  builder.setInsertionPointAfter(esp.getOuterLoop());
  // Gen the fir.array_merge_store ops for all LHS arrays.
  for (auto i : llvm::enumerate(esp.getOuterLoop().getResults()))
    if (auto ldOpt = esp.getLhsLoad(i.index())) {
      auto load = ldOpt.getValue();
      builder.create<fir::ArrayMergeStoreOp>(
          loc, load, i.value(), load.memref(), load.slice(), load.typeparams());
    }
  if (esp.loopCleanup.hasValue()) {
    esp.loopCleanup.getValue()(builder);
    esp.loopCleanup = llvm::None;
  }
  esp.initialArgs.clear();
  esp.innerArgs.clear();
  esp.outerLoop = llvm::None;
  esp.resetBindings();
  esp.incrementCounter();
}

mlir::Value Fortran::lower::genMaxWithZero(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value value) {
  mlir::Value zero = builder.createIntegerConstant(loc, value.getType(), 0);
  if (auto definingOp = value.getDefiningOp())
    if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(definingOp))
      if (auto intAttr = cst.value().dyn_cast<mlir::IntegerAttr>())
        return intAttr.getInt() < 0 ? zero : value;
  return Fortran::lower::genMax(builder, loc,
                                llvm::SmallVector<mlir::Value>{value, zero});
}
