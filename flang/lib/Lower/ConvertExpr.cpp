//===-- ConvertExpr.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertExpr.h"
#include "SymbolMap.h"
#include "flang/Common/default-kinds.h"
#include "flang/Common/unwrap.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/real.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CharRT.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define TODO() llvm_unreachable("not yet implemented")

// temporary band-aid
static mlir::Value getRootValue(const Fortran::lower::ExValue &exv) {
  return std::visit(Fortran::common::visitors{
                        [](const fir::UnboxedValue &x) { return x; },
                        [](const auto &x) { return x.getAddr(); },
                    },
                    exv);
}

namespace {

/// Lowering of Fortran::evaluate::Expr<T> expressions
class ExprLowering {
public:
  explicit ExprLowering(mlir::Location loc,
                        Fortran::lower::AbstractConverter &converter,
                        const Fortran::lower::SomeExpr &vop,
                        Fortran::lower::SymMap &map)
      : location{loc}, converter{converter},
        builder{converter.getFirOpBuilder()}, expr{vop}, symMap{map} {}

  /// Lower the expression `expr` into MLIR standard dialect
  mlir::Value gen() { return getRootValue(gen(expr)); }

  mlir::Value genval() {
    // FIXME: should really return the extended value
    return getRootValue(genval(expr));
  }

private:
  mlir::Location location;
  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::FirOpBuilder &builder;
  const Fortran::lower::SomeExpr &expr;
  Fortran::lower::SymMap &symMap;

  mlir::Location getLoc() { return location; }

  mlir::Value genUnboxedValue(const Fortran::lower::ExValue &val) {
    return std::visit(
        Fortran::common::visitors{[](const fir::UnboxedValue &x) { return x; },
                                  [](const auto &) { return mlir::Value{}; }},
        val);
  }
  template <typename A>
  mlir::Value genunbox(const A &expr) {
    return genUnboxedValue(genval(expr));
  }

  /// Convert parser's INTEGER relational operators to MLIR.  TODO: using
  /// unordered, but we may want to cons ordered in certain situation.
  static mlir::CmpIPredicate
  translateRelational(Fortran::common::RelationalOperator rop) {
    switch (rop) {
    case Fortran::common::RelationalOperator::LT:
      return mlir::CmpIPredicate::slt;
    case Fortran::common::RelationalOperator::LE:
      return mlir::CmpIPredicate::sle;
    case Fortran::common::RelationalOperator::EQ:
      return mlir::CmpIPredicate::eq;
    case Fortran::common::RelationalOperator::NE:
      return mlir::CmpIPredicate::ne;
    case Fortran::common::RelationalOperator::GT:
      return mlir::CmpIPredicate::sgt;
    case Fortran::common::RelationalOperator::GE:
      return mlir::CmpIPredicate::sge;
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
  static mlir::CmpFPredicate
  translateFloatRelational(Fortran::common::RelationalOperator rop) {
    switch (rop) {
    case Fortran::common::RelationalOperator::LT:
      return mlir::CmpFPredicate::OLT;
    case Fortran::common::RelationalOperator::LE:
      return mlir::CmpFPredicate::OLE;
    case Fortran::common::RelationalOperator::EQ:
      return mlir::CmpFPredicate::OEQ;
    case Fortran::common::RelationalOperator::NE:
      return mlir::CmpFPredicate::UNE;
    case Fortran::common::RelationalOperator::GT:
      return mlir::CmpFPredicate::OGT;
    case Fortran::common::RelationalOperator::GE:
      return mlir::CmpFPredicate::OGE;
    }
    llvm_unreachable("unhandled REAL relational operator");
  }

  /// Generate an integral constant of `value`
  template <int KIND>
  mlir::Value genIntegerConstant(mlir::MLIRContext *context,
                                 std::int64_t value) {
    auto type = converter.genType(Fortran::lower::IntegerCat, KIND);
    auto attr = builder.getIntegerAttr(type, value);
    return builder.create<mlir::ConstantOp>(getLoc(), type, attr);
  }

  /// Generate a logical/boolean constant of `value`
  mlir::Value genBoolConstant(mlir::MLIRContext *context, bool value) {
    auto i1Type = builder.getI1Type();
    auto attr = builder.getIntegerAttr(i1Type, value ? 1 : 0);
    return builder.create<mlir::ConstantOp>(getLoc(), i1Type, attr).getResult();
  }

  template <int KIND>
  mlir::Value genRealConstant(mlir::MLIRContext *context,
                              const llvm::APFloat &value) {
    auto fltTy = Fortran::lower::convertReal(context, KIND);
    auto attr = builder.getFloatAttr(fltTy, value);
    auto res = builder.create<mlir::ConstantOp>(getLoc(), fltTy, attr);
    return res.getResult();
  }

  mlir::Type getSomeKindInteger() { return builder.getIndexType(); }

  template <typename OpTy, typename A>
  mlir::Value createBinaryOp(const A &ex, const Fortran::lower::ExValue &lhs,
                             const Fortran::lower::ExValue &rhs) {
    return std::visit(Fortran::common::visitors{
                          [&](const fir::UnboxedValue &lhs,
                              const fir::UnboxedValue &rhs) -> mlir::Value {
                            assert(lhs && rhs && "argument did not lower");
                            return builder.create<OpTy>(getLoc(), lhs, rhs);
                          },
                          [](const auto &lhs, const auto &rhs) -> mlir::Value {
                            // binary ops can appear in array contexts
                            TODO();
                          }},
                      lhs, rhs);
  }
  template <typename OpTy, typename A>
  mlir::Value createBinaryOp(const A &ex, mlir::Value rhs) {
    return createBinaryOp<OpTy>(ex, genval(ex.left()), rhs);
  }
  template <typename OpTy, typename A>
  mlir::Value createBinaryOp(const A &ex) {
    return createBinaryOp<OpTy>(ex, genval(ex.left()), genval(ex.right()));
  }

  mlir::FuncOp getFunction(llvm::StringRef name, mlir::FunctionType funTy) {
    if (auto func = builder.getNamedFunction(name))
      return func;
    return builder.createFunction(name, funTy);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  mlir::FunctionType createFunctionType() {
    if constexpr (TC == Fortran::lower::IntegerCat) {
      auto output = converter.genType(Fortran::lower::IntegerCat, KIND);
      llvm::SmallVector<mlir::Type, 2> inputs;
      inputs.push_back(output);
      inputs.push_back(output);
      return mlir::FunctionType::get(inputs, output, builder.getContext());
    } else if constexpr (TC == Fortran::lower::RealCat) {
      auto output = Fortran::lower::convertReal(builder.getContext(), KIND);
      llvm::SmallVector<mlir::Type, 2> inputs;
      inputs.push_back(output);
      inputs.push_back(output);
      return mlir::FunctionType::get(inputs, output, builder.getContext());
    } else {
      llvm_unreachable("this category is not implemented");
    }
  }

  template <typename OpTy>
  mlir::Value createCompareOp(mlir::CmpIPredicate pred,
                              const Fortran::lower::ExValue &lhs,
                              const Fortran::lower::ExValue &rhs) {
    return std::visit(
        Fortran::common::visitors{
            [&](const fir::UnboxedValue &lhs,
                const fir::UnboxedValue &rhs) -> mlir::Value {
              return builder.create<OpTy>(getLoc(), pred, lhs, rhs);
            },
            [](const auto &, const auto &) -> mlir::Value { TODO(); }},
        lhs, rhs);
  }
  template <typename OpTy, typename A>
  mlir::Value createCompareOp(const A &ex, mlir::CmpIPredicate pred) {
    return createCompareOp<OpTy>(pred, genval(ex.left()), genval(ex.right()));
  }

  template <typename OpTy>
  mlir::Value createFltCmpOp(mlir::CmpFPredicate pred,
                             const Fortran::lower::ExValue &lhs,
                             const Fortran::lower::ExValue &rhs) {
    return std::visit(
        Fortran::common::visitors{
            [&](const fir::UnboxedValue &lhs,
                const fir::UnboxedValue &rhs) -> mlir::Value {
              return builder.create<OpTy>(getLoc(), pred, lhs, rhs);
            },
            [](const auto &, const auto &) -> mlir::Value { TODO(); }},
        lhs, rhs);
  }
  template <typename OpTy, typename A>
  mlir::Value createFltCmpOp(const A &ex, mlir::CmpFPredicate pred) {
    return createFltCmpOp<OpTy>(pred, genval(ex.left()), genval(ex.right()));
  }

  /// Create a call to the runtime to compare two CHARACTER values.
  /// Precondition: This assumes that the two values have `fir.boxchar` type.
  mlir::Value createCharCompare(mlir::CmpIPredicate pred,
                                const Fortran::lower::ExValue &lhs,
                                const Fortran::lower::ExValue &rhs) {
    return std::visit(
        Fortran::common::visitors{
            [&](const fir::UnboxedValue &lhs,
                const fir::UnboxedValue &rhs) -> mlir::Value {
              return Fortran::lower::genBoxCharCompare(converter, getLoc(),
                                                       pred, lhs, rhs);
            },
            [](const auto &, const auto &) -> mlir::Value { TODO(); }},
        lhs, rhs);
  }
  template <typename A>
  mlir::Value createCharCompare(const A &ex, mlir::CmpIPredicate pred) {
    return createCharCompare(pred, genval(ex.left()), genval(ex.right()));
  }

  /// Returns a reference to a symbol or its box/boxChar descriptor if it has
  /// one.
  Fortran::lower::ExValue gen(Fortran::semantics::SymbolRef sym) {
    if (auto val = symMap.lookupSymbol(sym))
      return val;
    llvm_unreachable("all symbols should be in the map");
    auto addr = builder.createTemporary(getLoc(), converter.genType(sym),
                                        sym->name().ToString());
    symMap.addSymbol(sym, addr);
    return addr;
  }

  mlir::Value genLoad(mlir::Value addr) {
    return builder.create<fir::LoadOp>(getLoc(), addr);
  }

  Fortran::lower::ExValue genval(Fortran::semantics::SymbolRef sym) {
    auto var = gen(sym);
    if (auto s = genUnboxedValue(var))
      if (fir::isReferenceLike(s.getType()))
        return genLoad(s);
    return var;
  }

  Fortran::lower::ExValue
  genval(const Fortran::evaluate::BOZLiteralConstant &) {
    TODO();
  }
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::ProcedureDesignator &) {
    TODO();
  }
  Fortran::lower::ExValue genval(const Fortran::evaluate::NullPointer &) {
    TODO();
  }
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::StructureConstructor &) {
    TODO();
  }
  Fortran::lower::ExValue genval(const Fortran::evaluate::ImpliedDoIndex &) {
    TODO();
  }

  Fortran::lower::ExValue
  genval(const Fortran::evaluate::DescriptorInquiry &desc) {
    auto descRef = symMap.lookupSymbol(desc.base().GetLastSymbol());
    assert(descRef && "no mlir::Value associated to Symbol");
    auto descType = descRef.getAddr().getType();
    mlir::Value res{};
    switch (desc.field()) {
    case Fortran::evaluate::DescriptorInquiry::Field::Len:
      if (descType.isa<fir::BoxCharType>()) {
        auto lenType = builder.getLengthType();
        res = builder.create<fir::BoxCharLenOp>(getLoc(), lenType, descRef);
      } else if (descType.isa<fir::BoxType>()) {
        TODO();
      } else {
        llvm_unreachable("not a descriptor");
      }
      break;
    default:
      TODO();
    }
    return res;
  }

  template <int KIND>
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::TypeParamInquiry<KIND> &) {
    TODO();
  }

  template <int KIND>
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::ComplexComponent<KIND> &part) {
    builder.setLocation(getLoc());
    auto lhs = genunbox(part.left());
    assert(lhs && "boxed type not handled");
    return builder.extractComplexPart(lhs, part.isImaginaryPart);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  Fortran::lower::ExValue genval(
      const Fortran::evaluate::Negate<Fortran::evaluate::Type<TC, KIND>> &op) {
    auto input = genunbox(op.left());
    assert(input && "boxed value not handled");
    if constexpr (TC == Fortran::lower::IntegerCat) {
      // Currently no Standard/FIR op for integer negation.
      auto zero = genIntegerConstant<KIND>(builder.getContext(), 0);
      return builder.create<mlir::SubIOp>(getLoc(), zero, input);
    } else if constexpr (TC == Fortran::lower::RealCat) {
      return builder.create<fir::NegfOp>(getLoc(), input);
    } else {
      static_assert(TC == Fortran::lower::ComplexCat, "Expected numeric type");
      return createBinaryOp<fir::NegcOp>(op);
    }
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::Add<Fortran::evaluate::Type<TC, KIND>> &op) {
    if constexpr (TC == Fortran::lower::IntegerCat) {
      return createBinaryOp<mlir::AddIOp>(op);
    } else if constexpr (TC == Fortran::lower::RealCat) {
      return createBinaryOp<fir::AddfOp>(op);
    } else {
      static_assert(TC == Fortran::lower::ComplexCat, "Expected numeric type");
      return createBinaryOp<fir::AddcOp>(op);
    }
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::Subtract<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    if constexpr (TC == Fortran::lower::IntegerCat) {
      return createBinaryOp<mlir::SubIOp>(op);
    } else if constexpr (TC == Fortran::lower::RealCat) {
      return createBinaryOp<fir::SubfOp>(op);
    } else {
      static_assert(TC == Fortran::lower::ComplexCat, "Expected numeric type");
      return createBinaryOp<fir::SubcOp>(op);
    }
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::Multiply<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    if constexpr (TC == Fortran::lower::IntegerCat) {
      return createBinaryOp<mlir::MulIOp>(op);
    } else if constexpr (TC == Fortran::lower::RealCat) {
      return createBinaryOp<fir::MulfOp>(op);
    } else {
      static_assert(TC == Fortran::lower::ComplexCat, "Expected numeric type");
      return createBinaryOp<fir::MulcOp>(op);
    }
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  Fortran::lower::ExValue genval(
      const Fortran::evaluate::Divide<Fortran::evaluate::Type<TC, KIND>> &op) {
    if constexpr (TC == Fortran::lower::IntegerCat) {
      return createBinaryOp<mlir::SignedDivIOp>(op);
    } else if constexpr (TC == Fortran::lower::RealCat) {
      return createBinaryOp<fir::DivfOp>(op);
    } else {
      static_assert(TC == Fortran::lower::ComplexCat, "Expected numeric type");
      return createBinaryOp<fir::DivcOp>(op);
    }
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  Fortran::lower::ExValue genval(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &op) {
    auto ty = converter.genType(TC, KIND);
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    assert(lhs && rhs && "boxed value not handled");
    return builder.genPow(ty, lhs, rhs);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  Fortran::lower::ExValue genval(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &op) {
    auto ty = converter.genType(TC, KIND);
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    assert(lhs && rhs && "boxed value not handled");
    return builder.genPow(ty, lhs, rhs);
  }

  template <int KIND>
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::ComplexConstructor<KIND> &op) {
    builder.setLocation(getLoc());
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    assert(lhs && rhs && "boxed value not handled");
    return builder.createComplex(KIND, lhs, rhs);
  }

  template <int KIND>
  Fortran::lower::ExValue genval(const Fortran::evaluate::Concat<KIND> &op) {
    TODO();
  }

  /// MIN and MAX operations
  template <Fortran::common::TypeCategory TC, int KIND>
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    std::string name =
        op.ordering == Fortran::evaluate::Ordering::Greater ? "max"s : "min"s;
    auto type = converter.genType(TC, KIND);
    auto lhs = genunbox(op.left());
    auto rhs = genunbox(op.right());
    assert(lhs && rhs && "boxed value not handled");
    llvm::SmallVector<mlir::Value, 2> operands{lhs, rhs};
    return builder.genIntrinsicCall(name, type, operands);
  }

  template <int KIND>
  Fortran::lower::ExValue genval(const Fortran::evaluate::SetLength<KIND> &) {
    TODO();
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    if constexpr (TC == Fortran::lower::IntegerCat) {
      return createCompareOp<mlir::CmpIOp>(op, translateRelational(op.opr));
    } else if constexpr (TC == Fortran::lower::RealCat) {
      return createFltCmpOp<fir::CmpfOp>(op, translateFloatRelational(op.opr));
    } else if constexpr (TC == Fortran::lower::ComplexCat) {
      bool eq{op.opr == Fortran::common::RelationalOperator::EQ};
      if (!eq && op.opr != Fortran::common::RelationalOperator::NE)
        llvm_unreachable("relation undefined for complex");
      builder.setLocation(getLoc());
      auto lhs = genunbox(op.left());
      auto rhs = genunbox(op.right());
      assert(lhs && rhs && "boxed value not handled");
      return builder.createComplexCompare(lhs, rhs, eq);
    } else {
      static_assert(TC == Fortran::lower::CharacterCat);
      builder.setLocation(getLoc());
      return createCharCompare(op, translateRelational(op.opr));
    }
  }

  Fortran::lower::ExValue
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &op) {
    return std::visit([&](const auto &x) { return genval(x); }, op.u);
  }

  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>,
                                          TC2> &convert) {
    auto ty = converter.genType(TC1, KIND);
    auto operand = genunbox(convert.left());
    assert(operand && "boxed value not handled");
    return builder.createConvert(getLoc(), ty, operand);
  }

  template <typename A>
  Fortran::lower::ExValue genval(const Fortran::evaluate::Parentheses<A> &op) {
    auto input = genunbox(op.left());
    assert(input && "boxed value not handled");
    return builder.create<fir::NoReassocOp>(getLoc(), input.getType(), input);
  }

  template <int KIND>
  Fortran::lower::ExValue genval(const Fortran::evaluate::Not<KIND> &op) {
    auto *context = builder.getContext();
    auto logical = genunbox(op.left());
    assert(logical && "boxed value not handled");
    auto one = genBoolConstant(context, true);
    auto val = builder.createConvert(getLoc(), builder.getI1Type(), logical);
    return builder.create<mlir::XOrOp>(getLoc(), val, one);
  }

  template <int KIND>
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::LogicalOperation<KIND> &op) {
    auto i1Type = builder.getI1Type();
    auto slhs = genunbox(op.left());
    auto srhs = genunbox(op.right());
    assert(slhs && srhs && "boxed value not handled");
    auto lhs = builder.createConvert(getLoc(), i1Type, slhs);
    auto rhs = builder.createConvert(getLoc(), i1Type, srhs);
    switch (op.logicalOperator) {
    case Fortran::evaluate::LogicalOperator::And:
      return createBinaryOp<mlir::AndOp>(op, lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Or:
      return createBinaryOp<mlir::OrOp>(op, lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Eqv:
      return createCompareOp<mlir::CmpIOp>(mlir::CmpIPredicate::eq, lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Neqv:
      return createCompareOp<mlir::CmpIOp>(mlir::CmpIPredicate::ne, lhs, rhs);
    case Fortran::evaluate::LogicalOperator::Not:
      // lib/evaluate expression for .NOT. is Fortran::evaluate::Not<KIND>.
      llvm_unreachable(".NOT. is not a binary operator");
    }
    llvm_unreachable("unhandled logical operation");
  }

  /// Construct a CHARACTER literal
  template <int KIND, typename E>
  fir::CharBoxValue genCharLit(const E &data, int64_t size) {
    auto type = fir::SequenceType::get(
        {size}, fir::CharacterType::get(builder.getContext(), KIND));
    // FIXME: for wider char types, use an array of i16 or i32
    // for now, just fake it that it's a i8 to get it past the C++ compiler
    std::string globalName =
        converter.uniqueCGIdent("cl", (const char *)data.c_str());
    auto global = builder.getNamedGlobal(globalName);
    if (!global)
      global = builder.createGlobalConstant(
          getLoc(), type, globalName,
          [&](Fortran::lower::FirOpBuilder &builder) {
            auto context = builder.getContext();
            // FIXME: more fakery
            auto strAttr =
                mlir::StringAttr::get((const char *)data.c_str(), context);
            auto valTag =
                mlir::Identifier::get(fir::StringLitOp::value(), context);
            mlir::NamedAttribute dataAttr(valTag, strAttr);
            auto sizeTag =
                mlir::Identifier::get(fir::StringLitOp::size(), context);
            mlir::NamedAttribute sizeAttr(sizeTag,
                                          builder.getI64IntegerAttr(size));
            llvm::SmallVector<mlir::NamedAttribute, 2> attrs{dataAttr,
                                                             sizeAttr};
            auto str = builder.create<fir::StringLitOp>(
                getLoc(), llvm::ArrayRef<mlir::Type>{type}, llvm::None, attrs);
            builder.create<fir::HasValueOp>(getLoc(), str);
          });
    auto addr = builder.create<fir::AddrOfOp>(getLoc(), global.resultType(),
                                              global.getSymbol());
    auto len = builder.createIntegerConstant(builder.getLengthType(), size);
    return fir::CharBoxValue{addr, len};
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  mlir::Value genScalarLit(
      const Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>
          &value) {
    if constexpr (TC == Fortran::lower::IntegerCat) {
      return genIntegerConstant<KIND>(builder.getContext(), value.ToInt64());
    } else if constexpr (TC == Fortran::lower::LogicalCat) {
        return genBoolConstant(builder.getContext(), opt->IsTrue());
    } else if constexpr (TC == Fortran::lower::RealCat) {
      std::string str = value.DumpHexadecimal();
      if constexpr (KIND == 2) {
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEhalf(), str};
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
    } else if constexpr (TC == Fortran::lower::ComplexCat) {
        using TR = Fortran::evaluate::Type<Fortran::lower::RealCat, KIND>;
        Fortran::evaluate::ComplexConstructor<KIND> ctor(
            Fortran::evaluate::Expr<TR>{
                Fortran::evaluate::Constant<TR>{opt->REAL()}},
            Fortran::evaluate::Expr<TR>{
                Fortran::evaluate::Constant<TR>{opt->AIMAG()}});
        auto cplx = genunbox(ctor);
        assert(cplx && "boxed value not handled");
        return cplx;
    } else if constexpr (TC == Fortran::lower::CharacterCat) {
      return genCharLit<KIND>(con.GetScalarValue().value(), con.LEN());
      } else {
      llvm_unreachable("unhandled constant");
    }
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  mlir::Value genArrayLit(
      const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
          &con) {
    // TODO:
    // - Multi-dimensional array
    if (con.Rank() > 1) {
      TODO(); // "Multi dimensional array not yet supported"
    }
    fir::SequenceType::Shape shape(1, con.shape()[0]);
    auto arrayTy = fir::SequenceType::get(shape, converter.genType(TC, KIND));
    auto idxTy = builder.getIndexType();
    mlir::Value array = builder.create<fir::UndefOp>(getLoc(), arrayTy);
    Fortran::evaluate::ConstantSubscripts subscripts = con.lbounds();
    std::int64_t counter = 0;
    do {
      auto constant = genScalarLit<TC, KIND>(con.At(subscripts));
      auto idx = builder.createIntegerConstant(idxTy, counter++);
      array = builder.create<fir::InsertValueOp>(getLoc(), arrayTy, array,
                                                 constant, idx);
    } while (con.IncrementSubscripts(subscripts));
    return array;
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  mlir::Value
  genval(const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
             &con) {
    // TODO:
    // - derived type constant
    if (con.Rank() > 0)
      return genArrayLit(con);

    using T = Fortran::evaluate::Type<TC, KIND>;
    const std::optional<Fortran::evaluate::Scalar<T>> &opt =
        con.GetScalarValue();
    if (!opt.has_value())
      llvm_unreachable("constant has no value");
    if constexpr (TC == Fortran::lower::CharacterCat) {
      return genCharLit<KIND>(opt.value(), con.LEN());
    }
    return genScalarLit<TC, KIND>(opt.value());
  }

  template <Fortran::common::TypeCategory TC>
  Fortran::lower::ExValue genval(
      const Fortran::evaluate::Constant<Fortran::evaluate::SomeKind<TC>> &con) {
    if constexpr (TC == Fortran::lower::IntegerCat) {
      auto opt = (*con).ToInt64();
      auto type = getSomeKindInteger();
      auto attr = builder.getIntegerAttr(type, opt);
      auto res = builder.create<mlir::ConstantOp>(getLoc(), type, attr);
      return res.getResult();
    } else {
      llvm_unreachable("unhandled constant of unknown kind");
    }
  }

  template <typename A>
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::ArrayConstructor<A> &) {
    TODO();
  }

  Fortran::lower::ExValue gen(const Fortran::evaluate::ComplexPart &) {
    TODO();
  }
  Fortran::lower::ExValue genval(const Fortran::evaluate::ComplexPart &) {
    TODO();
  }

  /// Reference to a substring.
  Fortran::lower::ExValue gen(const Fortran::evaluate::Substring &s) {
    // Get base string
    auto baseString = std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::DataRef &x) { return gen(x); },
            [&](const Fortran::evaluate::StaticDataObject::Pointer &)
                -> Fortran::lower::ExValue { TODO(); },
        },
        s.parent());
    llvm::SmallVector<mlir::Value, 2> bounds;
    auto lower = genunbox(s.lower());
    assert(lower && "boxed value not handled");
    bounds.push_back(lower);
    if (auto upperBound = s.upper()) {
      auto upper = genunbox(*upperBound);
      assert(upper && "boxed value not handled");
      bounds.push_back(upper);
    }
    // FIXME: a string should be a CharBoxValue
    auto addr = genUnboxedValue(baseString);
    assert(addr && "boxed type not handled");
    return builder.createSubstring(addr, bounds);
  }

  /// The value of a substring.
  Fortran::lower::ExValue genval(const Fortran::evaluate::Substring &ss) {
    // FIXME: why is the value of a substring being lowered the same as the
    // address of a substring?
    return gen(ss);
  }

  Fortran::lower::ExValue genval(const Fortran::evaluate::Triplet &trip) {
    TODO();
  }

  Fortran::lower::ExValue genval(const Fortran::evaluate::Subscript &subs) {
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::IndirectSubscriptIntegerExpr &x) {
              return genval(x.value());
            },
            [&](const Fortran::evaluate::Triplet &x) { return genval(x); },
        },
        subs.u);
  }

  Fortran::lower::ExValue gen(const Fortran::evaluate::DataRef &dref) {
    return std::visit([&](const auto &x) { return gen(x); }, dref.u);
  }
  Fortran::lower::ExValue genval(const Fortran::evaluate::DataRef &dref) {
    return std::visit([&](const auto &x) { return genval(x); }, dref.u);
  }

  // Helper function to turn the left-recursive Component structure into a list.
  // Returns the object used as the base coordinate for the component chain.
  static Fortran::evaluate::DataRef const *
  reverseComponents(const Fortran::evaluate::Component &cmpt,
                    std::list<const Fortran::evaluate::Component *> &list) {
    list.push_front(&cmpt);
    return std::visit(Fortran::common::visitors{
                          [&](const Fortran::evaluate::Component &x) {
                            return reverseComponents(x, list);
                          },
                          [&](auto &) { return &cmpt.base(); },
                      },
                      cmpt.base().u);
  }

  // Return the coordinate of the component reference
  Fortran::lower::ExValue gen(const Fortran::evaluate::Component &cmpt) {
    std::list<const Fortran::evaluate::Component *> list;
    auto *base = reverseComponents(cmpt, list);
    llvm::SmallVector<mlir::Value, 2> coorArgs;
    auto obj = genunbox(*base);
    assert(obj && "boxed value not handled");
    auto *sym = &cmpt.GetFirstSymbol();
    auto ty = converter.genType(*sym);
    for (auto *field : list) {
      sym = &field->GetLastSymbol();
      auto name = sym->name().ToString();
      // FIXME: as we're walking the chain of field names, we need to update the
      // subtype as we drill down
      coorArgs.push_back(builder.create<fir::FieldIndexOp>(getLoc(), name, ty));
    }
    assert(sym && "no component(s)?");
    ty = fir::ReferenceType::get(ty);
    return builder.create<fir::CoordinateOp>(getLoc(), ty, obj, coorArgs);
  }

  Fortran::lower::ExValue genval(const Fortran::evaluate::Component &cmpt) {
    auto c = gen(cmpt);
    if (auto val = genUnboxedValue(c))
      return genLoad(val);
    TODO();
  }

  // Determine the result type after removing `dims` dimensions from the array
  // type `arrTy`
  mlir::Type genSubType(mlir::Type arrTy, unsigned dims) {
    auto unwrapTy = arrTy.cast<fir::ReferenceType>().getEleTy();
    auto seqTy = unwrapTy.cast<fir::SequenceType>();
    auto shape = seqTy.getShape();
    assert(shape.size() > 0 && "removing columns for sequence sans shape");
    assert(dims <= shape.size() && "removing more columns than exist");
    fir::SequenceType::Shape newBnds;
    // follow Fortran semantics and remove columns (from right)
    auto e{shape.size() - dims};
    for (decltype(e) i{0}; i < e; ++i)
      newBnds.push_back(shape[i]);
    if (!newBnds.empty())
      return fir::SequenceType::get(newBnds, seqTy.getEleTy());
    return seqTy.getEleTy();
  }

  // Generate the code for a Bound value.
  Fortran::lower::ExValue genval(const Fortran::semantics::Bound &bound) {
    if (bound.isExplicit()) {
      auto sub = bound.GetExplicit();
      if (sub.has_value())
        return genval(*sub);
      return genIntegerConstant<8>(builder.getContext(), 1);
    }
    TODO();
  }

  Fortran::lower::ExValue
  genArrayRefComponent(const Fortran::evaluate::ArrayRef &aref) {
    auto base = getRootValue(gen(aref.base().GetComponent()));
    llvm::SmallVector<mlir::Value, 8> args;
    for (auto &subsc : aref.subscript()) {
      auto sv = genunbox(subsc);
      assert(sv && "boxed value not handled");
      args.push_back(sv);
    }
    auto ty = genSubType(base.getType(), args.size());
    ty = fir::ReferenceType::get(ty);
    return builder.create<fir::CoordinateOp>(getLoc(), ty, base, args);
  }

  Fortran::lower::ExValue gen(const Fortran::lower::SymIndex &si,
                              const Fortran::evaluate::ArrayRef &aref) {
    auto loc = getLoc();
    auto addr = si.getAddr();
    auto arrTy = fir::dyn_cast_ptrEleTy(addr.getType());
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    auto refTy = fir::ReferenceType::get(eleTy);
    auto base = builder.createConvert(loc, refTy, addr);
    auto idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(idxTy, 1);
    auto zero = builder.createIntegerConstant(idxTy, 0);
    auto genShaped = [&](const auto &arr, mlir::Value delta) -> mlir::Value {
      mlir::Value total = zero;
      assert(arr.shape.size() == aref.subscript().size());
      for (const auto &pair : llvm::zip(arr.shape, aref.subscript())) {
        auto sval = genunbox(std::get<1>(pair));
        assert(sval && "boxed value not handled");
        auto val = builder.createConvert(loc, idxTy, sval);
        auto diff = builder.create<mlir::SubIOp>(loc, val, one);
        auto prod = builder.create<mlir::MulIOp>(loc, delta, diff);
        total = builder.create<mlir::AddIOp>(loc, prod, total);
        delta = builder.create<mlir::MulIOp>(loc, delta, std::get<0>(pair));
      }
      return builder.create<fir::CoordinateOp>(
          loc, refTy, base, llvm::ArrayRef<mlir::Value>{total});
    };
    auto genFullDim = [&](const auto &arr, mlir::Value delta) -> mlir::Value {
      mlir::Value total = zero;
      assert(arr.shape.size() == aref.subscript().size());
      for (const auto &pair : llvm::zip(arr.shape, aref.subscript())) {
        auto sval = genunbox(std::get<1>(pair));
        assert(sval && "boxed value not handled");
        auto val = builder.createConvert(loc, idxTy, sval);
        auto lb =
            builder.createConvert(loc, idxTy, std::get<0>(std::get<0>(pair)));
        auto diff = builder.create<mlir::SubIOp>(loc, val, lb);
        auto prod = builder.create<mlir::MulIOp>(loc, delta, diff);
        total = builder.create<mlir::AddIOp>(loc, prod, total);
        if (auto ext = std::get<1>(std::get<0>(pair)))
          delta = builder.create<mlir::MulIOp>(loc, delta, ext);
      }
      return builder.create<fir::CoordinateOp>(
          loc, refTy, base, llvm::ArrayRef<mlir::Value>{total});
    };
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::lower::SymIndex::Shaped &arr) {
              return genShaped(arr, one);
            },
            [&](const Fortran::lower::SymIndex::FullDim &arr) {
              return genFullDim(arr, one);
            },
            [&](const Fortran::lower::SymIndex::CharShaped &arr) {
              return genShaped(arr, arr.len);
            },
            [&](const Fortran::lower::SymIndex::CharFullDim &arr) {
              return genFullDim(arr, arr.len);
            },
            [&](const Fortran::lower::SymIndex::Derived &arr) {
              TODO();
              return mlir::Value{};
            },
            [&](const auto &) {
              TODO();
              return mlir::Value{};
            }},
        si.v);
  }

  // Return the coordinate of the array reference
  Fortran::lower::ExValue gen(const Fortran::evaluate::ArrayRef &aref) {
    if (aref.base().IsSymbol()) {
      auto &symbol = aref.base().GetFirstSymbol();
      auto si = symMap.lookupSymbol(symbol);
      if (!si.hasConstantShape())
        return gen(si, aref);
      auto box = gen(symbol);
      auto base = getRootValue(box);
      assert(base && "boxed type not handled");
      // FIXME: the shape should already be part of box
      auto &shape =
          symbol.get<Fortran::semantics::ObjectEntityDetails>().shape();
      unsigned i = 0;
      llvm::SmallVector<mlir::Value, 8> args;
      for (auto &subsc : aref.subscript()) {
        auto val = genunbox(subsc);
        auto adj = genunbox(shape[i++].lbound());
        assert(val && adj && "boxed value not handled");
        auto ty = val.getType();
        args.push_back(builder.create<mlir::SubIOp>(getLoc(), ty, val, adj));
      }
      auto ty = genSubType(base.getType(), args.size());
      ty = fir::ReferenceType::get(ty);
      return builder.create<fir::CoordinateOp>(getLoc(), ty, base, args);
    }
    return genArrayRefComponent(aref);
  }

  Fortran::lower::ExValue genval(const Fortran::evaluate::ArrayRef &aref) {
    return genLoad(getRootValue(gen(aref)));
  }

  // Return a coordinate of the coarray reference. This is necessary as a
  // Component may have a CoarrayRef as its base coordinate.
  Fortran::lower::ExValue gen(const Fortran::evaluate::CoarrayRef &coref) {
    // FIXME: need to visit the cosubscripts...
    // return gen(coref.base());
    TODO();
  }
  Fortran::lower::ExValue genval(const Fortran::evaluate::CoarrayRef &coref) {
    return genLoad(getRootValue(gen(coref)));
  }

  template <typename A>
  Fortran::lower::ExValue gen(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return gen(x); }, des.u);
  }
  template <typename A>
  Fortran::lower::ExValue genval(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return genval(x); }, des.u);
  }

  // call a function
  template <typename A>
  Fortran::lower::ExValue gen(const Fortran::evaluate::FunctionRef<A> &funRef) {
    TODO();
  }
  template <typename A>
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::FunctionRef<A> &funRef) {
    TODO(); // Derived type functions (user + intrinsics)
  }

  mlir::Value
  genIntrinsicRef(const Fortran::evaluate::ProcedureRef &procRef,
                  const Fortran::evaluate::SpecificIntrinsic &intrinsic,
                  mlir::ArrayRef<mlir::Type> resultType) {
    if (resultType.size() != 1)
      TODO(); // Intrinsic subroutine

    llvm::SmallVector<mlir::Value, 2> operands;
    // Lower arguments
    // For now, logical arguments for intrinsic are lowered to `fir.logical`
    // so that TRANSFER can work. For some arguments, it could lead to useless
    // conversions (e.g scalar MASK of MERGE will be converted to `i1`), but
    // the generated code is at least correct. To improve this, the intrinsic
    // lowering facility should control argument lowering.
    for (const auto &arg : procRef.arguments()) {
      if (auto *expr = Fortran::evaluate::UnwrapExpr<
              Fortran::evaluate::Expr<Fortran::evaluate::SomeType>>(arg)) {
        auto a = genunbox(*expr);
        assert(a && "boxed argument not handled");
        operands.push_back(a);
      } else {
        operands.push_back(nullptr); // optional
      }
    }
    // Let the intrinsic library lower the intrinsic procedure call
    llvm::StringRef name{intrinsic.name};
    return builder.genIntrinsicCall(name, resultType[0], operands);
  }

  template <typename A>
  bool isCharacterType(const A &exp) {
    if (auto type = exp.GetType())
      return type->category() == Fortran::lower::CharacterCat;
    return false;
  }

  mlir::Value genProcedureRef(const Fortran::evaluate::ProcedureRef procRef,
                              mlir::ArrayRef<mlir::Type> resultType) {
    if (const auto *intrinsic = procRef.proc().GetSpecificIntrinsic())
      return genIntrinsicRef(procRef, *intrinsic, resultType[0]);

    mlir::FunctionType funTy;
    if (auto *sym = procRef.proc().GetSymbol())
      if (auto *iface = Fortran::semantics::FindInterface(*sym))
        funTy = converter.genType(*iface).cast<mlir::FunctionType>();

    // Implicit interface implementation only
    // TODO: Explicit interface, we need to use Characterize here,
    // evaluate::IntrinsicProcTable is required to use it.
    llvm::SmallVector<mlir::Type, 8> argTypes;
    llvm::SmallVector<mlir::Value, 8> operands;
    // Arguments of user functions must be lowered to the correct type.
    for (const auto &arg : procRef.arguments()) {
      if (!arg.has_value())
        TODO(); // optional arguments
      const auto *expr = arg->UnwrapExpr();
      if (!expr)
        TODO(); // assumed type arguments
      if (const auto *sym =
              Fortran::evaluate::UnwrapWholeSymbolDataRef(*expr)) {
        mlir::Value argRef = symMap.lookupSymbol(*sym);
        assert(argRef && "could not get symbol reference");
        if (builder.isCharacter(argRef.getType())) {
          argTypes.push_back(fir::BoxCharType::get(
              builder.getContext(),
              builder.getCharacterKind(argRef.getType())));
          auto ch = builder.materializeCharacter(argRef);
          operands.push_back(builder.createEmboxChar(ch.first, ch.second));
        } else {
          argTypes.push_back(argRef.getType());
          operands.push_back(argRef);
        }
      } else {
        // create a temp to store the expression value
        auto exv = genval(*expr);
        // FIXME: should use the box values, etc.
        auto val = getRootValue(exv);
        mlir::Value addr;
        if (fir::isa_passbyref_type(val.getType())) {
          // expression is already a reference, so just pass it
          addr = val;
        } else {
          // expression is a value, so store it in a temporary so we can
          // pass-by-reference
          addr = builder.createTemporary(getLoc(), val.getType());
          builder.create<fir::StoreOp>(getLoc(), val, addr);
        }
        if (builder.isCharacter(addr.getType())) {
          argTypes.push_back(fir::BoxCharType::get(
              builder.getContext(), builder.getCharacterKind(addr.getType())));
          auto ch = builder.materializeCharacter(addr);
          addr = builder.createEmboxChar(ch.first, ch.second);
        } else {
          argTypes.push_back(addr.getType());
        }
        operands.push_back(addr);
      }
    }
    if (!funTy)
      funTy =
          mlir::FunctionType::get(argTypes, resultType, builder.getContext());

    auto funName = applyNameMangling(procRef.proc());
    auto func = getFunction(funName, funTy);
    if (func.getType() != funTy) {
      // In older Fortran, procedure argument types are inferenced. Deal with
      // the potential mismatches by adding casts to the arguments when the
      // inferenced types do not match exactly.
      llvm::SmallVector<mlir::Value, 8> castedOperands;
      for (const auto &op : llvm::zip(operands, func.getType().getInputs())) {
        auto cast = builder.convertWithSemantics(getLoc(), std::get<1>(op),
                                                 std::get<0>(op));
        castedOperands.push_back(cast);
      }
      operands.swap(castedOperands);
    }
    auto call = builder.create<fir::CallOp>(
        getLoc(), resultType, builder.getSymbolRefAttr(funName), operands);

    if (resultType.size() == 0)
      return {}; // subroutine call
    // For now, Fortran returned values are implemented with a single MLIR
    // function return value.
    assert(call.getNumResults() == 1 &&
           "Expected exactly one result in FUNCTION call");
    return call.getResult(0);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::FunctionRef<Fortran::evaluate::Type<TC, KIND>>
             &funRef) {
    llvm::SmallVector<mlir::Type, 1> resTy;
    resTy.push_back(converter.genType(TC, KIND));
    return genProcedureRef(funRef, resTy);
  }

  Fortran::lower::ExValue
  genval(const Fortran::evaluate::ProcedureRef &procRef) {
    llvm::SmallVector<mlir::Type, 1> resTy;
    if (procRef.HasAlternateReturns())
      resTy.push_back(builder.getIndexType());
    return genProcedureRef(procRef, resTy);
  }

  template <typename A>
  Fortran::lower::ExValue gen(const Fortran::evaluate::Expr<A> &exp) {
    return std::visit([&](const auto &e) { return genref(e); }, exp.u);
  }
  template <typename A>
  Fortran::lower::ExValue genval(const Fortran::evaluate::Expr<A> &exp) {
    return std::visit([&](const auto &e) { return genval(e); }, exp.u);
  }

  template <int KIND>
  Fortran::lower::ExValue
  genval(const Fortran::evaluate::Expr<
         Fortran::evaluate::Type<Fortran::lower::LogicalCat, KIND>> &exp) {
    return std::visit([&](const auto &e) { return genval(e); }, exp.u);
  }

  using RefSet =
      std::tuple<Fortran::evaluate::ComplexPart, Fortran::evaluate::Substring,
                 Fortran::evaluate::DataRef, Fortran::evaluate::Component,
                 Fortran::evaluate::ArrayRef, Fortran::evaluate::CoarrayRef,
                 Fortran::semantics::SymbolRef>;
  template <typename A>
  static constexpr bool inRefSet = Fortran::common::HasMember<A, RefSet>;

  template <typename A>
  Fortran::lower::ExValue genref(const Fortran::evaluate::Designator<A> &x) {
    return gen(x);
  }
  template <typename A>
  Fortran::lower::ExValue genref(const Fortran::evaluate::FunctionRef<A> &x) {
    return gen(x);
  }
  template <typename A>
  Fortran::lower::ExValue genref(const Fortran::evaluate::Expr<A> &x) {
    return gen(x);
  }
  template <typename A>
  Fortran::lower::ExValue genref(const A &a) {
    if constexpr (inRefSet<std::decay_t<decltype(a)>>) {
      return gen(a);
    } else {
      llvm_unreachable("expression error");
    }
  }

  std::string
  applyNameMangling(const Fortran::evaluate::ProcedureDesignator &proc) {
    if (const auto *symbol = proc.GetSymbol())
      return converter.mangleName(*symbol);
    // Do not mangle intrinsic for now
    assert(proc.GetSpecificIntrinsic() &&
           "expected intrinsic procedure in designator");
    return proc.GetName();
  }
};

} // namespace

mlir::Value Fortran::lower::createSomeExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap) {
  return ExprLowering{loc, converter, expr, symMap}.genval();
}

mlir::Value Fortran::lower::createSomeAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap) {
  return ExprLowering{loc, converter, expr, symMap}.gen();
}
