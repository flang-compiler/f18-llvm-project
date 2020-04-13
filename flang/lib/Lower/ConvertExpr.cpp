//===-- ConvertExpr.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertExpr.h"
#include "flang/Common/default-kinds.h"
#include "flang/Common/unwrap.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/real.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
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

namespace {

#define TODO() llvm_unreachable("not yet implemented")

/// Lowering of Fortran::evaluate::Expr<T> expressions
class ExprLowering {
  mlir::Location location;
  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::FirOpBuilder &builder;
  const Fortran::lower::SomeExpr &expr;
  Fortran::lower::SymMap &symMap;
  const Fortran::lower::IntrinsicLibrary &intrinsics;
  bool genLogicalAsI1{false};

  mlir::Location getLoc() { return location; }

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
    auto res = builder.create<mlir::ConstantOp>(getLoc(), type, attr);
    return res.getResult();
  }

  /// Generate a logical/boolean constant of `value`
  mlir::Value genLogicalConstantAsI1(mlir::MLIRContext *context, bool value) {
    auto i1Type = mlir::IntegerType::get(1, builder.getContext());
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

  mlir::Type getSomeKindInteger() {
    return mlir::IndexType::get(builder.getContext());
  }

  template <typename OpTy, typename A>
  mlir::Value createBinaryOp(const A &ex, mlir::Value lhs, mlir::Value rhs) {
    assert(lhs && rhs && "argument did not lower");
    auto x = builder.create<OpTy>(getLoc(), lhs, rhs);
    return x.getResult();
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
    if (auto func = builder.getNamedFunction(name)) {
      assert(func.getType() == funTy &&
             "function already declared with a different type");
      return func;
    }
    return builder.createFunction(name, funTy);
  }

  // FIXME binary operation :: ('a, 'a) -> 'a
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

  template <typename OpTy, typename A>
  mlir::Value createCompareOp(const A &ex, mlir::CmpIPredicate pred,
                              mlir::Value lhs, mlir::Value rhs) {
    assert(lhs && rhs && "argument did not lower");
    auto x = builder.create<OpTy>(getLoc(), pred, lhs, rhs);
    return x.getResult();
  }
  template <typename OpTy, typename A>
  mlir::Value createCompareOp(const A &ex, mlir::CmpIPredicate pred) {
    return createCompareOp<OpTy>(ex, pred, genval(ex.left()),
                                 genval(ex.right()));
  }
  template <typename OpTy, typename A>
  mlir::Value createFltCmpOp(const A &ex, mlir::CmpFPredicate pred,
                             mlir::Value lhs, mlir::Value rhs) {
    assert(lhs && rhs && "argument did not lower");
    auto x = builder.create<OpTy>(getLoc(), pred, lhs, rhs);
    return x.getResult();
  }
  template <typename OpTy, typename A>
  mlir::Value createFltCmpOp(const A &ex, mlir::CmpFPredicate pred) {
    return createFltCmpOp<OpTy>(ex, pred, genval(ex.left()),
                                genval(ex.right()));
  }

  /// Returns a reference to a symbol or its box/boxChar descriptor if it has
  /// one.
  mlir::Value gen(Fortran::semantics::SymbolRef sym) {
    // FIXME: not all symbols are local
    if (auto val = symMap.lookupSymbol(sym))
      return val;
    auto addr = builder.createTemporary(
        getLoc(), converter.genType(sym), sym->name().ToString());
    symMap.addSymbol(sym, addr);
    return addr;
  }

  mlir::Value gendef(Fortran::semantics::SymbolRef sym) { return gen(sym); }

  mlir::Value genval(Fortran::semantics::SymbolRef sym) {
    auto var = gen(sym);
    if (fir::isReferenceLike(var.getType()))
      return builder.create<fir::LoadOp>(getLoc(), var);
    return var;
  }

  mlir::Value genval(const Fortran::evaluate::BOZLiteralConstant &) { TODO(); }
  mlir::Value genval(const Fortran::evaluate::ProcedureRef &procRef) {
    llvm::SmallVector<mlir::Type, 1> resTy;
    return genProcedureRef(procRef, resTy);
  }
  mlir::Value genval(const Fortran::evaluate::ProcedureDesignator &) { TODO(); }
  mlir::Value genval(const Fortran::evaluate::NullPointer &) { TODO(); }
  mlir::Value genval(const Fortran::evaluate::StructureConstructor &) {
    TODO();
  }
  mlir::Value genval(const Fortran::evaluate::ImpliedDoIndex &) { TODO(); }
  mlir::Value genval(const Fortran::evaluate::DescriptorInquiry &desc) {
    auto descRef = symMap.lookupSymbol(desc.base().GetLastSymbol());
    assert(descRef && "no mlir::Value associated to Symbol");
    auto descType = descRef.getType();
    mlir::Value res{};
    switch (desc.field()) {
    case Fortran::evaluate::DescriptorInquiry::Field::Len:
      if (descType.isa<fir::BoxCharType>()) {
        auto lenType{mlir::IntegerType::get(64, builder.getContext())};
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
  mlir::Value genval(const Fortran::evaluate::TypeParamInquiry<KIND> &) {
    TODO();
  }

  template <int KIND>
  mlir::Value genval(const Fortran::evaluate::ComplexComponent<KIND> &part) {
    builder.setLocation(getLoc());
    return builder.extractComplexPart(genval(part.left()),
                                      part.isImaginaryPart);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  mlir::Value genval(
      const Fortran::evaluate::Negate<Fortran::evaluate::Type<TC, KIND>> &op) {
    auto input = genval(op.left());
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
  mlir::Value
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
  mlir::Value
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
  mlir::Value
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
  mlir::Value genval(
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
  mlir::Value genval(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &op) {
    llvm::SmallVector<mlir::Value, 2> operands{genval(op.left()),
                                               genval(op.right())};
    auto ty = converter.genType(TC, KIND);
    return intrinsics.genval(getLoc(), builder, "pow", ty, operands);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  mlir::Value genval(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &op) {
    // TODO: runtime as limited integer kind support. Look if the conversions
    // are ok
    llvm::SmallVector<mlir::Value, 2> operands{genval(op.left()),
                                               genval(op.right())};
    auto ty = converter.genType(TC, KIND);
    return intrinsics.genval(getLoc(), builder, "pow", ty, operands);
  }

  template <int KIND>
  mlir::Value genval(const Fortran::evaluate::ComplexConstructor<KIND> &op) {
    builder.setLocation(getLoc());
    return builder.createComplex(KIND, genval(op.left()), genval(op.right()));
  }

  template <int KIND>
  mlir::Value genval(const Fortran::evaluate::Concat<KIND> &op) {
    TODO();
  }

  /// MIN and MAX operations
  template <Fortran::common::TypeCategory TC, int KIND>
  mlir::Value
  genval(const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    std::string name =
        op.ordering == Fortran::evaluate::Ordering::Greater ? "max"s : "min"s;
    auto type = converter.genType(TC, KIND);
    llvm::SmallVector<mlir::Value, 2> operands{genval(op.left()),
                                               genval(op.right())};
    return intrinsics.genval(getLoc(), builder, name, type, operands);
  }

  template <int KIND>
  mlir::Value genval(const Fortran::evaluate::SetLength<KIND> &) {
    TODO();
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  mlir::Value
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    mlir::Value result{};
    if constexpr (TC == Fortran::lower::IntegerCat) {
      result = createCompareOp<mlir::CmpIOp>(op, translateRelational(op.opr));
    } else if constexpr (TC == Fortran::lower::RealCat) {
      result =
          createFltCmpOp<fir::CmpfOp>(op, translateFloatRelational(op.opr));
    } else if constexpr (TC == Fortran::lower::ComplexCat) {
      bool eq{op.opr == Fortran::common::RelationalOperator::EQ};
      assert(eq || op.opr == Fortran::common::RelationalOperator::NE &&
                       "relation undefined for complex");
      builder.setLocation(getLoc());
      result = builder.createComplexCompare(genval(op.left()),
                                            genval(op.right()), eq);
    } else {
      static_assert(TC == Fortran::lower::CharacterCat);
      TODO();
    }
    return result;
  }

  mlir::Value
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &op) {
    return std::visit([&](const auto &x) { return genval(x); }, op.u);
  }

  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  mlir::Value
  genval(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>,
                                          TC2> &convert) {
    auto ty = converter.genType(TC1, KIND);
    auto operand = genval(convert.left());
    if (TC1 == Fortran::lower::LogicalCat && genLogicalAsI1) {
      // If an i1 result is needed, it does not make sens to convert between
      // `fir.logical` types to later convert back to the result to i1.
      return operand;
    }
    return builder.create<fir::ConvertOp>(getLoc(), ty, operand);
  }

  template <typename A>
  mlir::Value genval(const Fortran::evaluate::Parentheses<A> &op) {
    auto input = genval(op.left());
    return builder.create<fir::NoReassocOp>(getLoc(), input.getType(), input);
  }

  template <int KIND>
  mlir::Value genval(const Fortran::evaluate::Not<KIND> &op) {
    // Request operands to be generated as `i1` and restore after this scope.
    auto restorer = Fortran::common::ScopedSet(genLogicalAsI1, true);
    auto *context = builder.getContext();
    auto logical = genval(op.left());
    auto one = genLogicalConstantAsI1(context, true);
    return builder.create<mlir::XOrOp>(getLoc(), logical, one).getResult();
  }

  template <int KIND>
  mlir::Value genval(const Fortran::evaluate::LogicalOperation<KIND> &op) {
    // Request operands to be generated as `i1` and restore after this scope.
    auto restorer = Fortran::common::ScopedSet(genLogicalAsI1, true);
    mlir::Value result;
    switch (op.logicalOperator) {
    case Fortran::evaluate::LogicalOperator::And:
      result = createBinaryOp<mlir::AndOp>(op);
      break;
    case Fortran::evaluate::LogicalOperator::Or:
      result = createBinaryOp<mlir::OrOp>(op);
      break;
    case Fortran::evaluate::LogicalOperator::Eqv:
      result = createCompareOp<mlir::CmpIOp>(op, mlir::CmpIPredicate::eq);
      break;
    case Fortran::evaluate::LogicalOperator::Neqv:
      result = createCompareOp<mlir::CmpIOp>(op, mlir::CmpIPredicate::ne);
      break;
    case Fortran::evaluate::LogicalOperator::Not:
      // lib/evaluate expression for .NOT. is Fortran::evaluate::Not<KIND>.
      llvm_unreachable(".NOT. is not a binary operator");
      break;
    }
    if (!result)
      llvm_unreachable("unhandled logical operation");
    return result;
  }

  /// Construct a CHARACTER literal
  template <int KIND, typename E>
  mlir::Value genCharLit(const E &data, int64_t size) {
    auto type = fir::SequenceType::get(
        {size}, fir::CharacterType::get(builder.getContext(), KIND));
    // FIXME: for wider char types, use an array of i16 or i32
    // for now, just fake it that it's a i8 to get it past the C++ compiler
    if constexpr (KIND == 1) {
      std::string globalName = converter.uniqueCGIdent(data);
      auto global = builder.getNamedGlobal(globalName);
      if (!global)
        global = builder.createGlobalConstant(
            getLoc(), type, globalName,
            [&](Fortran::lower::FirOpBuilder &builder) {
              auto context = builder.getContext();
              auto strAttr = mlir::StringAttr::get(data.c_str(), context);
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
                  getLoc(), llvm::ArrayRef<mlir::Type>{type}, llvm::None,
                  attrs);
              builder.create<fir::HasValueOp>(getLoc(), str);
            });
      return builder.create<fir::AddrOfOp>(getLoc(), global.resultType(),
                                           global.getSymbol());
    }
    auto context = builder.getContext();
    auto valTag = mlir::Identifier::get(fir::StringLitOp::value(), context);
    auto strAttr = mlir::StringAttr::get((const char *)data.c_str(), context);
    mlir::NamedAttribute dataAttr(valTag, strAttr);
    auto sizeTag = mlir::Identifier::get(fir::StringLitOp::size(), context);
    mlir::NamedAttribute sizeAttr(sizeTag, builder.getI64IntegerAttr(size));
    llvm::SmallVector<mlir::NamedAttribute, 2> attrs{dataAttr, sizeAttr};
    return builder.create<fir::StringLitOp>(
        getLoc(), llvm::ArrayRef<mlir::Type>{type}, llvm::None, attrs);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  mlir::Value
  genval(const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
             &con) {
    // TODO:
    // - character type constant
    // - array constant not handled
    // - derived type constant
    if constexpr (TC == Fortran::lower::IntegerCat) {
      auto opt = con.GetScalarValue();
      if (opt.has_value())
        return genIntegerConstant<KIND>(builder.getContext(), opt->ToInt64());
      llvm_unreachable("integer constant has no value");
    } else if constexpr (TC == Fortran::lower::LogicalCat) {
      auto opt = con.GetScalarValue();
      if (opt.has_value())
        return genLogicalConstantAsI1(builder.getContext(), opt->IsTrue());
      llvm_unreachable("logical constant has no value");
    } else if constexpr (TC == Fortran::lower::RealCat) {
      auto opt = con.GetScalarValue();
      if (opt.has_value()) {
        std::string str = opt.value().DumpHexadecimal();
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
      }
      llvm_unreachable("real constant has no value");
    } else if constexpr (TC == Fortran::lower::ComplexCat) {
      auto opt = con.GetScalarValue();
      if (opt.has_value()) {
        using TR = Fortran::evaluate::Type<Fortran::lower::RealCat, KIND>;
        return genval(Fortran::evaluate::ComplexConstructor<KIND>{
            Fortran::evaluate::Expr<TR>{
                Fortran::evaluate::Constant<TR>{opt->REAL()}},
            Fortran::evaluate::Expr<TR>{
                Fortran::evaluate::Constant<TR>{opt->AIMAG()}}});
      }
      llvm_unreachable("array of complex unhandled");
    } else if constexpr (TC == Fortran::lower::CharacterCat) {
      return genCharLit<KIND>(con.GetScalarValue().value(), con.LEN());
    } else {
      llvm_unreachable("unhandled constant");
    }
  }

  template <Fortran::common::TypeCategory TC>
  mlir::Value genval(
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
  mlir::Value genval(const Fortran::evaluate::ArrayConstructor<A> &) {
    TODO();
  }
  mlir::Value gen(const Fortran::evaluate::ComplexPart &) { TODO(); }
  mlir::Value gendef(const Fortran::evaluate::ComplexPart &cp) {
    return gen(cp);
  }
  mlir::Value genval(const Fortran::evaluate::ComplexPart &) { TODO(); }

  mlir::Value gen(const Fortran::evaluate::Substring &s) {
    // Get base string
    auto baseString = std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::DataRef &x) { return gen(x); },
            [&](const Fortran::evaluate::StaticDataObject::Pointer &)
                -> mlir::Value { TODO(); },
        },
        s.parent());
    llvm::SmallVector<mlir::Value, 2> bounds;
    bounds.push_back(genval(s.lower()));
    if (auto upperBound{s.upper()}) {
      bounds.push_back(genval(*upperBound));
    }
    return builder.createSubstring(baseString, bounds);
  }

  mlir::Value gendef(const Fortran::evaluate::Substring &ss) { return gen(ss); }
  mlir::Value genval(const Fortran::evaluate::Substring &ss) { return gen(ss); }
  mlir::Value genval(const Fortran::evaluate::Triplet &trip) { TODO(); }

  mlir::Value genval(const Fortran::evaluate::Subscript &subs) {
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::IndirectSubscriptIntegerExpr &x) {
              return genval(x.value());
            },
            [&](const Fortran::evaluate::Triplet &x) { return genval(x); },
        },
        subs.u);
  }

  mlir::Value gen(const Fortran::evaluate::DataRef &dref) {
    return std::visit([&](const auto &x) { return gen(x); }, dref.u);
  }
  mlir::Value gendef(const Fortran::evaluate::DataRef &dref) {
    return gen(dref);
  }
  mlir::Value genval(const Fortran::evaluate::DataRef &dref) {
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
  mlir::Value gen(const Fortran::evaluate::Component &cmpt) {
    std::list<const Fortran::evaluate::Component *> list;
    auto *base = reverseComponents(cmpt, list);
    llvm::SmallVector<mlir::Value, 2> coorArgs;
    auto obj = gen(*base);
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

  mlir::Value gendef(const Fortran::evaluate::Component &cmpt) {
    return gen(cmpt);
  }
  mlir::Value genval(const Fortran::evaluate::Component &cmpt) {
    return builder.create<fir::LoadOp>(getLoc(), gen(cmpt));
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
  mlir::Value genval(const Fortran::semantics::Bound &bound) {
    if (bound.isExplicit()) {
      auto sub = bound.GetExplicit();
      if (sub.has_value())
        return genval(*sub);
      return genIntegerConstant<8>(builder.getContext(), 1);
    }
    TODO();
  }

  mlir::Value genArrayRefComponent(const Fortran::evaluate::ArrayRef &aref) {
    mlir::Value base = gen(aref.base().GetComponent());
    llvm::SmallVector<mlir::Value, 8> args;
    for (auto &subsc : aref.subscript())
      args.push_back(genval(subsc));
    auto ty = genSubType(base.getType(), args.size());
    ty = fir::ReferenceType::get(ty);
    return builder.create<fir::CoordinateOp>(getLoc(), ty, base, args);
  }

  // Return the coordinate of the array reference
  mlir::Value gen(const Fortran::evaluate::ArrayRef &aref) {
    if (aref.base().IsSymbol()) {
      auto &symbol = aref.base().GetFirstSymbol();
      mlir::Value base = gen(symbol);
      auto &shape =
          symbol.get<Fortran::semantics::ObjectEntityDetails>().shape();
      unsigned i = 0;
      llvm::SmallVector<mlir::Value, 8> args;
      for (auto &subsc : aref.subscript()) {
        auto val = genval(subsc);
        auto adj = genval(shape[i++].lbound());
        auto ty = val.getType();
        args.push_back(builder.create<mlir::SubIOp>(getLoc(), ty, val, adj));
      }
      auto ty = genSubType(base.getType(), args.size());
      ty = fir::ReferenceType::get(ty);
      return builder.create<fir::CoordinateOp>(getLoc(), ty, base, args);
    }
    return genArrayRefComponent(aref);
  }

  mlir::Value gendef(const Fortran::evaluate::ArrayRef &aref) {
    return gen(aref);
  }

  mlir::Value genval(const Fortran::evaluate::ArrayRef &aref) {
    return builder.create<fir::LoadOp>(getLoc(), gen(aref));
  }

  // Return a coordinate of the coarray reference. This is necessary as a
  // Component may have a CoarrayRef as its base coordinate.
  mlir::Value gen(const Fortran::evaluate::CoarrayRef &coref) {
    // FIXME: need to visit the cosubscripts...
    // return gen(coref.base());
    TODO();
  }
  mlir::Value gendef(const Fortran::evaluate::CoarrayRef &coref) {
    return gen(coref);
  }
  mlir::Value genval(const Fortran::evaluate::CoarrayRef &coref) {
    return builder.create<fir::LoadOp>(getLoc(), gen(coref));
  }

  template <typename A>
  mlir::Value gen(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return gen(x); }, des.u);
  }
  template <typename A>
  mlir::Value gendef(const Fortran::evaluate::Designator<A> &des) {
    return gen(des);
  }
  template <typename A>
  mlir::Value genval(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return genval(x); }, des.u);
  }

  // call a function
  template <typename A>
  mlir::Value gen(const Fortran::evaluate::FunctionRef<A> &funRef) {
    TODO();
  }
  template <typename A>
  mlir::Value gendef(const Fortran::evaluate::FunctionRef<A> &funRef) {
    return gen(funRef);
  }
  template <typename A>
  mlir::Value genval(const Fortran::evaluate::FunctionRef<A> &funRef) {
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
    auto restorer = Fortran::common::ScopedSet(genLogicalAsI1, false);
    for (const auto &arg : procRef.arguments()) {
      if (auto *expr = Fortran::evaluate::UnwrapExpr<
              Fortran::evaluate::Expr<Fortran::evaluate::SomeType>>(arg)) {
        operands.push_back(genval(*expr));
      } else {
        operands.push_back(nullptr); // optional
      }
    }
    // Let the intrinsic library lower the intrinsic procedure call
    llvm::StringRef name{intrinsic.name};
    return intrinsics.genval(getLoc(), builder, name, resultType[0], operands);
  }

  mlir::Value genProcedureRef(const Fortran::evaluate::ProcedureRef procRef,
                              mlir::ArrayRef<mlir::Type> resultType) {
    if (const auto *intrinsic{procRef.proc().GetSpecificIntrinsic()}) {
      return genIntrinsicRef(procRef, *intrinsic, resultType[0]);
    }
    // Implicit interface implementation only
    // TODO: Explicit interface, we need to use Characterize here,
    // evaluate::IntrinsicProcTable is required to use it.
    llvm::SmallVector<mlir::Type, 2> argTypes;
    llvm::SmallVector<mlir::Value, 2> operands;
    // Logical arguments of user functions must be lowered to `fir.logical`
    // and not `i1`.
    auto restorer = Fortran::common::ScopedSet(genLogicalAsI1, false);
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
        argTypes.push_back(argRef.getType());
        operands.push_back(argRef);
      } else {
        // create a temp to store the expression value
        auto val = genval(*expr);
        auto addr = builder.createTemporary(getLoc(), val.getType());
        builder.create<fir::StoreOp>(getLoc(), val, addr);
        argTypes.push_back(addr.getType());
        operands.push_back(addr);
      }
    }
    mlir::FunctionType funTy =
        mlir::FunctionType::get(argTypes, resultType, builder.getContext());
    auto funName = applyNameMangling(procRef.proc());
    getFunction(funName, funTy);
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
  mlir::Value
  genval(const Fortran::evaluate::FunctionRef<Fortran::evaluate::Type<TC, KIND>>
             &funRef) {
    llvm::SmallVector<mlir::Type, 1> resTy;
    resTy.push_back(converter.genType(TC, KIND));
    return genProcedureRef(funRef, resTy);
  }

  template <typename A>
  mlir::Value gen(const Fortran::evaluate::Expr<A> &exp) {
    // must be a designator or function-reference (R902)
    return std::visit([&](const auto &e) { return gendef(e); }, exp.u);
  }
  template <typename A>
  mlir::Value gendef(const Fortran::evaluate::Expr<A> &exp) {
    return gen(exp);
  }
  template <typename A>
  mlir::Value genval(const Fortran::evaluate::Expr<A> &exp) {
    return std::visit([&](const auto &e) { return genval(e); }, exp.u);
  }

  template <int KIND>
  mlir::Value
  genval(const Fortran::evaluate::Expr<
         Fortran::evaluate::Type<Fortran::lower::LogicalCat, KIND>> &exp) {
    auto result = std::visit([&](const auto &e) { return genval(e); }, exp.u);
    // Handle the `i1` to `fir.logical` conversions as needed.
    if (result) {
      mlir::Type type = result.getType();
      if (type.isa<fir::LogicalType>()) {
        if (genLogicalAsI1)
          result = builder.create<fir::ConvertOp>(getLoc(), builder.getI1Type(),
                                                  result);
      } else if (type.isa<mlir::IntegerType>()) {
        if (!genLogicalAsI1) {
          auto firLogicalType =
              converter.genType(Fortran::lower::LogicalCat, KIND);
          result =
              builder.create<fir::ConvertOp>(getLoc(), firLogicalType, result);
        }
      } else if (auto seqType{type.dyn_cast_or_null<fir::SequenceType>()}) {
        // TODO: Conversions at array level should probably be avoided.
        // This depends on how array expressions will be lowered.
        llvm_unreachable("logical array loads not yet implemented");
      } else {
        llvm_unreachable("unexpected logical type in expression");
      }
    }
    return result;
  }

  template <typename A>
  mlir::Value gendef(const A &) {
    llvm_unreachable("expression error");
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

public:
  explicit ExprLowering(mlir::Location loc,
                        Fortran::lower::AbstractConverter &converter,
                        const Fortran::lower::SomeExpr &vop,
                        Fortran::lower::SymMap &map,
                        const Fortran::lower::IntrinsicLibrary &intr,
                        bool logicalAsI1 = false)
      : location{loc}, converter{converter},
        builder{converter.getFirOpBuilder()}, expr{vop}, symMap{map},
        intrinsics{intr}, genLogicalAsI1{logicalAsI1} {}

  /// Lower the expression `expr` into MLIR standard dialect
  mlir::Value gen() { return gen(expr); }
  mlir::Value genval() { return genval(expr); }
};

} // namespace

mlir::Value Fortran::lower::createSomeExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap,
    const Fortran::lower::IntrinsicLibrary &intrinsics) {
  return ExprLowering{loc, converter, expr, symMap, intrinsics, false}.genval();
}

mlir::Value Fortran::lower::createI1LogicalExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap,
    const Fortran::lower::IntrinsicLibrary &intrinsics) {
  return ExprLowering{loc, converter, expr, symMap, intrinsics, true}.genval();
}

mlir::Value Fortran::lower::createSomeAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    Fortran::lower::SymMap &symMap,
    const Fortran::lower::IntrinsicLibrary &intrinsics) {
  return ExprLowering{loc, converter, expr, symMap, intrinsics}.gen();
}
