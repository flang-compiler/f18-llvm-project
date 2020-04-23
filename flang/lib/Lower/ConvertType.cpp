//===-- ConvertType.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertType.h"
#include "../../runtime/io-api.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/Utils.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"

namespace {

template <typename A>
bool isConstant(const Fortran::evaluate::Expr<A> &e) {
  return Fortran::evaluate::IsConstantExpr(Fortran::lower::SomeExpr{e});
}

template <typename A>
int64_t toConstant(const Fortran::evaluate::Expr<A> &e) {
  auto opt = Fortran::evaluate::ToInt64(e);
  assert(opt.has_value() && "expression didn't resolve to a constant");
  return opt.value();
}

#undef TODO
#define TODO()                                                                 \
  assert(false && "not yet implemented");                                      \
  return {}

// one argument template, must be specialized
template <Fortran::common::TypeCategory TC>
mlir::Type genFIRType(mlir::MLIRContext *, int) {
  return {};
}

// two argument template
template <Fortran::common::TypeCategory TC, int KIND>
mlir::Type genFIRType(mlir::MLIRContext *context) {
  if constexpr (TC == Fortran::lower::IntegerCat) {
    auto bits{Fortran::evaluate::Type<Fortran::lower::IntegerCat,
                                      KIND>::Scalar::bits};
    return mlir::IntegerType::get(bits, context);
  } else if constexpr (TC == Fortran::lower::LogicalCat ||
                       TC == Fortran::lower::CharacterCat ||
                       TC == Fortran::lower::ComplexCat) {
    return genFIRType<TC>(context, KIND);
  } else {
    return {};
  }
}

template <>
mlir::Type genFIRType<Fortran::lower::RealCat, 2>(mlir::MLIRContext *context) {
  return mlir::FloatType::getF16(context);
}

template <>
mlir::Type genFIRType<Fortran::lower::RealCat, 3>(mlir::MLIRContext *context) {
  return mlir::FloatType::getBF16(context);
}

template <>
mlir::Type genFIRType<Fortran::lower::RealCat, 4>(mlir::MLIRContext *context) {
  return mlir::FloatType::getF32(context);
}

template <>
mlir::Type genFIRType<Fortran::lower::RealCat, 8>(mlir::MLIRContext *context) {
  return mlir::FloatType::getF64(context);
}

template <>
mlir::Type genFIRType<Fortran::lower::RealCat, 10>(mlir::MLIRContext *context) {
  return fir::RealType::get(context, 10);
}

template <>
mlir::Type genFIRType<Fortran::lower::RealCat, 16>(mlir::MLIRContext *context) {
  return fir::RealType::get(context, 16);
}

template <>
mlir::Type genFIRType<Fortran::lower::RealCat>(mlir::MLIRContext *context,
                                               int kind) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(Fortran::lower::RealCat,
                                                    kind)) {
    switch (kind) {
    case 2:
      return genFIRType<Fortran::lower::RealCat, 2>(context);
    case 3:
      return genFIRType<Fortran::lower::RealCat, 3>(context);
    case 4:
      return genFIRType<Fortran::lower::RealCat, 4>(context);
    case 8:
      return genFIRType<Fortran::lower::RealCat, 8>(context);
    case 10:
      return genFIRType<Fortran::lower::RealCat, 10>(context);
    case 16:
      return genFIRType<Fortran::lower::RealCat, 16>(context);
    }
    assert(false && "type translation not implemented");
  }
  return {};
}

template <>
mlir::Type genFIRType<Fortran::lower::IntegerCat>(mlir::MLIRContext *context,
                                                  int kind) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(Fortran::lower::IntegerCat,
                                                    kind)) {
    switch (kind) {
    case 1:
      return genFIRType<Fortran::lower::IntegerCat, 1>(context);
    case 2:
      return genFIRType<Fortran::lower::IntegerCat, 2>(context);
    case 4:
      return genFIRType<Fortran::lower::IntegerCat, 4>(context);
    case 8:
      return genFIRType<Fortran::lower::IntegerCat, 8>(context);
    case 16:
      return genFIRType<Fortran::lower::IntegerCat, 16>(context);
    }
    assert(false && "type translation not implemented");
  }
  return {};
}

template <>
mlir::Type genFIRType<Fortran::lower::LogicalCat>(mlir::MLIRContext *context,
                                                  int KIND) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(Fortran::lower::LogicalCat,
                                                    KIND))
    return fir::LogicalType::get(context, KIND);
  return {};
}

template <>
mlir::Type genFIRType<Fortran::lower::CharacterCat>(mlir::MLIRContext *context,
                                                    int KIND) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(
          Fortran::lower::CharacterCat, KIND))
    return fir::CharacterType::get(context, KIND);
  return {};
}

template <>
mlir::Type genFIRType<Fortran::lower::ComplexCat>(mlir::MLIRContext *context,
                                                  int KIND) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(Fortran::lower::ComplexCat,
                                                    KIND))
    return fir::CplxType::get(context, KIND);
  return {};
}

/// Recover the type of an Fortran::evaluate::Expr<T> and convert it to an
/// mlir::Type. The type returned can be a MLIR standard or FIR type.
class TypeBuilder {
  mlir::MLIRContext *context;
  const Fortran::common::IntrinsicTypeDefaultKinds &defaults;

  template <Fortran::common::TypeCategory TC>
  int defaultKind() {
    return defaultKind(TC);
  }
  int defaultKind(Fortran::common::TypeCategory TC) {
    return defaults.GetDefaultKind(TC);
  }

  mlir::InFlightDiagnostic emitError(const llvm::Twine &message) {
    return mlir::emitError(mlir::UnknownLoc::get(context), message);
  }

  mlir::InFlightDiagnostic emitWarning(const llvm::Twine &message) {
    return mlir::emitWarning(mlir::UnknownLoc::get(context), message);
  }

  fir::SequenceType::Shape seqShapeHelper(Fortran::semantics::SymbolRef symbol,
                                          fir::SequenceType::Shape &bounds) {
    auto &details = symbol->get<Fortran::semantics::ObjectEntityDetails>();
    const auto size = details.shape().size();
    for (auto &ss : details.shape()) {
      auto lb = ss.lbound();
      auto ub = ss.ubound();
      if (lb.isAssumed() && ub.isAssumed() && size == 1)
        return {};
      if (lb.isExplicit() && ub.isExplicit()) {
        auto &lbv = lb.GetExplicit();
        auto &ubv = ub.GetExplicit();
        if (lbv.has_value() && ubv.has_value() && isConstant(lbv.value()) &&
            isConstant(ubv.value())) {
          bounds.emplace_back(toConstant(ubv.value()) -
                              toConstant(lbv.value()) + 1);
        } else {
          bounds.emplace_back(fir::SequenceType::getUnknownExtent());
        }
      } else {
        bounds.emplace_back(fir::SequenceType::getUnknownExtent());
      }
    }
    return bounds;
  }

public:
  explicit TypeBuilder(
      mlir::MLIRContext *context,
      const Fortran::common::IntrinsicTypeDefaultKinds &defaults)
      : context{context}, defaults{defaults} {}

  // non-template, arguments are runtime values
  mlir::Type genFIRTy(Fortran::common::TypeCategory tc, int kind) {
    switch (tc) {
    case Fortran::lower::RealCat:
      return genFIRType<Fortran::lower::RealCat>(context, kind);
    case Fortran::lower::IntegerCat:
      return genFIRType<Fortran::lower::IntegerCat>(context, kind);
    case Fortran::lower::ComplexCat:
      return genFIRType<Fortran::lower::ComplexCat>(context, kind);
    case Fortran::lower::LogicalCat:
      return genFIRType<Fortran::lower::LogicalCat>(context, kind);
    case Fortran::lower::CharacterCat:
      return genFIRType<Fortran::lower::CharacterCat>(context, kind);
    default:
      break;
    }
    assert(false && "unhandled type category");
    return {};
  }

  // non-template, category is runtime values, kind is defaulted
  mlir::Type genFIRTy(Fortran::common::TypeCategory tc) {
    return genFIRTy(tc, defaultKind(tc));
  }

  mlir::Type gen(const Fortran::evaluate::ImpliedDoIndex &) {
    return genFIRType<Fortran::lower::IntegerCat>(
        context, defaultKind<Fortran::lower::IntegerCat>());
  }

  template <template <typename> typename A, Fortran::common::TypeCategory TC>
  mlir::Type gen(const A<Fortran::evaluate::SomeKind<TC>> &) {
    return genFIRType<TC>(context, defaultKind<TC>());
  }

  template <int KIND>
  mlir::Type gen(const Fortran::evaluate::TypeParamInquiry<KIND> &) {
    return genFIRType<Fortran::lower::IntegerCat, KIND>(context);
  }

  template <typename A>
  mlir::Type gen(const Fortran::evaluate::Relational<A> &) {
    return genFIRType<Fortran::lower::LogicalCat, 1>(context);
  }

  template <template <typename> typename A, Fortran::common::TypeCategory TC,
            int KIND>
  mlir::Type gen(const A<Fortran::evaluate::Type<TC, KIND>> &) {
    return genFIRType<TC, KIND>(context);
  }

  // breaks the conflict between A<Type<TC,KIND>> and Expr<B> deduction
  template <Fortran::common::TypeCategory TC, int KIND>
  mlir::Type
  gen(const Fortran::evaluate::Expr<Fortran::evaluate::Type<TC, KIND>> &) {
    return genFIRType<TC, KIND>(context);
  }

  template <typename A>
  mlir::Type genVariant(const A &variant) {
    return std::visit([&](const auto &x) { return gen(x); }, variant.u);
  }

  // breaks the conflict between A<SomeKind<TC>> and Expr<B> deduction
  template <Fortran::common::TypeCategory TC>
  mlir::Type
  gen(const Fortran::evaluate::Expr<Fortran::evaluate::SomeKind<TC>> &expr) {
    return genVariant(expr);
  }

  template <typename A>
  mlir::Type gen(const Fortran::evaluate::Expr<A> &expr) {
    return genVariant(expr);
  }

  mlir::Type gen(const Fortran::evaluate::DataRef &dref) {
    return genVariant(dref);
  }

  mlir::Type mkVoid() { return mlir::TupleType::get(context); }

  fir::SequenceType::Shape genSeqShape(Fortran::semantics::SymbolRef symbol) {
    assert(symbol->IsObjectArray());
    fir::SequenceType::Shape bounds;
    return seqShapeHelper(symbol, bounds);
  }

  fir::SequenceType::Shape genSeqShape(Fortran::semantics::SymbolRef symbol,
                                       fir::SequenceType::Extent charLen) {
    assert(symbol->IsObjectArray());
    fir::SequenceType::Shape bounds;
    bounds.push_back(charLen);
    return seqShapeHelper(symbol, bounds);
  }

  mlir::Type genDummyArgType(const Fortran::semantics::Symbol &dummy) {
    if (auto *type{dummy.GetType()}) {
      auto *tySpec{type->AsIntrinsic()};
      if (tySpec && tySpec->category() == Fortran::lower::CharacterCat) {
        auto kind = toConstant(tySpec->kind());
        return fir::BoxCharType::get(context, kind);
      }
    }
    if (Fortran::semantics::IsDescriptor(dummy)) {
      // FIXME: This should be the first case, but it seems to
      // fire at assumed length character on purpose which is
      // not what I expect.
      TODO();
    }
    return fir::ReferenceType::get(gen(dummy));
  }

  mlir::FunctionType genFunctionType(Fortran::semantics::SymbolRef symbol) {
    llvm::SmallVector<mlir::Type, 1> returnTys;
    llvm::SmallVector<mlir::Type, 4> inputTys;
    if (auto *proc =
            symbol->detailsIf<Fortran::semantics::SubprogramDetails>()) {
      if (proc->isFunction())
        returnTys.emplace_back(gen(proc->result()));
      else if (Fortran::semantics::HasAlternateReturns(symbol))
        returnTys.emplace_back(mlir::IndexType::get(context));
      for (auto *arg : proc->dummyArgs()) {
        // A nullptr arg is an alternate return label specifier; skip it.
        if (arg)
          inputTys.emplace_back(genDummyArgType(*arg));
      }
    } else if (symbol->detailsIf<Fortran::semantics::ProcEntityDetails>()) {
      // TODO Should probably use Fortran::evaluate::Characteristics for that.
      TODO();
    } else if (symbol->detailsIf<Fortran::semantics::MainProgramDetails>()) {
    } else {
      assert(false && "unexpected symbol details for function");
    }
    return mlir::FunctionType::get(inputTys, returnTys, context);
  }

  /// Type consing from a symbol. A symbol's type must be created from the type
  /// discovered by the front-end at runtime.
  mlir::Type gen(Fortran::semantics::SymbolRef symbol) {
    if (symbol->detailsIf<Fortran::semantics::SubprogramDetails>())
      return genFunctionType(symbol);
    mlir::Type returnTy;
    if (auto *type{symbol->GetType()}) {
      if (auto *tySpec{type->AsIntrinsic()}) {
        int kind = toConstant(tySpec->kind());
        switch (tySpec->category()) {
        case Fortran::lower::IntegerCat:
          returnTy = genFIRType<Fortran::lower::IntegerCat>(context, kind);
          break;
        case Fortran::lower::RealCat:
          returnTy = genFIRType<Fortran::lower::RealCat>(context, kind);
          break;
        case Fortran::lower::ComplexCat:
          returnTy = genFIRType<Fortran::lower::ComplexCat>(context, kind);
          break;
        case Fortran::lower::CharacterCat:
          returnTy = genFIRType<Fortran::lower::CharacterCat>(context, kind);
          break;
        case Fortran::lower::LogicalCat:
          returnTy = genFIRType<Fortran::lower::LogicalCat>(context, kind);
          break;
        default:
          emitError("symbol has unknown intrinsic type");
          return {};
        }
      } else if (auto *tySpec = type->AsDerived()) {
        std::vector<std::pair<std::string, mlir::Type>> ps;
        std::vector<std::pair<std::string, mlir::Type>> cs;
        auto &symbol = tySpec->typeSymbol();
        // FIXME: don't want to recurse forever here, but this won't happen
        // since we don't know the components at this time
        auto rec = fir::RecordType::get(context, toStringRef(symbol.name()));
        auto &details = symbol.get<Fortran::semantics::DerivedTypeDetails>();
        for (auto &param : details.paramDecls()) {
          auto &p{*param};
          ps.push_back(std::pair{p.name().ToString(), gen(p)});
        }
#if 0
        // this functionality is missing in the front-end
        for (auto &comp : details.componentDecls()) {
          auto &c{*comp};
          cs.push_back(std::pair{c.name().ToString(), gen(c)});
        }
#else
        emitWarning("the front-end returns symbols of derived type that have "
                    "components that are simple names and not symbols, so "
                    "cannot construct type " +
                    toStringRef(symbol.name()));
#endif
        rec.finalize(ps, cs);
        returnTy = rec;
      } else {
        emitError("symbol's type must have a type spec");
        return {};
      }
    } else {
      emitError("symbol must have a type");
      return {};
    }
    if (symbol->IsObjectArray()) {
      if (symbol->GetType()->category() ==
          Fortran::semantics::DeclTypeSpec::Character) {
        auto charLen = fir::SequenceType::getUnknownExtent();
        const auto &lenParam = symbol->GetType()->characterTypeSpec().length();
        if (auto expr = lenParam.GetExplicit()) {
          auto len = Fortran::evaluate::AsGenericExpr(std::move(*expr));
          auto asInt = Fortran::evaluate::ToInt64(len);
          if (asInt)
            charLen = *asInt;
        }
        return fir::SequenceType::get(genSeqShape(symbol, charLen), returnTy);
      }
      return fir::SequenceType::get(genSeqShape(symbol), returnTy);
    }
    if (Fortran::semantics::IsPointer(*symbol)) {
      // FIXME: what about allocatable?
      return fir::ReferenceType::get(returnTy);
    }
    return returnTy;
  }

  fir::SequenceType::Shape trivialShape(int size) {
    fir::SequenceType::Shape bounds;
    bounds.emplace_back(size);
    return bounds;
  }

  // some sequence of `n` bytes
  mlir::Type gen(const Fortran::evaluate::StaticDataObject::Pointer &ptr) {
    mlir::Type byteTy{mlir::IntegerType::get(8, context)};
    return fir::SequenceType::get(trivialShape(ptr->itemBytes()), byteTy);
  }

  mlir::Type gen(const Fortran::evaluate::Substring &ss) {
    return genVariant(ss.GetBaseObject());
  }

  mlir::Type genTypelessPtr() { return fir::ReferenceType::get(mkVoid()); }

  mlir::Type gen(const Fortran::evaluate::NullPointer &) {
    return genTypelessPtr();
  }
  mlir::Type gen(const Fortran::evaluate::ProcedureRef &) {
    return genTypelessPtr();
  }
  mlir::Type gen(const Fortran::evaluate::ProcedureDesignator &) {
    return genTypelessPtr();
  }
  mlir::Type gen(const Fortran::evaluate::BOZLiteralConstant &) {
    return genTypelessPtr();
  }

  mlir::Type gen(const Fortran::evaluate::ArrayRef &) { TODO(); }
  mlir::Type gen(const Fortran::evaluate::CoarrayRef &) { TODO(); }
  mlir::Type gen(const Fortran::evaluate::Component &) { TODO(); }
  mlir::Type gen(const Fortran::evaluate::ComplexPart &) { TODO(); }
  mlir::Type gen(const Fortran::evaluate::DescriptorInquiry &) { TODO(); }
  mlir::Type gen(const Fortran::evaluate::StructureConstructor &) { TODO(); }
};

} // namespace

mlir::Type Fortran::lower::getFIRType(
    mlir::MLIRContext *context,
    const Fortran::common::IntrinsicTypeDefaultKinds &defaults,
    Fortran::common::TypeCategory tc, int kind) {
  return TypeBuilder{context, defaults}.genFIRTy(tc, kind);
}

mlir::Type Fortran::lower::getFIRType(
    mlir::MLIRContext *context,
    const Fortran::common::IntrinsicTypeDefaultKinds &defaults,
    Fortran::common::TypeCategory tc) {
  return TypeBuilder{context, defaults}.genFIRTy(tc);
}

mlir::Type Fortran::lower::translateDataRefToFIRType(
    mlir::MLIRContext *context,
    const Fortran::common::IntrinsicTypeDefaultKinds &defaults,
    const Fortran::evaluate::DataRef &dataRef) {
  return TypeBuilder{context, defaults}.gen(dataRef);
}

// Builds the FIR type from an instance of SomeExpr
mlir::Type Fortran::lower::translateSomeExprToFIRType(
    mlir::MLIRContext *context,
    const Fortran::common::IntrinsicTypeDefaultKinds &defaults,
    const SomeExpr *expr) {
  return TypeBuilder{context, defaults}.gen(*expr);
}

// This entry point avoids gratuitously wrapping the Symbol instance in layers
// of Expr<T> that will then be immediately peeled back off and discarded.
mlir::Type Fortran::lower::translateSymbolToFIRType(
    mlir::MLIRContext *context,
    const Fortran::common::IntrinsicTypeDefaultKinds &defaults,
    const SymbolRef symbol) {
  return TypeBuilder{context, defaults}.gen(symbol);
}

mlir::FunctionType Fortran::lower::translateSymbolToFIRFunctionType(
    mlir::MLIRContext *context,
    const Fortran::common::IntrinsicTypeDefaultKinds &defaults,
    const SymbolRef symbol) {
  return TypeBuilder{context, defaults}.genFunctionType(symbol);
}

mlir::Type Fortran::lower::convertReal(mlir::MLIRContext *context, int kind) {
  return genFIRType<RealCat>(context, kind);
}

mlir::Type Fortran::lower::getSequenceRefType(mlir::Type refType) {
  auto type{refType.dyn_cast<fir::ReferenceType>()};
  assert(type && "expected a reference type");
  auto elementType{type.getEleTy()};
  fir::SequenceType::Shape shape{fir::SequenceType::getUnknownExtent()};
  return fir::ReferenceType::get(fir::SequenceType::get(shape, elementType));
}
