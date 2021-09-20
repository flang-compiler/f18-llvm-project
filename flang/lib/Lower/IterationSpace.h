//===-- IterationSpace.h ----------------------------------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_ITERATIONSPACE_H
#define FORTRAN_LOWER_ITERATIONSPACE_H

#include "StatementContext.h"
#include "flang/Evaluate/tools.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"

namespace llvm {
class raw_ostream;
}

namespace Fortran {
namespace evaluate {
struct SomeType;
template <typename>
class Expr;
} // namespace evaluate

namespace lower {

using MaskAddrAndShape = std::pair<mlir::Value, mlir::Value>;
using FrontEndExpr = const evaluate::Expr<evaluate::SomeType> *;
using FrontEndSymbol = const semantics::Symbol *;

class AbstractConverter;

template <typename A>
class StackableConstructExpr {
public:
  bool empty() const { return stack.empty(); }

  void growStack() {
    if (empty())
      stmtCtx.reset();
    stack.push_back(A{});
  }

  void shrinkStack() {
    assert(!empty());
    stack.pop_back();
    if (empty())
      stmtCtx.finalize();
  }

  /// Bind a front-end expression to a value and shape.
  void bind(FrontEndExpr e, mlir::Value v, mlir::Value shape) {
    vmap.try_emplace(e, v, shape);
  }
  void bind(FrontEndExpr e, const MaskAddrAndShape &p) { vmap.insert({e, p}); }

  /// Get the value bound to the front-end expression, `e`.
  mlir::Value getBinding(FrontEndExpr e) const {
    return getBindingWithShape(e).first;
  }

  /// Get the value and shape bound to the front-end expression, `e`.
  MaskAddrAndShape getBindingWithShape(FrontEndExpr e) const {
    assert(vmap.count(e) && "key not already in map");
    return vmap.lookup(e);
  }

  /// Has the front-end expression, `e`, been lowered and bound?
  bool isLowered(FrontEndExpr e) const { return vmap.count(e); }

  /// Replace the binding of front-end expression `e` with a new value and
  /// shape pair.
  void replaceBinding(FrontEndExpr e, mlir::Value v, mlir::Value shape) {
    vmap.erase(e);
    vmap.try_emplace(e, v, shape);
  }

  StatementContext &stmtContext() { return stmtCtx; }

protected:
  // The stack for the construct information.
  llvm::SmallVector<A> stack;

  // Map each mask expression back to the temporary holding the initial
  // evaluation results.
  llvm::DenseMap<FrontEndExpr, MaskAddrAndShape> vmap;

  // Inflate the statement context for the entire construct. We have to cache
  // the mask expression results, which are always evaluated first, across the
  // entire construct.
  StatementContext stmtCtx;
};

class ImplicitIterSpace;
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const ImplicitIterSpace &);

/// All array expressions have an implicit iteration space, which is isomorphic
/// to the shape of the base array that facilitates the expression having a
/// non-zero rank. This implied iteration space may be conditionalized
/// (disjunctively) with an if-elseif-else like structure, specifically
/// Fortran's WHERE construct.
///
/// This class is used in the bridge to collect the expressions from the
/// front end (the WHERE construct mask expressions), forward them for lowering
/// as array expressions in an "evaluate once" (copy-in, copy-out) semantics.
/// See 10.2.3.2p3, 10.2.3.2p13, etc.
class ImplicitIterSpace
    : public StackableConstructExpr<llvm::SmallVector<FrontEndExpr>> {
public:
  using FrontEndMaskExpr = FrontEndExpr;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const ImplicitIterSpace &);

  LLVM_DUMP_METHOD void dump() const;

  void append(FrontEndMaskExpr e) {
    assert(!empty());
    getMasks().back().push_back(e);
  }

  llvm::SmallVector<FrontEndMaskExpr> getExprs() const {
    auto maskList = getMasks()[0];
    for (size_t i = 1, d = getMasks().size(); i < d; ++i)
      maskList.append(getMasks()[i].begin(), getMasks()[i].end());
    return maskList;
  }

  /// Add a variable binding, `var`, along with its shape for the mask
  /// expression `exp`.
  void addMaskVariable(FrontEndExpr exp, mlir::Value var, mlir::Value shape) {
    maskVarMap.try_emplace(exp, std::make_pair(var, shape));
  }

  /// Lookup the variable corresponding to the temporary buffer that contains
  /// the mask array expression results.
  mlir::Value lookupMaskVariable(FrontEndExpr exp) {
    return maskVarMap.lookup(exp).first;
  }

  /// Lookup the variable containing the shape vector for the mask array
  /// expression results.
  mlir::Value lookupMaskShapeBuffer(FrontEndExpr exp) {
    return maskVarMap.lookup(exp).second;
  }

  // Stack of WHERE constructs, each building a list of mask expressions.
  llvm::SmallVector<llvm::SmallVector<FrontEndMaskExpr>> &getMasks() {
    return stack;
  }
  const llvm::SmallVector<llvm::SmallVector<FrontEndMaskExpr>> &
  getMasks() const {
    return stack;
  }

private:
  llvm::DenseMap<FrontEndExpr, std::pair<mlir::Value, mlir::Value>> maskVarMap;
};

class ExplicitIterSpace;
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const ExplicitIterSpace &);

/// Create all the array_load ops for the explicit iteration space context. The
/// nest of FORALLs must have been analyzed a priori.
void createArrayLoads(AbstractConverter &converter, ExplicitIterSpace &esp,
                      SymMap &symMap);

/// Create the array_merge_store ops after the explicit iteration space context
/// is conmpleted.
void createArrayMergeStores(AbstractConverter &converter,
                            ExplicitIterSpace &esp);

/// Fortran also allows arrays to be evaluated under constructs which allow the
/// user to explicitly specify the iteration space using concurrent-control
/// expressions. These constructs allow the user to define both an iteration
/// space and explicit access vectors on arrays. These need not be isomorphic.
/// The explicit iteration spaces may be conditionalized (conjunctively) with an
/// "and" structure and may be found in FORALL (and DO CONCURRENT) constructs.
///
/// This class is used in the bridge to collect a stack of lists of
/// concurrent-control expressions to be used to generate the iteration space
/// and associated masks (if any) for a set of nested FORALL constructs around
/// assignment and WHERE constructs.
class ExplicitIterSpace {
public:
  using IterSpaceDim =
      std::tuple<FrontEndSymbol, FrontEndExpr, FrontEndExpr, FrontEndExpr>;
  using ConcurrentSpec =
      std::pair<llvm::SmallVector<IterSpaceDim>, FrontEndExpr>;
  using ArrayBases = std::variant<FrontEndSymbol, const evaluate::Component *,
                                  const evaluate::ArrayRef *>;

  friend void createArrayLoads(AbstractConverter &converter,
                               ExplicitIterSpace &esp, SymMap &symMap);
  friend void createArrayMergeStores(AbstractConverter &converter,
                                     ExplicitIterSpace &esp);

  /// Is a FORALL context presently active?
  /// If we are lowering constructs/statements nested within a FORALL, then a
  /// FORALL context is active.
  bool isActive() const { return forallContextOpen != 0; }

  /// Get the statement context.
  StatementContext &stmtContext() { return stmtCtx; }

  //===--------------------------------------------------------------------===//
  // Analysis support
  //===--------------------------------------------------------------------===//

  /// Open a new construct. The analysis phase starts here.
  void pushLevel();

  /// Close the construct.
  void popLevel();

  /// Add new concurrent header control variable symbol.
  void addSymbol(FrontEndSymbol sym);

  /// Collect array bases from the expression, `x`.
  void exprBase(FrontEndExpr x, bool lhs);

  /// Called at the end of a assignment statement.
  void endAssign();

  /// Return all the active control variables on the stack.
  llvm::SmallVector<FrontEndSymbol> collectAllSymbols();

  //===--------------------------------------------------------------------===//
  // Code gen support
  //===--------------------------------------------------------------------===//

  /// Enter a FORALL context.
  void enter() { forallContextOpen++; }

  /// Leave a FORALL context.
  void leave();

  void pushLoopNest(std::function<void()> lambda) {
    ccLoopNest.push_back(lambda);
  }

  /// Get the inner arguments that correspond to the output arrays.
  mlir::ValueRange getInnerArgs() const { return innerArgs; }

  /// Set the inner arguments for the next loop level.
  void setInnerArgs(llvm::ArrayRef<mlir::BlockArgument> args) {
    innerArgs.clear();
    for (auto &arg : args)
      innerArgs.push_back(arg);
  }

  void setOuterLoop(fir::DoLoopOp loop) {
    if (!outerLoop.hasValue())
      outerLoop = loop;
  }

  void setInnerArg(size_t offset, mlir::Value val) {
    assert(offset < innerArgs.size());
    innerArgs[offset] = val;
  }

  /// Get the types of the output arrays.
  llvm::SmallVector<mlir::Type> innerArgTypes() const {
    llvm::SmallVector<mlir::Type> result;
    for (auto &arg : innerArgs)
      result.push_back(arg.getType());
    return result;
  }

  /// Create a binding between an Ev::Expr node pointer and a fir::array_load
  /// op. This bindings will be used when generating the IR.
  void bindLoad(const ArrayBases &base, fir::ArrayLoadOp load);

  template <typename A>
  fir::ArrayLoadOp findBinding(const A *base) {
    using T = std::remove_cv_t<std::remove_pointer_t<decltype(base)>>;
    return loadBindings.lookup(static_cast<void *>(const_cast<T *>(base)));
  }
  fir::ArrayLoadOp findBinding(const ArrayBases &base) {
    return std::visit([&](const auto *p) { return findBinding(p); }, base);
  }

  /// `load` must be a LHS array_load. Returns `llvm::None` on error.
  llvm::Optional<size_t> findArgPosition(fir::ArrayLoadOp load);

  bool isLHS(fir::ArrayLoadOp load) { return findArgPosition(load).hasValue(); }

  /// `load` must be a LHS array_load. Determine the threaded inner argument
  /// corresponding to this load.
  mlir::Value findArgumentOfLoad(fir::ArrayLoadOp load) {
    if (auto opt = findArgPosition(load))
      return innerArgs[*opt];
    llvm_unreachable("array load argument not found");
  }

  size_t argPosition(mlir::Value arg) {
    for (auto i : llvm::enumerate(innerArgs))
      if (arg == i.value())
        return i.index();
    llvm_unreachable("inner argument value was not found");
  }

  llvm::Optional<fir::ArrayLoadOp> getLhsLoad(size_t i) {
    assert(i < lhsBases.size());
    if (lhsBases[counter].hasValue())
      return findBinding(lhsBases[counter].getValue());
    return llvm::None;
  }

  /// Return the outermost loop in this FORALL nest.
  fir::DoLoopOp getOuterLoop() {
    assert(outerLoop.hasValue());
    return outerLoop.getValue();
  }

  /// Return the statement context for the entire, outermost FORALL construct.
  StatementContext &outermostContext() { return outerContext; }

  /// Generate the explicit loop nest.
  void genLoopNest() {
    for (auto &lambda : ccLoopNest)
      lambda();
  }

  /// Clear the array_load bindings.
  void resetBindings() { loadBindings.clear(); }

  /// Get the current counter value.
  std::size_t getCounter() const { return counter; }

  /// Increment the counter value to the next assignment statement.
  void incrementCounter() { counter++; }

  bool isOutermostForall() const {
    assert(forallContextOpen);
    return forallContextOpen == 1;
  }

  void attachLoopCleanup(std::function<void(fir::FirOpBuilder &builder)> fn) {
    if (!loopCleanup.hasValue()) {
      loopCleanup = fn;
      return;
    }
    auto oldFn = loopCleanup.getValue();
    loopCleanup = [=](fir::FirOpBuilder &builder) {
      oldFn(builder);
      fn(builder);
    };
  }

  // LLVM standard dump method.
  LLVM_DUMP_METHOD void dump() const;

  // Pretty-print.
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const ExplicitIterSpace &);

  /// Finalize the current body statement context.
  void finalizeContext() {
    stmtCtx.finalize();
    stmtCtx.reset();
  }

private:
  /// Cleanup the analysis results.
  void conditionalCleanup();

  StatementContext outerContext;

  // A stack of lists of front-end symbols.
  llvm::SmallVector<llvm::SmallVector<FrontEndSymbol>> symbolStack;
  llvm::SmallVector<llvm::Optional<ArrayBases>> lhsBases;
  llvm::SmallVector<llvm::SmallVector<ArrayBases>> rhsBases;
  llvm::DenseMap<void *, fir::ArrayLoadOp> loadBindings;

  // Stack of lambdas to create the loop nest.
  llvm::SmallVector<std::function<void()>> ccLoopNest;

  // Assignment statement context (inside the loop nest).
  StatementContext stmtCtx;
  llvm::SmallVector<mlir::Value> innerArgs;
  llvm::Optional<fir::DoLoopOp> outerLoop;
  llvm::Optional<std::function<void(fir::FirOpBuilder &)>> loopCleanup;
  std::size_t forallContextOpen = 0;
  std::size_t counter = 0;
};

/// Is there a Symbol in common between the concurrent header set and the set
/// of symbols in the expression?
template <typename A>
bool symbolSetsIntersect(llvm::ArrayRef<FrontEndSymbol> ctrlSet,
                         const A &exprSyms) {
  for (const auto &sym : exprSyms)
    if (std::find(ctrlSet.begin(), ctrlSet.end(), &sym.get()) != ctrlSet.end())
      return true;
  return false;
}

/// Determine if the subscript expression symbols from an Ev::ArrayRef
/// intersects with the set of concurrent control symbols, `ctrlSet`.
template <typename A>
bool symbolsIntersectSubscripts(llvm::ArrayRef<FrontEndSymbol> ctrlSet,
                                const A &subscripts) {
  for (auto &sub : subscripts) {
    if (const auto *expr =
            std::get_if<evaluate::IndirectSubscriptIntegerExpr>(&sub.u))
      if (symbolSetsIntersect(ctrlSet, evaluate::CollectSymbols(expr->value())))
        return true;
  }
  return false;
}

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_ITERATIONSPACE_H
