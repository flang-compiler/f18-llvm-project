//===-- OpenMP.cpp -- Open MP directive lowering --------------------------===//
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

#include "flang/Lower/OpenMP.h"
#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include <algorithm>

static const Fortran::parser::Name *
getDesignatorNameIfDataRef(const Fortran::parser::Designator &designator) {
  const auto *dataRef = std::get_if<Fortran::parser::DataRef>(&designator.u);
  return dataRef ? std::get_if<Fortran::parser::Name>(&dataRef->u) : nullptr;
}

template <typename T>
static void createPrivateVarSyms(Fortran::lower::AbstractConverter &converter,
                                 const T *clause) {
  Fortran::semantics::Symbol *sym = nullptr;
  const Fortran::parser::OmpObjectList &ompObjectList = clause->v;
  for (const Fortran::parser::OmpObject &ompObject : ompObjectList.v) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Designator &designator) {
              if (const Fortran::parser::Name *name =
                      getDesignatorNameIfDataRef(designator)) {
                sym = name->symbol;
              }
            },
            [&](const Fortran::parser::Name &name) { sym = name.symbol; }},
        ompObject.u);

    // Privatization for symbols which are pre-determined (like loop index
    // variables) happen separately, for everything else privatize here
    if constexpr (std::is_same_v<T, Fortran::parser::OmpClause::Firstprivate>) {
      converter.copyHostAssociateVar(*sym);
    } else {
      bool success = converter.createHostAssociateVarClone(*sym);
      (void)success;
      assert(success && "Privatization failed due to existing binding");
    }
  }
}

static void privatizeVars(Fortran::lower::AbstractConverter &converter,
                          const Fortran::parser::OmpClauseList &opClauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  auto insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());
  for (const Fortran::parser::OmpClause &clause : opClauseList.v) {
    if (const auto &privateClause =
            std::get_if<Fortran::parser::OmpClause::Private>(&clause.u)) {
      createPrivateVarSyms(converter, privateClause);
    } else if (const auto &firstPrivateClause =
                   std::get_if<Fortran::parser::OmpClause::Firstprivate>(
                       &clause.u)) {
      createPrivateVarSyms(converter, firstPrivateClause);
    }
  }
  firOpBuilder.restoreInsertionPoint(insPt);
}

static void genObjectList(const Fortran::parser::OmpObjectList &objectList,
                          Fortran::lower::AbstractConverter &converter,
                          llvm::SmallVectorImpl<Value> &operands) {
  auto addOperands = [&](Fortran::lower::SymbolRef sym) {
    const mlir::Value variable = converter.getSymbolAddress(sym);
    if (variable) {
      operands.push_back(variable);
    } else {
      if (const auto *details =
              sym->detailsIf<Fortran::semantics::HostAssocDetails>()) {
        operands.push_back(converter.getSymbolAddress(details->symbol()));
        converter.copySymbolBinding(details->symbol(), sym);
      }
    }
  };
  for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
    std::visit(Fortran::common::visitors{
                   [&](const Fortran::parser::Designator &designator) {
                     if (const Fortran::parser::Name *name =
                             getDesignatorNameIfDataRef(designator)) {
                       addOperands(*name->symbol);
                     }
                   },
                   [&](const Fortran::parser::Name &name) {
                     addOperands(*name.symbol);
                   }},
               ompObject.u);
  }
}

/// Create empty blocks for the current region.
/// These blocks replace blocks parented to an enclosing region.
void createEmptyRegionBlocks(
    fir::FirOpBuilder &firOpBuilder,
    std::list<Fortran::lower::pft::Evaluation> &evaluationList) {
  mlir::Region *region = &firOpBuilder.getRegion();
  for (Fortran::lower::pft::Evaluation &eval : evaluationList) {
    if (eval.block) {
      if (eval.block->empty()) {
        eval.block->erase();
        eval.block = firOpBuilder.createBlock(region);
      } else {
        mlir::Operation &terminatorOp = eval.block->back();
        (void)terminatorOp;
        assert((mlir::isa<mlir::omp::TerminatorOp>(terminatorOp) ||
                mlir::isa<mlir::omp::YieldOp>(terminatorOp)) &&
               "expected terminator op");
        // FIXME: Some subset of cases may need to insert a branch,
        // although this could be handled elsewhere.
        // if (?) {
        //   auto insertPt = firOpBuilder.saveInsertionPoint();
        //   firOpBuilder.setInsertionPointAfter(region->getParentOp());
        //   firOpBuilder.create<mlir::BranchOp>(
        //       terminatorOp.getLoc(), eval.block);
        //   firOpBuilder.restoreInsertionPoint(insertPt);
        // }
      }
    }
    if (!eval.isDirective() && eval.hasNestedEvaluations())
      createEmptyRegionBlocks(firOpBuilder, eval.getNestedEvaluations());
  }
}

static mlir::Type getLoopVarType(Fortran::lower::AbstractConverter &converter,
                                 std::size_t loopVarTypeSize) {
  // OpenMP runtime requires 32-bit or 64-bit loop variables.
  loopVarTypeSize = loopVarTypeSize * 8;
  if (loopVarTypeSize < 32) {
    loopVarTypeSize = 32;
  } else if (loopVarTypeSize == 128) {
    loopVarTypeSize = 64;
    mlir::emitWarning(converter.getCurrentLocation(),
                      "OpenMP loop iteration variable cannot have more than 64 "
                      "bits size and will be narrowed into 64 bits.");
  }
  assert((loopVarTypeSize == 32 || loopVarTypeSize == 64) &&
         "OpenMP loop iteration variable size must be transformed into 32-bit "
         "or 64-bit");
  return converter.getFirOpBuilder().getIntegerType(loopVarTypeSize);
}

template <typename Op>
static void createBodyOfOp(
    Op &op, Fortran::lower::AbstractConverter &converter, mlir::Location &loc,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OmpClauseList *clauses = nullptr,
    const llvm::SmallVector<const Fortran::semantics::Symbol *> &args = {},
    bool outerCombined = false) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  // If an argument for the region is provided then create the block with that
  // argument. Also update the symbol's address with the mlir argument value.
  // e.g. For loops the argument is the induction variable. And all further
  // uses of the induction variable should use this mlir value.
  mlir::Operation *storeOp = nullptr;
  if (args.size()) {
    std::size_t loopVarTypeSize = 0;
    for (const Fortran::semantics::Symbol *arg : args)
      loopVarTypeSize = std::max(loopVarTypeSize, arg->GetUltimate().size());
    mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
    llvm::SmallVector<Type> tiv;
    for (std::size_t i = 0; i < args.size(); i++)
      tiv.push_back(loopVarType);
    firOpBuilder.createBlock(&op.getRegion(), {}, tiv);
    int argIndex = 0;
    // The argument is not currently in memory, so make a temporary for the
    // argument, and store it there, then bind that location to the argument.
    for (const Fortran::semantics::Symbol *arg : args) {
      mlir::Value val =
          fir::getBase(op.getRegion().front().getArgument(argIndex));
      mlir::Value temp = firOpBuilder.createTemporary(
          loc, loopVarType,
          llvm::ArrayRef<mlir::NamedAttribute>{
              Fortran::lower::getAdaptToByRefAttr(firOpBuilder)});
      storeOp = firOpBuilder.create<fir::StoreOp>(loc, val, temp);
      // Always add the binding to the inner-most level of the symbol map.
      converter.bindSymbol(*arg, temp);
      argIndex++;
    }
  } else {
    firOpBuilder.createBlock(&op.getRegion());
  }
  // Set the insert for the terminator operation to go at the end of the
  // block - this is either empty or the block with the stores above,
  // the end of the block works for both.
  mlir::Block &block = op.getRegion().back();
  firOpBuilder.setInsertionPointToEnd(&block);

  // If it is an unstructured region and is not the outer region of a combined
  // construct, create empty blocks for all evaluations.
  if (eval.lowerAsUnstructured() && !outerCombined)
    createEmptyRegionBlocks(firOpBuilder, eval.getNestedEvaluations());

  // Insert the terminator.
  if constexpr (std::is_same_v<Op, omp::WsLoopOp>) {
    mlir::ValueRange results;
    firOpBuilder.create<mlir::omp::YieldOp>(loc, results);
  } else {
    firOpBuilder.create<mlir::omp::TerminatorOp>(loc);
  }

  // Reset the insert point to before the terminator.
  if (storeOp)
    firOpBuilder.setInsertionPointAfter(storeOp);
  else
    firOpBuilder.setInsertionPointToStart(&block);

  // Handle privatization. Do not privatize if this is the outer operation.
  if (clauses && !outerCombined)
    privatizeVars(converter, *clauses);
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPSimpleStandaloneConstruct
                       &simpleStandaloneConstruct) {
  const auto &directive =
      std::get<Fortran::parser::OmpSimpleStandaloneDirective>(
          simpleStandaloneConstruct.t);
  switch (directive.v) {
  default:
    break;
  case llvm::omp::Directive::OMPD_barrier:
    converter.getFirOpBuilder().create<mlir::omp::BarrierOp>(
        converter.getCurrentLocation());
    break;
  case llvm::omp::Directive::OMPD_taskwait:
    converter.getFirOpBuilder().create<mlir::omp::TaskwaitOp>(
        converter.getCurrentLocation());
    break;
  case llvm::omp::Directive::OMPD_taskyield:
    converter.getFirOpBuilder().create<mlir::omp::TaskyieldOp>(
        converter.getCurrentLocation());
    break;
  case llvm::omp::Directive::OMPD_target_enter_data:
    TODO(converter.getCurrentLocation(), "OMPD_target_enter_data");
  case llvm::omp::Directive::OMPD_target_exit_data:
    TODO(converter.getCurrentLocation(), "OMPD_target_exit_data");
  case llvm::omp::Directive::OMPD_target_update:
    TODO(converter.getCurrentLocation(), "OMPD_target_update");
  case llvm::omp::Directive::OMPD_ordered:
    TODO(converter.getCurrentLocation(), "OMPD_ordered");
  }
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPStandaloneConstruct &standaloneConstruct) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPSimpleStandaloneConstruct
                  &simpleStandaloneConstruct) {
            genOMP(converter, eval, simpleStandaloneConstruct);
          },
          [&](const Fortran::parser::OpenMPFlushConstruct &flushConstruct) {
            llvm::SmallVector<Value, 4> operandRange;
            if (const auto &ompObjectList =
                    std::get<std::optional<Fortran::parser::OmpObjectList>>(
                        flushConstruct.t))
              genObjectList(*ompObjectList, converter, operandRange);
            // FIXME: Add support for handling memory order clause. Adding
            // a TODO will invoke a crash, so commented it for now.
            // if (std::get<std::optional<
            //        std::list<Fortran::parser::OmpMemoryOrderClause>>>(
            //        flushConstruct.t))
            converter.getFirOpBuilder().create<mlir::omp::FlushOp>(
                converter.getCurrentLocation(), operandRange);
          },
          [&](const Fortran::parser::OpenMPCancelConstruct &cancelConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPCancelConstruct");
          },
          [&](const Fortran::parser::OpenMPCancellationPointConstruct
                  &cancellationPointConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPCancelConstruct");
          },
      },
      standaloneConstruct.u);
}

template <typename Directive, bool isCombined>
static void createParallelOp(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::pft::Evaluation &eval,
                             const Directive &directive) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  Fortran::lower::StatementContext stmtCtx;
  llvm::ArrayRef<mlir::Type> argTy;
  mlir::Value ifClauseOperand, numThreadsClauseOperand;
  llvm::SmallVector<Value, 4> privateClauseOperands, firstprivateClauseOperands,
      sharedClauseOperands, copyinClauseOperands, allocatorOperands,
      allocateOperands;
  Attribute defaultClauseOperand, procBindClauseOperand;
  const auto &opClauseList =
      std::get<Fortran::parser::OmpClauseList>(directive.t);
  for (const Fortran::parser::OmpClause &clause : opClauseList.v) {
    if (const auto &ifClause =
            std::get_if<Fortran::parser::OmpClause::If>(&clause.u)) {
      auto &expr = std::get<Fortran::parser::ScalarLogicalExpr>(ifClause->v.t);
      ifClauseOperand = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(expr), stmtCtx));
    } else if (const auto &numThreadsClause =
                   std::get_if<Fortran::parser::OmpClause::NumThreads>(
                       &clause.u)) {
      numThreadsClauseOperand = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(numThreadsClause->v), stmtCtx));
    } else if (const auto &sharedClause =
                   std::get_if<Fortran::parser::OmpClause::Shared>(&clause.u)) {
      const Fortran::parser::OmpObjectList &ompObjectList = sharedClause->v;
      genObjectList(ompObjectList, converter, sharedClauseOperands);
    } else if (const auto &copyinClause =
                   std::get_if<Fortran::parser::OmpClause::Copyin>(&clause.u)) {
      const Fortran::parser::OmpObjectList &ompObjectList = copyinClause->v;
      genObjectList(ompObjectList, converter, copyinClauseOperands);
    } else if (const auto &allocateClause =
                   std::get_if<Fortran::parser::OmpClause::Allocate>(
                       &clause.u)) {
      mlir::Value allocatorOperand;
      const Fortran::parser::OmpAllocateClause &ompAllocateClause =
          allocateClause->v;
      const Fortran::parser::OmpObjectList &ompObjectList =
          std::get<Fortran::parser::OmpObjectList>(ompAllocateClause.t);
      // Check if allocate clause has allocator specified. If so, add it
      // to list of allocators, otherwise, add default allocator to
      // list of allocators.
      const auto &allocatorValue = std::get<
          std::optional<Fortran::parser::OmpAllocateClause::Allocator>>(
          ompAllocateClause.t);
      if (allocatorValue) {
        allocatorOperand = fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(allocatorValue->v), stmtCtx));
        allocatorOperands.insert(allocatorOperands.end(),
                                 ompObjectList.v.size(), allocatorOperand);
      } else {
        allocatorOperand = firOpBuilder.createIntegerConstant(
            currentLocation, firOpBuilder.getI32Type(), 1);
        allocatorOperands.insert(allocatorOperands.end(),
                                 ompObjectList.v.size(), allocatorOperand);
      }
      genObjectList(ompObjectList, converter, allocateOperands);
    }
  }
  // Create and insert the operation.
  auto parallelOp = firOpBuilder.create<mlir::omp::ParallelOp>(
      currentLocation, argTy, ifClauseOperand, numThreadsClauseOperand,
      defaultClauseOperand.dyn_cast_or_null<StringAttr>(),
      privateClauseOperands, firstprivateClauseOperands, sharedClauseOperands,
      copyinClauseOperands, allocateOperands, allocatorOperands,
      procBindClauseOperand.dyn_cast_or_null<StringAttr>());
  for (const Fortran::parser::OmpClause &clause : opClauseList.v) {
    if (const auto &defaultClause =
            std::get_if<Fortran::parser::OmpClause::Default>(&clause.u)) {
      const auto &ompDefaultClause{defaultClause->v};
      switch (ompDefaultClause.v) {
      case Fortran::parser::OmpDefaultClause::Type::Private:
        parallelOp.default_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseDefault(omp::ClauseDefault::defprivate)));
        break;
      case Fortran::parser::OmpDefaultClause::Type::Firstprivate:
        parallelOp.default_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseDefault(omp::ClauseDefault::deffirstprivate)));
        break;
      case Fortran::parser::OmpDefaultClause::Type::Shared:
        parallelOp.default_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseDefault(omp::ClauseDefault::defshared)));
        break;
      case Fortran::parser::OmpDefaultClause::Type::None:
        parallelOp.default_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseDefault(omp::ClauseDefault::defnone)));
        break;
      }
    }
    if (const auto &procBindClause =
            std::get_if<Fortran::parser::OmpClause::ProcBind>(&clause.u)) {
      const auto &ompProcBindClause{procBindClause->v};
      switch (ompProcBindClause.v) {
      case Fortran::parser::OmpProcBindClause::Type::Master:
        parallelOp.proc_bind_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseProcBindKind(omp::ClauseProcBindKind::master)));
        break;
      case Fortran::parser::OmpProcBindClause::Type::Close:
        parallelOp.proc_bind_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseProcBindKind(omp::ClauseProcBindKind::close)));
        break;
      case Fortran::parser::OmpProcBindClause::Type::Spread:
        parallelOp.proc_bind_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseProcBindKind(omp::ClauseProcBindKind::spread)));
        break;
      }
    }
  }

  createBodyOfOp<omp::ParallelOp>(parallelOp, converter, currentLocation, eval,
                                  &opClauseList, {}, isCombined);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPBlockConstruct &blockConstruct) {
  const auto &beginBlockDirective =
      std::get<Fortran::parser::OmpBeginBlockDirective>(blockConstruct.t);
  const auto &blockDirective =
      std::get<Fortran::parser::OmpBlockDirective>(beginBlockDirective.t);

  if (blockDirective.v == llvm::omp::OMPD_parallel) {
    createParallelOp<Fortran::parser::OmpBeginBlockDirective, false>(
        converter, eval,
        std::get<Fortran::parser::OmpBeginBlockDirective>(blockConstruct.t));
  } else if (blockDirective.v == llvm::omp::OMPD_master) {
    fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
    mlir::Location currentLocation = converter.getCurrentLocation();
    auto masterOp = firOpBuilder.create<mlir::omp::MasterOp>(currentLocation);
    createBodyOfOp<omp::MasterOp>(masterOp, converter, currentLocation, eval);
  } else if (blockDirective.v == llvm::omp::OMPD_ordered) {
    auto &firOpBuilder = converter.getFirOpBuilder();
    auto currentLocation = converter.getCurrentLocation();
    mlir::Attribute simdClauseOperand;
    auto orderedOp = firOpBuilder.create<mlir::omp::OrderedRegionOp>(
        currentLocation, simdClauseOperand.dyn_cast_or_null<UnitAttr>());
    const auto &clauseList =
        std::get<Fortran::parser::OmpClauseList>(beginBlockDirective.t);
    for (const auto &clause : clauseList.v) {
      if (std::get_if<Fortran::parser::OmpClause::Simd>(&clause.u)) {
        TODO(currentLocation, "OpenMP ORDERED SIMD");
      }
    }
    createBodyOfOp<omp::OrderedRegionOp>(orderedOp, converter, currentLocation,
                                         eval);
  }
}

static mlir::omp::ScheduleModifier
translateModifier(const Fortran::parser::OmpScheduleModifierType &m) {
  switch (m.v) {
  case Fortran::parser::OmpScheduleModifierType::ModType::Monotonic:
    return mlir::omp::ScheduleModifier::monotonic;
  case Fortran::parser::OmpScheduleModifierType::ModType::Nonmonotonic:
    return mlir::omp::ScheduleModifier::nonmonotonic;
  case Fortran::parser::OmpScheduleModifierType::ModType::Simd:
    return mlir::omp::ScheduleModifier::simd;
  }
  return mlir::omp::ScheduleModifier::none;
}

static mlir::omp::ScheduleModifier
getScheduleModifiers(const Fortran::parser::OmpScheduleClause &x) {
  const auto &modifier =
      std::get<std::optional<Fortran::parser::OmpScheduleModifier>>(x.t);
  // The input may have the modifier any order, so we look for one that isn't
  // SIMD. If modifier is not set at all, fall down to the bottom and return
  // "none".
  if (modifier) {
    const auto &modType1 =
        std::get<Fortran::parser::OmpScheduleModifier::Modifier1>(modifier->t);
    if (modType1.v.v ==
        Fortran::parser::OmpScheduleModifierType::ModType::Simd) {
      const auto &modType2 = std::get<
          std::optional<Fortran::parser::OmpScheduleModifier::Modifier2>>(
          modifier->t);
      if (modType2 &&
          modType2->v.v !=
              Fortran::parser::OmpScheduleModifierType::ModType::Simd)
        return translateModifier(modType2->v);

      return mlir::omp::ScheduleModifier::none;
    }

    return translateModifier(modType1.v);
  }
  return mlir::omp::ScheduleModifier::none;
}

static mlir::omp::ScheduleModifier
getSIMDModifier(const Fortran::parser::OmpScheduleClause &x) {
  const auto &modifier =
      std::get<std::optional<Fortran::parser::OmpScheduleModifier>>(x.t);
  // Either of the two possible modifiers in the input can be the SIMD modifier,
  // so look in either one, and return simd if we find one. Not found = return
  // "none".
  if (modifier) {
    const auto &modType1 =
        std::get<Fortran::parser::OmpScheduleModifier::Modifier1>(modifier->t);
    if (modType1.v.v == Fortran::parser::OmpScheduleModifierType::ModType::Simd)
      return mlir::omp::ScheduleModifier::simd;

    const auto &modType2 = std::get<
        std::optional<Fortran::parser::OmpScheduleModifier::Modifier2>>(
        modifier->t);
    if (modType2 && modType2->v.v ==
                        Fortran::parser::OmpScheduleModifierType::ModType::Simd)
      return mlir::omp::ScheduleModifier::simd;
  }
  return mlir::omp::ScheduleModifier::none;
}

int64_t Fortran::lower::getCollapseValue(
    const Fortran::parser::OmpClauseList &clauseList) {
  for (const Fortran::parser::OmpClause &clause : clauseList.v) {
    if (const auto &collapseClause =
            std::get_if<Fortran::parser::OmpClause::Collapse>(&clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(collapseClause->v);
      return Fortran::evaluate::ToInt64(*expr).value();
    }
  }
  return 1;
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPLoopConstruct &loopConstruct) {

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  SmallVector<Value, 4> lowerBound, upperBound, step, linearVars,
      linearStepVars, reductionVars;
  mlir::Value scheduleChunkClauseOperand;
  mlir::Attribute scheduleClauseOperand, collapseClauseOperand,
      noWaitClauseOperand, orderedClauseOperand, orderClauseOperand;
  const auto &wsLoopOpClauseList = std::get<Fortran::parser::OmpClauseList>(
      std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t).t);
  if (llvm::omp::OMPD_parallel_do ==
      std::get<Fortran::parser::OmpLoopDirective>(
          std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t).t)
          .v) {
    createParallelOp<Fortran::parser::OmpBeginLoopDirective, true>(
        converter, eval,
        std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t));
  }

  for (const Fortran::parser::OmpClause &clause : wsLoopOpClauseList.v) {
    if (const auto &scheduleClause =
            std::get_if<Fortran::parser::OmpClause::Schedule>(&clause.u)) {
      if (const auto &chunkExpr =
              std::get<std::optional<Fortran::parser::ScalarIntExpr>>(
                  scheduleClause->v.t)) {
        if (const auto *expr = Fortran::semantics::GetExpr(*chunkExpr)) {
          Fortran::lower::StatementContext stmtCtx;
          scheduleChunkClauseOperand =
              fir::getBase(converter.genExprValue(*expr, stmtCtx));
        }
      }
    }
  }

  // Collect the loops to collapse.
  Fortran::lower::pft::Evaluation *doConstructEval =
      &eval.getFirstNestedEvaluation();

  std::int64_t collapseValue =
      Fortran::lower::getCollapseValue(wsLoopOpClauseList);
  std::size_t loopVarTypeSize = 0;
  llvm::SmallVector<const Fortran::semantics::Symbol *> iv;
  do {
    Fortran::lower::pft::Evaluation *doLoop =
        &doConstructEval->getFirstNestedEvaluation();
    auto *doStmt = doLoop->getIf<Fortran::parser::NonLabelDoStmt>();
    assert(doStmt && "Expected do loop to be in the nested evaluation");
    const auto &loopControl =
        std::get<std::optional<Fortran::parser::LoopControl>>(doStmt->t);
    const Fortran::parser::LoopControl::Bounds *bounds =
        std::get_if<Fortran::parser::LoopControl::Bounds>(&loopControl->u);
    if (bounds) {
      Fortran::lower::StatementContext stmtCtx;
      lowerBound.push_back(fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(bounds->lower), stmtCtx)));
      upperBound.push_back(fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(bounds->upper), stmtCtx)));
      if (bounds->step) {
        step.push_back(fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(bounds->step), stmtCtx)));
      }
      // If `step` is not present, assume it as `1`.
      else {
        step.push_back(firOpBuilder.createIntegerConstant(
            currentLocation, firOpBuilder.getIntegerType(32), 1));
      }
      iv.push_back(bounds->name.thing.symbol);
      loopVarTypeSize = std::max(
          loopVarTypeSize, bounds->name.thing.symbol->GetUltimate().size());
    }

    collapseValue--;
    doConstructEval =
        &*std::next(doConstructEval->getNestedEvaluations().begin());
  } while (collapseValue > 0);

  // The types of lower bound, upper bound, and step are converted into the
  // type of the loop variable if necessary.
  mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
  for (unsigned it = 0; it < (unsigned)lowerBound.size(); it++) {
    lowerBound[it] = firOpBuilder.createConvert(currentLocation, loopVarType,
                                                lowerBound[it]);
    upperBound[it] = firOpBuilder.createConvert(currentLocation, loopVarType,
                                                upperBound[it]);
    step[it] =
        firOpBuilder.createConvert(currentLocation, loopVarType, step[it]);
  }

  // FIXME: Add support for following clauses:
  // 1. linear
  // 2. order
  auto wsLoopOp = firOpBuilder.create<mlir::omp::WsLoopOp>(
      currentLocation, lowerBound, upperBound, step, linearVars, linearStepVars,
      reductionVars, /*reductions=*/nullptr,
      scheduleClauseOperand.dyn_cast_or_null<StringAttr>(),
      scheduleChunkClauseOperand, /*schedule_modifiers=*/nullptr,
      /*simd_modifier=*/nullptr,
      collapseClauseOperand.dyn_cast_or_null<IntegerAttr>(),
      noWaitClauseOperand.dyn_cast_or_null<UnitAttr>(),
      orderedClauseOperand.dyn_cast_or_null<IntegerAttr>(),
      orderClauseOperand.dyn_cast_or_null<StringAttr>(),
      /*inclusive=*/firOpBuilder.getUnitAttr());

  // Handle attribute based clauses.
  for (const Fortran::parser::OmpClause &clause : wsLoopOpClauseList.v) {
    if (const auto &collapseClause =
            std::get_if<Fortran::parser::OmpClause::Collapse>(&clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(collapseClause->v);
      const std::optional<std::int64_t> collapseValue =
          Fortran::evaluate::ToInt64(*expr);
      wsLoopOp.collapse_valAttr(firOpBuilder.getI64IntegerAttr(*collapseValue));
    } else if (const auto &orderedClause =
                   std::get_if<Fortran::parser::OmpClause::Ordered>(
                       &clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(orderedClause->v);
      const std::optional<std::int64_t> orderedValue =
          Fortran::evaluate::ToInt64(*expr);
      wsLoopOp.ordered_valAttr(firOpBuilder.getI64IntegerAttr(*orderedValue));
    } else if (const auto &scheduleClause =
                   std::get_if<Fortran::parser::OmpClause::Schedule>(
                       &clause.u)) {
      const auto &scheduleType = scheduleClause->v;
      const auto &scheduleKind =
          std::get<Fortran::parser::OmpScheduleClause::ScheduleType>(
              scheduleType.t);
      switch (scheduleKind) {
      case Fortran::parser::OmpScheduleClause::ScheduleType::Static:
        wsLoopOp.schedule_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseScheduleKind(omp::ClauseScheduleKind::Static)));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Dynamic:
        wsLoopOp.schedule_valAttr(
            firOpBuilder.getStringAttr(omp::stringifyClauseScheduleKind(
                omp::ClauseScheduleKind::Dynamic)));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Guided:
        wsLoopOp.schedule_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseScheduleKind(omp::ClauseScheduleKind::Guided)));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Auto:
        wsLoopOp.schedule_valAttr(firOpBuilder.getStringAttr(
            omp::stringifyClauseScheduleKind(omp::ClauseScheduleKind::Auto)));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Runtime:
        wsLoopOp.schedule_valAttr(
            firOpBuilder.getStringAttr(omp::stringifyClauseScheduleKind(
                omp::ClauseScheduleKind::Runtime)));
        break;
      }
      wsLoopOp.schedule_modifiersAttr(
          firOpBuilder.getStringAttr(omp::stringifyScheduleModifier(
              getScheduleModifiers(scheduleClause->v))));
      wsLoopOp.simd_modifierAttr(firOpBuilder.getStringAttr(
          omp::stringifyScheduleModifier(getSIMDModifier(scheduleClause->v))));
    }
  }
  // In FORTRAN `nowait` clause occur at the end of `omp do` directive.
  // i.e
  // !$omp do
  // <...>
  // !$omp end do nowait
  if (const auto &endClauseList =
          std::get<std::optional<Fortran::parser::OmpEndLoopDirective>>(
              loopConstruct.t)) {
    const auto &clauseList =
        std::get<Fortran::parser::OmpClauseList>((*endClauseList).t);
    for (const Fortran::parser::OmpClause &clause : clauseList.v)
      if (std::get_if<Fortran::parser::OmpClause::Nowait>(&clause.u))
        wsLoopOp.nowaitAttr(firOpBuilder.getUnitAttr());
  }

  createBodyOfOp<omp::WsLoopOp>(wsLoopOp, converter, currentLocation, eval,
                                &wsLoopOpClauseList, iv);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPCriticalConstruct &criticalConstruct) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  std::string name;
  const Fortran::parser::OmpCriticalDirective &cd =
      std::get<Fortran::parser::OmpCriticalDirective>(criticalConstruct.t);
  if (std::get<std::optional<Fortran::parser::Name>>(cd.t).has_value()) {
    name =
        std::get<std::optional<Fortran::parser::Name>>(cd.t).value().ToString();
  }

  uint64_t hint = 0;
  const auto &clauseList = std::get<Fortran::parser::OmpClauseList>(cd.t);
  for (const Fortran::parser::OmpClause &clause : clauseList.v)
    if (auto hintClause =
            std::get_if<Fortran::parser::OmpClause::Hint>(&clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(hintClause->v);
      hint = *Fortran::evaluate::ToInt64(*expr);
      break;
    }

  mlir::omp::CriticalOp criticalOp = [&]() -> mlir::omp::CriticalOp {
    if (name.empty()) {
      return firOpBuilder.create<mlir::omp::CriticalOp>(currentLocation,
                                                        FlatSymbolRefAttr());
    } else {
      mlir::ModuleOp module = firOpBuilder.getModule();
      mlir::OpBuilder modBuilder(module.getBodyRegion());
      auto global = module.lookupSymbol<mlir::omp::CriticalDeclareOp>(name);
      if (!global)
        global = modBuilder.create<mlir::omp::CriticalDeclareOp>(
            currentLocation, name, hint);
      return firOpBuilder.create<mlir::omp::CriticalOp>(
          currentLocation, mlir::FlatSymbolRefAttr::get(
                               firOpBuilder.getContext(), global.sym_name()));
    }
  }();
  createBodyOfOp<omp::CriticalOp>(criticalOp, converter, currentLocation, eval);
}

void Fortran::lower::genOpenMPConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPConstruct &ompConstruct) {

  std::visit(
      common::visitors{
          [&](const Fortran::parser::OpenMPStandaloneConstruct
                  &standaloneConstruct) {
            genOMP(converter, eval, standaloneConstruct);
          },
          [&](const Fortran::parser::OpenMPSectionsConstruct
                  &sectionsConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPSectionsConstruct");
          },
          [&](const Fortran::parser::OpenMPLoopConstruct &loopConstruct) {
            genOMP(converter, eval, loopConstruct);
          },
          [&](const Fortran::parser::OpenMPDeclarativeAllocate
                  &execAllocConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclarativeAllocate");
          },
          [&](const Fortran::parser::OpenMPExecutableAllocate
                  &execAllocConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPExecutableAllocate");
          },
          [&](const Fortran::parser::OpenMPBlockConstruct &blockConstruct) {
            genOMP(converter, eval, blockConstruct);
          },
          [&](const Fortran::parser::OpenMPAtomicConstruct &atomicConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPAtomicConstruct");
          },
          [&](const Fortran::parser::OpenMPCriticalConstruct
                  &criticalConstruct) {
            genOMP(converter, eval, criticalConstruct);
          },
      },
      ompConstruct.u);
}

void Fortran::lower::genOpenMPDeclarativeConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclarativeConstruct &ompDeclConstruct) {

  std::visit(
      common::visitors{
          [&](const Fortran::parser::OpenMPDeclarativeAllocate
                  &declarativeAllocate) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclarativeAllocate");
          },
          [&](const Fortran::parser::OpenMPDeclareReductionConstruct
                  &declareReductionConstruct) {
            TODO(converter.getCurrentLocation(),
                 "OpenMPDeclareReductionConstruct");
          },
          [&](const Fortran::parser::OpenMPDeclareSimdConstruct
                  &declareSimdConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclareSimdConstruct");
          },
          [&](const Fortran::parser::OpenMPDeclareTargetConstruct
                  &declareTargetConstruct) {
            TODO(converter.getCurrentLocation(),
                 "OpenMPDeclareTargetConstruct");
          },
          [&](const Fortran::parser::OpenMPThreadprivate &threadprivate) {
            TODO(converter.getCurrentLocation(), "OpenMPThreadprivate");
          },
      },
      ompDeclConstruct.u);
}
