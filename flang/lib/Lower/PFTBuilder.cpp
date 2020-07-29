//===-- PFTBuilder.cc -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Utils.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/tools.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<bool> clDisableStructuredFir(
    "no-structured-fir", llvm::cl::desc("disable generation of structured FIR"),
    llvm::cl::init(false), llvm::cl::Hidden);

using namespace Fortran;

namespace {
/// Helpers to unveil parser node inside Fortran::parser::Statement<>,
/// Fortran::parser::UnlabeledStatement, and Fortran::common::Indirection<>
template <typename A>
struct RemoveIndirectionHelper {
  using Type = A;
};
template <typename A>
struct RemoveIndirectionHelper<common::Indirection<A>> {
  using Type = A;
};

template <typename A>
struct UnwrapStmt {
  static constexpr bool isStmt{false};
};
template <typename A>
struct UnwrapStmt<parser::Statement<A>> {
  static constexpr bool isStmt{true};
  using Type = typename RemoveIndirectionHelper<A>::Type;
  constexpr UnwrapStmt(const parser::Statement<A> &a)
      : unwrapped{removeIndirection(a.statement)}, position{a.source},
        label{a.label} {}
  const Type &unwrapped;
  parser::CharBlock position;
  std::optional<parser::Label> label;
};
template <typename A>
struct UnwrapStmt<parser::UnlabeledStatement<A>> {
  static constexpr bool isStmt{true};
  using Type = typename RemoveIndirectionHelper<A>::Type;
  constexpr UnwrapStmt(const parser::UnlabeledStatement<A> &a)
      : unwrapped{removeIndirection(a.statement)}, position{a.source} {}
  const Type &unwrapped;
  parser::CharBlock position;
  std::optional<parser::Label> label;
};

/// The instantiation of a parse tree visitor (Pre and Post) is extremely
/// expensive in terms of compile and link time.  So one goal here is to
/// limit the bridge to one such instantiation.
class PFTBuilder {
public:
  PFTBuilder(const semantics::SemanticsContext &semanticsContext)
      : pgm{std::make_unique<lower::pft::Program>()},
        parentVariantStack{*pgm.get()}, semanticsContext{semanticsContext} {}

  /// Get the result
  std::unique_ptr<lower::pft::Program> result() { return std::move(pgm); }

  template <typename A>
  constexpr bool Pre(const A &a) {
    if constexpr (lower::pft::isFunctionLike<A>) {
      return enterFunction(a, semanticsContext);
    } else if constexpr (lower::pft::isConstruct<A> ||
                         lower::pft::isDirective<A>) {
      return enterConstructOrDirective(a);
    } else if constexpr (UnwrapStmt<A>::isStmt) {
      using T = typename UnwrapStmt<A>::Type;
      // Node "a" being visited has one of the following types:
      // Statement<T>, Statement<Indirection<T>>, UnlabeledStatement<T>,
      // or UnlabeledStatement<Indirection<T>>
      auto stmt{UnwrapStmt<A>(a)};
      if constexpr (lower::pft::isConstructStmt<T> ||
                    lower::pft::isOtherStmt<T>) {
        addEvaluation(lower::pft::Evaluation{stmt.unwrapped,
                                             parentVariantStack.back(),
                                             stmt.position, stmt.label});
        return false;
      } else if constexpr (std::is_same_v<T, parser::ActionStmt>) {
        addEvaluation(
            makeEvaluationAction(stmt.unwrapped, stmt.position, stmt.label));
        return true;
      }
    }
    return true;
  }

  template <typename A>
  constexpr void Post(const A &) {
    if constexpr (lower::pft::isFunctionLike<A>) {
      exitFunction();
    } else if constexpr (lower::pft::isConstruct<A> ||
                         lower::pft::isDirective<A>) {
      exitConstructOrDirective();
    }
  }

  // Module like
  bool Pre(const parser::Module &node) { return enterModule(node); }
  bool Pre(const parser::Submodule &node) { return enterModule(node); }

  void Post(const parser::Module &) { exitModule(); }
  void Post(const parser::Submodule &) { exitModule(); }

  // Block data
  bool Pre(const parser::BlockData &node) {
    addUnit(lower::pft::BlockDataUnit{node, parentVariantStack.back(),
                                      semanticsContext});
    return false;
  }

  // Get rid of production wrapper
  bool Pre(const parser::UnlabeledStatement<parser::ForallAssignmentStmt>
               &statement) {
    addEvaluation(std::visit(
        [&](const auto &x) {
          return lower::pft::Evaluation{
              x, parentVariantStack.back(), statement.source, {}};
        },
        statement.statement.u));
    return false;
  }
  bool Pre(const parser::Statement<parser::ForallAssignmentStmt> &statement) {
    addEvaluation(std::visit(
        [&](const auto &x) {
          return lower::pft::Evaluation{x, parentVariantStack.back(),
                                        statement.source, statement.label};
        },
        statement.statement.u));
    return false;
  }
  bool Pre(const parser::WhereBodyConstruct &whereBody) {
    return std::visit(
        common::visitors{
            [&](const parser::Statement<parser::AssignmentStmt> &stmt) {
              // Not caught as other AssignmentStmt because it is not
              // wrapped in a parser::ActionStmt.
              addEvaluation(lower::pft::Evaluation{stmt.statement,
                                                   parentVariantStack.back(),
                                                   stmt.source, stmt.label});
              return false;
            },
            [&](const auto &) { return true; },
        },
        whereBody.u);
  }

private:
  /// Initialize a new module-like unit and make it the builder's focus.
  template <typename A>
  bool enterModule(const A &func) {
    auto &unit =
        addUnit(lower::pft::ModuleLikeUnit{func, parentVariantStack.back()});
    functionList = &unit.nestedFunctions;
    parentVariantStack.emplace_back(unit);
    return true;
  }

  void exitModule() {
    parentVariantStack.pop_back();
    resetFunctionState();
  }

  /// Ensure that a function ends with a valid branch target (and is nonempty).
  void endFunctionBody() {
    if (evaluationListStack.empty())
      return;
    auto evaluationList = evaluationListStack.back();
    if (evaluationList->empty() ||
        !evaluationList->back().isA<parser::ContinueStmt>()) {
      static const parser::ContinueStmt endTarget{};
      addEvaluation(
          lower::pft::Evaluation{endTarget, parentVariantStack.back(), {}, {}});
    }
    lastLexicalEvaluation = nullptr;
  }

  /// Initialize a new function-like unit and make it the builder's focus.
  template <typename A>
  bool enterFunction(const A &func,
                     const semantics::SemanticsContext &semanticsContext) {
    endFunctionBody(); // enclosing host subprogram body, if any
    auto &unit = addFunction(lower::pft::FunctionLikeUnit{
        func, parentVariantStack.back(), semanticsContext});
    labelEvaluationMap = &unit.labelEvaluationMap;
    assignSymbolLabelMap = &unit.assignSymbolLabelMap;
    functionList = &unit.nestedFunctions;
    pushEvaluationList(&unit.evaluationList);
    parentVariantStack.emplace_back(unit);
    return true;
  }

  void exitFunction() {
    endFunctionBody();
    analyzeBranches(nullptr, *evaluationListStack.back()); // add branch links
    processEntryPoints();
    popEvaluationList();
    labelEvaluationMap = nullptr;
    assignSymbolLabelMap = nullptr;
    parentVariantStack.pop_back();
    resetFunctionState();
  }

  /// Initialize a new construct and make it the builder's focus.
  template <typename A>
  bool enterConstructOrDirective(const A &construct) {
    auto &eval = addEvaluation(
        lower::pft::Evaluation{construct, parentVariantStack.back()});
    eval.evaluationList.reset(new lower::pft::EvaluationList);
    pushEvaluationList(eval.evaluationList.get());
    parentVariantStack.emplace_back(eval);
    constructAndDirectiveStack.emplace_back(&eval);
    return true;
  }

  void exitConstructOrDirective() {
    popEvaluationList();
    parentVariantStack.pop_back();
    constructAndDirectiveStack.pop_back();
  }

  /// Reset function state to that of an enclosing host function.
  void resetFunctionState() {
    if (!parentVariantStack.empty()) {
      parentVariantStack.back().visit(common::visitors{
          [&](lower::pft::FunctionLikeUnit &p) {
            functionList = &p.nestedFunctions;
            labelEvaluationMap = &p.labelEvaluationMap;
            assignSymbolLabelMap = &p.assignSymbolLabelMap;
          },
          [&](lower::pft::ModuleLikeUnit &p) {
            functionList = &p.nestedFunctions;
          },
          [&](auto &) { functionList = nullptr; },
      });
    }
  }

  template <typename A>
  A &addUnit(A &&unit) {
    pgm->getUnits().emplace_back(std::move(unit));
    return std::get<A>(pgm->getUnits().back());
  }

  template <typename A>
  A &addFunction(A &&func) {
    if (functionList) {
      functionList->emplace_back(std::move(func));
      return functionList->back();
    }
    return addUnit(std::move(func));
  }

  // ActionStmt has a couple of non-conforming cases, explicitly handled here.
  // The other cases use an Indirection, which are discarded in the PFT.
  lower::pft::Evaluation
  makeEvaluationAction(const parser::ActionStmt &statement,
                       parser::CharBlock position,
                       std::optional<parser::Label> label) {
    return std::visit(
        common::visitors{
            [&](const auto &x) {
              return lower::pft::Evaluation{removeIndirection(x),
                                            parentVariantStack.back(), position,
                                            label};
            },
        },
        statement.u);
  }

  /// Append an Evaluation to the end of the current list.
  lower::pft::Evaluation &addEvaluation(lower::pft::Evaluation &&eval) {
    assert(functionList && "not in a function");
    assert(!evaluationListStack.empty() && "empty evaluation list stack");
    if (!constructAndDirectiveStack.empty())
      eval.parentConstruct = constructAndDirectiveStack.back();
    auto &entryPointList = eval.getOwningProcedure()->entryPointList;
    evaluationListStack.back()->emplace_back(std::move(eval));
    lower::pft::Evaluation *p = &evaluationListStack.back()->back();
    if (p->isActionStmt() || p->isConstructStmt()) {
      if (lastLexicalEvaluation) {
        lastLexicalEvaluation->lexicalSuccessor = p;
        p->printIndex = lastLexicalEvaluation->printIndex + 1;
      } else {
        p->printIndex = 1;
      }
      lastLexicalEvaluation = p;
      for (auto entryIndex = entryPointList.size() - 1;
           entryIndex && !entryPointList[entryIndex].second->lexicalSuccessor;
           --entryIndex)
        // Link to the entry's first executable statement.
        entryPointList[entryIndex].second->lexicalSuccessor = p;
    } else if (const auto *entryStmt = p->getIf<parser::EntryStmt>()) {
      const auto *sym = std::get<Fortran::parser::Name>(entryStmt->t).symbol;
      assert(sym->has<semantics::SubprogramDetails>() &&
             "entry must be a subprogram");
      entryPointList.push_back(std::pair{sym, p});
    }
    if (p->label.has_value()) {
      labelEvaluationMap->try_emplace(*p->label, p);
    }
    return evaluationListStack.back()->back();
  }

  /// push a new list on the stack of Evaluation lists
  void pushEvaluationList(lower::pft::EvaluationList *eval) {
    assert(functionList && "not in a function");
    assert(eval && eval->empty() && "evaluation list isn't correct");
    evaluationListStack.emplace_back(eval);
  }

  /// pop the current list and return to the last Evaluation list
  void popEvaluationList() {
    assert(functionList && "not in a function");
    evaluationListStack.pop_back();
  }

  /// Mark I/O statement ERR, EOR, and END specifier branch targets.
  template <typename A>
  void analyzeIoBranches(lower::pft::Evaluation &eval, const A &stmt) {
    auto processIfLabel{[&](const auto &specs) {
      using LabelNodes =
          std::tuple<parser::ErrLabel, parser::EorLabel, parser::EndLabel>;
      for (const auto &spec : specs) {
        const auto *label = std::visit(
            [](const auto &label) -> const parser::Label * {
              using B = std::decay_t<decltype(label)>;
              if constexpr (common::HasMember<B, LabelNodes>) {
                return &label.v;
              }
              return nullptr;
            },
            spec.u);

        if (label)
          markBranchTarget(eval, *label);
      }
    }};

    using OtherIOStmts =
        std::tuple<parser::BackspaceStmt, parser::CloseStmt,
                   parser::EndfileStmt, parser::FlushStmt, parser::OpenStmt,
                   parser::RewindStmt, parser::WaitStmt>;

    if constexpr (std::is_same_v<A, parser::ReadStmt> ||
                  std::is_same_v<A, parser::WriteStmt>) {
      processIfLabel(stmt.controls);
    } else if constexpr (std::is_same_v<A, parser::InquireStmt>) {
      if (const auto *specs =
              std::get_if<std::list<parser::InquireSpec>>(&stmt.u))
        processIfLabel(*specs);
    } else if constexpr (common::HasMember<A, OtherIOStmts>) {
      processIfLabel(stmt.v);
    } else {
      // Always crash if this is instantiated
      static_assert(!std::is_same_v<A, parser::ReadStmt>,
                    "Unexpected IO statement");
    }
  }

  /// Set the exit of a construct, possibly from multiple enclosing constructs.
  void setConstructExit(lower::pft::Evaluation &eval) {
    eval.constructExit = &eval.evaluationList->back().nonNopSuccessor();
  }

  /// Mark the target of a branch as a new block.
  void markBranchTarget(lower::pft::Evaluation &sourceEvaluation,
                        lower::pft::Evaluation &targetEvaluation) {
    sourceEvaluation.isUnstructured = true;
    if (!sourceEvaluation.controlSuccessor)
      sourceEvaluation.controlSuccessor = &targetEvaluation;
    targetEvaluation.isNewBlock = true;
    // If this is a branch into the body of a construct (usually illegal,
    // but allowed in some legacy cases), then the targetEvaluation and its
    // ancestors must be marked as unstructured.
    auto *sourceConstruct = sourceEvaluation.parentConstruct;
    auto *targetConstruct = targetEvaluation.parentConstruct;
    if (targetEvaluation.isConstructStmt() &&
        &targetConstruct->getFirstNestedEvaluation() == &targetEvaluation)
      // A branch to an initial constructStmt is a branch to the construct.
      targetConstruct = targetConstruct->parentConstruct;
    if (targetConstruct) {
      while (sourceConstruct && sourceConstruct != targetConstruct)
        sourceConstruct = sourceConstruct->parentConstruct;
      if (sourceConstruct != targetConstruct)
        for (auto *eval = &targetEvaluation; eval; eval = eval->parentConstruct)
          eval->isUnstructured = true;
    }
  }
  void markBranchTarget(lower::pft::Evaluation &sourceEvaluation,
                        parser::Label label) {
    assert(label && "missing branch target label");
    lower::pft::Evaluation *targetEvaluation{
        labelEvaluationMap->find(label)->second};
    assert(targetEvaluation && "missing branch target evaluation");
    markBranchTarget(sourceEvaluation, *targetEvaluation);
  }

  /// Mark the successor of an Evaluation as a new block.
  void markSuccessorAsNewBlock(lower::pft::Evaluation &eval) {
    eval.nonNopSuccessor().isNewBlock = true;
  }

  template <typename A>
  inline std::string getConstructName(const A &stmt) {
    using MaybeConstructNameWrapper =
        std::tuple<parser::BlockStmt, parser::CycleStmt, parser::ElseStmt,
                   parser::ElsewhereStmt, parser::EndAssociateStmt,
                   parser::EndBlockStmt, parser::EndCriticalStmt,
                   parser::EndDoStmt, parser::EndForallStmt, parser::EndIfStmt,
                   parser::EndSelectStmt, parser::EndWhereStmt,
                   parser::ExitStmt>;
    if constexpr (common::HasMember<A, MaybeConstructNameWrapper>) {
      if (stmt.v)
        return stmt.v->ToString();
    }

    using MaybeConstructNameInTuple = std::tuple<
        parser::AssociateStmt, parser::CaseStmt, parser::ChangeTeamStmt,
        parser::CriticalStmt, parser::ElseIfStmt, parser::EndChangeTeamStmt,
        parser::ForallConstructStmt, parser::IfThenStmt, parser::LabelDoStmt,
        parser::MaskedElsewhereStmt, parser::NonLabelDoStmt,
        parser::SelectCaseStmt, parser::SelectRankCaseStmt,
        parser::TypeGuardStmt, parser::WhereConstructStmt>;

    if constexpr (common::HasMember<A, MaybeConstructNameInTuple>) {
      if (auto name{std::get<std::optional<parser::Name>>(stmt.t)})
        return name->ToString();
    }

    // These statements have several std::optional<parser::Name>
    if constexpr (std::is_same_v<A, parser::SelectRankStmt> ||
                  std::is_same_v<A, parser::SelectTypeStmt>) {
      if (auto name{std::get<0>(stmt.t)})
        return name->ToString();
    }
    return {};
  }

  /// \p parentConstruct can be null if this statement is at the highest
  /// level of a program.
  template <typename A>
  void insertConstructName(const A &stmt,
                           lower::pft::Evaluation *parentConstruct) {
    std::string name{getConstructName(stmt)};
    if (!name.empty())
      constructNameMap[name] = parentConstruct;
  }

  /// Insert branch links for a list of Evaluations.
  /// \p parentConstruct can be null if the evaluationList contains the
  /// top-level statements of a program.
  void analyzeBranches(lower::pft::Evaluation *parentConstruct,
                       std::list<lower::pft::Evaluation> &evaluationList) {
    lower::pft::Evaluation *lastConstructStmtEvaluation{nullptr};
    lower::pft::Evaluation *lastIfStmtEvaluation{nullptr};
    for (auto &eval : evaluationList) {
      eval.visit(common::visitors{
          // Action statements
          [&](const parser::CallStmt &s) {
            // Look for alternate return specifiers.
            const auto &args{std::get<std::list<parser::ActualArgSpec>>(s.v.t)};
            for (const auto &arg : args) {
              const auto &actual{std::get<parser::ActualArg>(arg.t)};
              if (const auto *altReturn{
                      std::get_if<parser::AltReturnSpec>(&actual.u)}) {
                markBranchTarget(eval, altReturn->v);
              }
            }
          },
          [&](const parser::CycleStmt &s) {
            std::string name{getConstructName(s)};
            lower::pft::Evaluation *construct{name.empty()
                                                  ? doConstructStack.back()
                                                  : constructNameMap[name]};
            assert(construct && "missing CYCLE construct");
            markBranchTarget(eval, construct->evaluationList->back());
          },
          [&](const parser::ExitStmt &s) {
            std::string name{getConstructName(s)};
            lower::pft::Evaluation *construct{name.empty()
                                                  ? doConstructStack.back()
                                                  : constructNameMap[name]};
            assert(construct && "missing EXIT construct");
            markBranchTarget(eval, *construct->constructExit);
          },
          [&](const parser::GotoStmt &s) { markBranchTarget(eval, s.v); },
          [&](const parser::IfStmt &) { lastIfStmtEvaluation = &eval; },
          [&](const parser::ReturnStmt &) {
            eval.isUnstructured = true;
            if (eval.lexicalSuccessor->lexicalSuccessor)
              markSuccessorAsNewBlock(eval);
          },
          [&](const parser::StopStmt &) {
            eval.isUnstructured = true;
            if (eval.lexicalSuccessor->lexicalSuccessor)
              markSuccessorAsNewBlock(eval);
          },
          [&](const parser::ComputedGotoStmt &s) {
            for (auto &label : std::get<std::list<parser::Label>>(s.t))
              markBranchTarget(eval, label);
          },
          [&](const parser::ArithmeticIfStmt &s) {
            markBranchTarget(eval, std::get<1>(s.t));
            markBranchTarget(eval, std::get<2>(s.t));
            markBranchTarget(eval, std::get<3>(s.t));
            if (semantics::ExprHasTypeCategory(
                    *semantics::GetExpr(std::get<parser::Expr>(s.t)),
                    common::TypeCategory::Real)) {
              // Real expression evaluation uses an additional local block.
              eval.localBlocks.emplace_back(nullptr);
            }
          },
          [&](const parser::AssignStmt &s) { // legacy label assignment
            auto &label = std::get<parser::Label>(s.t);
            const auto *sym = std::get<parser::Name>(s.t).symbol;
            assert(sym && "missing AssignStmt symbol");
            lower::pft::Evaluation *target{
                labelEvaluationMap->find(label)->second};
            assert(target && "missing branch target evaluation");
            if (!target->isA<parser::FormatStmt>()) {
              target->isNewBlock = true;
            }
            auto iter = assignSymbolLabelMap->find(*sym);
            if (iter == assignSymbolLabelMap->end()) {
              lower::pft::LabelSet labelSet{};
              labelSet.insert(label);
              assignSymbolLabelMap->try_emplace(*sym, labelSet);
            } else {
              iter->second.insert(label);
            }
          },
          [&](const parser::AssignedGotoStmt &) {
            // Although this statement is a branch, it doesn't have any
            // explicit control successors.  So the code at the end of the
            // loop won't mark the successor.  Do that here.
            eval.isUnstructured = true;
            markSuccessorAsNewBlock(eval);
          },

          // Construct statements
          [&](const parser::AssociateStmt &s) {
            insertConstructName(s, parentConstruct);
          },
          [&](const parser::BlockStmt &s) {
            insertConstructName(s, parentConstruct);
          },
          [&](const parser::SelectCaseStmt &s) {
            insertConstructName(s, parentConstruct);
            lastConstructStmtEvaluation = &eval;
          },
          [&](const parser::CaseStmt &) {
            eval.isNewBlock = true;
            lastConstructStmtEvaluation->controlSuccessor = &eval;
            lastConstructStmtEvaluation = &eval;
          },
          [&](const parser::EndSelectStmt &) {
            eval.nonNopSuccessor().isNewBlock = true;
            lastConstructStmtEvaluation = nullptr;
          },
          [&](const parser::ChangeTeamStmt &s) {
            insertConstructName(s, parentConstruct);
          },
          [&](const parser::CriticalStmt &s) {
            insertConstructName(s, parentConstruct);
          },
          [&](const parser::NonLabelDoStmt &s) {
            insertConstructName(s, parentConstruct);
            doConstructStack.push_back(parentConstruct);
            auto &control{std::get<std::optional<parser::LoopControl>>(s.t)};
            // eval.block is the loop preheader block, which will be set
            // elsewhere if the NonLabelDoStmt is itself a target.
            // eval.localBlocks[0] is the loop header block.
            eval.localBlocks.emplace_back(nullptr);
            if (!control.has_value()) {
              eval.isUnstructured = true; // infinite loop
              return;
            }
            eval.nonNopSuccessor().isNewBlock = true;
            eval.controlSuccessor = &evaluationList.back();
            if (std::holds_alternative<parser::ScalarLogicalExpr>(control->u))
              eval.isUnstructured = true; // while loop
            // Defer additional processing for an unstructured concurrent loop
            // to the EndDoStmt, when the loop is known to be unstructured.
          },
          [&](const parser::EndDoStmt &) {
            lower::pft::Evaluation &doEval{evaluationList.front()};
            eval.controlSuccessor = &doEval;
            doConstructStack.pop_back();
            if (parentConstruct->lowerAsStructured())
              return;

            // Now that the loop is known to be unstructured, finish concurrent
            // loop processing, using NonLabelDoStmt information.
            parentConstruct->constructExit->isNewBlock = true;
            const auto &doStmt = doEval.getIf<parser::NonLabelDoStmt>();
            assert(doStmt && "missing NonLabelDoStmt");
            auto &control =
                std::get<std::optional<parser::LoopControl>>(doStmt->t);
            if (!control.has_value())
              return; // infinite loop

            const auto *concurrent =
                std::get_if<parser::LoopControl::Concurrent>(&control->u);
            if (!concurrent)
              return;

            // Unstructured concurrent loop.  NonLabelDoStmt code accounts
            // for one concurrent loop dimension.  Reserve preheader,
            // header, and latch blocks for the remaining dimensions, and
            // one block for a mask expression.
            const auto &header =
                std::get<parser::ConcurrentHeader>(concurrent->t);
            auto dims =
                std::get<std::list<parser::ConcurrentControl>>(header.t).size();
            for (; dims > 1; --dims) {
              doEval.localBlocks.emplace_back(nullptr); // preheader
              doEval.localBlocks.emplace_back(nullptr); // header
              eval.localBlocks.emplace_back(nullptr);   // latch
            }
            if (std::get<std::optional<parser::ScalarLogicalExpr>>(header.t))
              doEval.localBlocks.emplace_back(nullptr); // mask
          },
          [&](const parser::IfThenStmt &s) {
            insertConstructName(s, parentConstruct);
            eval.lexicalSuccessor->isNewBlock = true;
            lastConstructStmtEvaluation = &eval;
          },
          [&](const parser::ElseIfStmt &) {
            eval.isNewBlock = true;
            eval.lexicalSuccessor->isNewBlock = true;
            lastConstructStmtEvaluation->controlSuccessor = &eval;
            lastConstructStmtEvaluation = &eval;
          },
          [&](const parser::ElseStmt &) {
            eval.isNewBlock = true;
            lastConstructStmtEvaluation->controlSuccessor = &eval;
            lastConstructStmtEvaluation = nullptr;
          },
          [&](const parser::EndIfStmt &) {
            if (parentConstruct->lowerAsUnstructured())
              parentConstruct->constructExit->isNewBlock = true;
            if (lastConstructStmtEvaluation) {
              lastConstructStmtEvaluation->controlSuccessor =
                  parentConstruct->constructExit;
              lastConstructStmtEvaluation = nullptr;
            }
          },
          [&](const parser::SelectRankStmt &s) {
            insertConstructName(s, parentConstruct);
          },
          [&](const parser::SelectRankCaseStmt &) { eval.isNewBlock = true; },
          [&](const parser::SelectTypeStmt &s) {
            insertConstructName(s, parentConstruct);
          },
          [&](const parser::TypeGuardStmt &) { eval.isNewBlock = true; },

          // Constructs - set (unstructured) construct exit targets
          [&](const parser::AssociateConstruct &) { setConstructExit(eval); },
          [&](const parser::BlockConstruct &) {
            // EndBlockStmt may have code.
            eval.constructExit = &eval.evaluationList->back();
          },
          [&](const parser::CaseConstruct &) {
            setConstructExit(eval);
            eval.isUnstructured = true;
          },
          [&](const parser::ChangeTeamConstruct &) {
            // EndChangeTeamStmt may have code.
            eval.constructExit = &eval.evaluationList->back();
          },
          [&](const parser::CriticalConstruct &) {
            // EndCriticalStmt may have code.
            eval.constructExit = &eval.evaluationList->back();
          },
          [&](const parser::DoConstruct &) { setConstructExit(eval); },
          [&](const parser::IfConstruct &) { setConstructExit(eval); },
          [&](const parser::SelectRankConstruct &) {
            setConstructExit(eval);
            eval.isUnstructured = true;
          },
          [&](const parser::SelectTypeConstruct &) {
            setConstructExit(eval);
            eval.isUnstructured = true;
          },

          [&](const auto &stmt) {
            using A = std::decay_t<decltype(stmt)>;
            using IoStmts = std::tuple<parser::BackspaceStmt, parser::CloseStmt,
                                       parser::EndfileStmt, parser::FlushStmt,
                                       parser::InquireStmt, parser::OpenStmt,
                                       parser::ReadStmt, parser::RewindStmt,
                                       parser::WaitStmt, parser::WriteStmt>;
            if constexpr (common::HasMember<A, IoStmts>) {
              analyzeIoBranches(eval, stmt);
            }

            /* do nothing */
          },
      });

      // Analyze construct evaluations.
      if (eval.evaluationList)
        analyzeBranches(&eval, *eval.evaluationList);

      // Insert branch links for an unstructured IF statement.
      if (lastIfStmtEvaluation && lastIfStmtEvaluation != &eval) {
        // eval is the action substatement of an IfStmt.
        if (eval.lowerAsUnstructured()) {
          eval.isNewBlock = true;
          markSuccessorAsNewBlock(eval);
          lastIfStmtEvaluation->isUnstructured = true;
        }
        lastIfStmtEvaluation->controlSuccessor = &eval.nonNopSuccessor();
        lastIfStmtEvaluation = nullptr;
      }

      // Set the successor of the last statement in an IF or SELECT block.
      if (!eval.controlSuccessor && eval.lexicalSuccessor &&
          eval.lexicalSuccessor->isIntermediateConstructStmt()) {
        eval.controlSuccessor = parentConstruct->constructExit;
        eval.lexicalSuccessor->isNewBlock = true;
      }

      // Propagate isUnstructured flag to enclosing construct.
      if (parentConstruct && eval.isUnstructured) {
        parentConstruct->isUnstructured = true;
      }

      // The successor of a branch starts a new block.
      if (eval.controlSuccessor && eval.isActionStmt() &&
          eval.lowerAsUnstructured()) {
        markSuccessorAsNewBlock(eval);
      }
    }
  }

  /// For multiple entry subprograms, build a list of the dummy arguments that
  /// appear in some, but not all entry points.  For those that are functions,
  /// also find one of the largest function results, since a single result
  /// container holds the result for all entries.
  void processEntryPoints() {
    auto *unit = evaluationListStack.back()->front().getOwningProcedure();
    int entryCount = unit->entryPointList.size();
    if (entryCount == 1)
      return;
    llvm::DenseMap<Fortran::semantics::Symbol *, int> dummyCountMap;
    for (int entryIndex = 0; entryIndex < entryCount; ++entryIndex) {
      unit->setActiveEntry(entryIndex);
      const auto &details = unit->getSubprogramSymbol()
                                .get<Fortran::semantics::SubprogramDetails>();
      for (auto *arg : details.dummyArgs()) {
        if (!arg)
          continue; // alternate return specifier (no actual argument)
        const auto iter = dummyCountMap.find(arg);
        if (iter == dummyCountMap.end())
          dummyCountMap.try_emplace(arg, 1);
        else
          ++iter->second;
      }
      if (details.isFunction()) {
        const auto *resultSym = &details.result();
        assert(resultSym && "missing result symbol");
        if (!unit->primaryResult ||
            unit->primaryResult->size() < resultSym->size())
          unit->primaryResult = resultSym;
      }
    }
    unit->setActiveEntry(0);
    for (auto arg : dummyCountMap)
      if (arg.second < entryCount)
        unit->nonUniversalDummyArguments.push_back(arg.first);
    // Sort to provide generated code order stability.
    std::sort(unit->nonUniversalDummyArguments.begin(),
              unit->nonUniversalDummyArguments.end(), std::greater<>());
  }

  std::unique_ptr<lower::pft::Program> pgm;
  std::vector<lower::pft::ParentVariant> parentVariantStack;
  const semantics::SemanticsContext &semanticsContext;

  /// functionList points to the internal or module procedure function list
  /// of a FunctionLikeUnit or a ModuleLikeUnit.  It may be null.
  std::list<lower::pft::FunctionLikeUnit> *functionList{nullptr};
  std::vector<lower::pft::Evaluation *> constructAndDirectiveStack{};
  std::vector<lower::pft::Evaluation *> doConstructStack{};
  /// evaluationListStack is the current nested construct evaluationList state.
  std::vector<lower::pft::EvaluationList *> evaluationListStack{};
  llvm::DenseMap<parser::Label, lower::pft::Evaluation *> *labelEvaluationMap{};
  lower::pft::SymbolLabelMap *assignSymbolLabelMap{nullptr};
  std::map<std::string, lower::pft::Evaluation *> constructNameMap{};
  lower::pft::Evaluation *lastLexicalEvaluation{nullptr};
};

class PFTDumper {
public:
  void dumpPFT(llvm::raw_ostream &outputStream,
               const lower::pft::Program &pft) {
    for (auto &unit : pft.getUnits()) {
      std::visit(common::visitors{
                     [&](const lower::pft::BlockDataUnit &unit) {
                       outputStream << getNodeIndex(unit) << " ";
                       outputStream << "BlockData: ";
                       outputStream << "\nEndBlockData\n\n";
                     },
                     [&](const lower::pft::FunctionLikeUnit &func) {
                       dumpFunctionLikeUnit(outputStream, func);
                     },
                     [&](const lower::pft::ModuleLikeUnit &unit) {
                       dumpModuleLikeUnit(outputStream, unit);
                     },
                 },
                 unit);
    }
  }

  llvm::StringRef evaluationName(const lower::pft::Evaluation &eval) {
    return eval.visit([](const auto &parseTreeNode) {
      return parser::ParseTreeDumper::GetNodeName(parseTreeNode);
    });
  }

  void dumpEvaluation(llvm::raw_ostream &outputStream,
                      const lower::pft::Evaluation &eval,
                      const std::string &indentString, int indent = 1) {
    llvm::StringRef name = evaluationName(eval);
    std::string bang = eval.isUnstructured ? "!" : "";
    if (eval.isConstruct() || eval.isDirective()) {
      outputStream << indentString << "<<" << name << bang << ">>";
      if (eval.constructExit)
        outputStream << " -> " << eval.constructExit->printIndex;
      outputStream << '\n';
      dumpEvaluationList(outputStream, *eval.evaluationList, indent + 1);
      outputStream << indentString << "<<End " << name << bang << ">>\n";
      return;
    }
    outputStream << indentString;
    if (eval.printIndex)
      outputStream << eval.printIndex << ' ';
    if (eval.isNewBlock)
      outputStream << '^';
    if (eval.localBlocks.size())
      outputStream << '*';
    outputStream << name << bang;
    if (eval.isActionStmt() || eval.isConstructStmt()) {
      if (eval.controlSuccessor)
        outputStream << " -> " << eval.controlSuccessor->printIndex;
    } else if (eval.isA<parser::EntryStmt>() && eval.lexicalSuccessor) {
      outputStream << " -> " << eval.lexicalSuccessor->printIndex;
    }
    if (!eval.position.empty())
      outputStream << ": " << eval.position.ToString();
    outputStream << '\n';
  }

  void dumpEvaluation(llvm::raw_ostream &ostream,
                      const lower::pft::Evaluation &eval) {
    dumpEvaluation(ostream, eval, "");
  }

  void dumpEvaluationList(llvm::raw_ostream &outputStream,
                          const lower::pft::EvaluationList &evaluationList,
                          int indent = 1) {
    static const auto white = "                                      ++"s;
    auto indentString = white.substr(0, indent * 2);
    for (const auto &eval : evaluationList)
      dumpEvaluation(outputStream, eval, indentString, indent);
  }

  void
  dumpFunctionLikeUnit(llvm::raw_ostream &outputStream,
                       const lower::pft::FunctionLikeUnit &functionLikeUnit) {
    outputStream << getNodeIndex(functionLikeUnit) << " ";
    llvm::StringRef unitKind;
    llvm::StringRef name;
    llvm::StringRef header;
    if (functionLikeUnit.beginStmt) {
      functionLikeUnit.beginStmt->visit(common::visitors{
          [&](const parser::Statement<parser::ProgramStmt> &stmt) {
            unitKind = "Program";
            name = toStringRef(stmt.statement.v.source);
          },
          [&](const parser::Statement<parser::FunctionStmt> &stmt) {
            unitKind = "Function";
            name = toStringRef(std::get<parser::Name>(stmt.statement.t).source);
            header = toStringRef(stmt.source);
          },
          [&](const parser::Statement<parser::SubroutineStmt> &stmt) {
            unitKind = "Subroutine";
            name = toStringRef(std::get<parser::Name>(stmt.statement.t).source);
            header = toStringRef(stmt.source);
          },
          [&](const parser::Statement<parser::MpSubprogramStmt> &stmt) {
            unitKind = "MpSubprogram";
            name = toStringRef(stmt.statement.v.source);
            header = toStringRef(stmt.source);
          },
          [&](const auto &) { llvm_unreachable("not a valid begin stmt"); },
      });
    } else {
      unitKind = "Program";
      name = "<anonymous>";
    }
    outputStream << unitKind << ' ' << name;
    if (!header.empty())
      outputStream << ": " << header;
    outputStream << '\n';
    dumpEvaluationList(outputStream, functionLikeUnit.evaluationList);
    if (!functionLikeUnit.nestedFunctions.empty()) {
      outputStream << "\nContains\n";
      for (auto &func : functionLikeUnit.nestedFunctions)
        dumpFunctionLikeUnit(outputStream, func);
      outputStream << "EndContains\n";
    }
    outputStream << "End" << unitKind << ' ' << name << "\n\n";
  }

  void dumpModuleLikeUnit(llvm::raw_ostream &outputStream,
                          const lower::pft::ModuleLikeUnit &moduleLikeUnit) {
    outputStream << getNodeIndex(moduleLikeUnit) << " ";
    outputStream << "ModuleLike: ";
    outputStream << "\nContains\n";
    for (auto &func : moduleLikeUnit.nestedFunctions)
      dumpFunctionLikeUnit(outputStream, func);
    outputStream << "EndContains\nEndModuleLike\n\n";
  }

  template <typename T>
  std::size_t getNodeIndex(const T &node) {
    auto addr = static_cast<const void *>(&node);
    auto it = nodeIndexes.find(addr);
    if (it != nodeIndexes.end())
      return it->second;
    nodeIndexes.try_emplace(addr, nextIndex);
    return nextIndex++;
  }
  std::size_t getNodeIndex(const lower::pft::Program &) { return 0; }

private:
  llvm::DenseMap<const void *, std::size_t> nodeIndexes;
  std::size_t nextIndex{1}; // 0 is the root
};

} // namespace

template <typename A, typename T>
static lower::pft::FunctionLikeUnit::FunctionStatement
getFunctionStmt(const T &func) {
  return std::get<parser::Statement<A>>(func.t);
}
template <typename A, typename T>
static lower::pft::ModuleLikeUnit::ModuleStatement getModuleStmt(const T &mod) {
  return std::get<parser::Statement<A>>(mod.t);
}

static const semantics::Symbol *getSymbol(
    std::optional<lower::pft::FunctionLikeUnit::FunctionStatement> &beginStmt) {
  if (!beginStmt)
    return nullptr;

  const auto *symbol = beginStmt->visit(common::visitors{
      [](const parser::Statement<parser::ProgramStmt> &stmt)
          -> const semantics::Symbol * { return stmt.statement.v.symbol; },
      [](const parser::Statement<parser::FunctionStmt> &stmt)
          -> const semantics::Symbol * {
        return std::get<parser::Name>(stmt.statement.t).symbol;
      },
      [](const parser::Statement<parser::SubroutineStmt> &stmt)
          -> const semantics::Symbol * {
        return std::get<parser::Name>(stmt.statement.t).symbol;
      },
      [](const parser::Statement<parser::MpSubprogramStmt> &stmt)
          -> const semantics::Symbol * { return stmt.statement.v.symbol; },
      [](const auto &) -> const semantics::Symbol * {
        llvm_unreachable("unknown FunctionLike beginStmt");
        return nullptr;
      }});
  assert(symbol && "parser::Name must have resolved symbol");
  return symbol;
}

bool Fortran::lower::pft::Evaluation::lowerAsStructured() const {
  return !lowerAsUnstructured();
}

bool Fortran::lower::pft::Evaluation::lowerAsUnstructured() const {
  return isUnstructured || clDisableStructuredFir;
}

lower::pft::FunctionLikeUnit *
Fortran::lower::pft::Evaluation::getOwningProcedure() const {
  return parentVariant.visit(common::visitors{
      [](lower::pft::FunctionLikeUnit &c) { return &c; },
      [&](lower::pft::Evaluation &c) { return c.getOwningProcedure(); },
      [](auto &) -> lower::pft::FunctionLikeUnit * { return nullptr; },
  });
}

namespace {
struct IntervalSet : public llvm::IntervalMap<std::size_t, std::size_t, 16> {
  using IntervalMap::IntervalMap;
  using Allocator = IntervalMap::Allocator;

  // Handles the merging of overlapping intervals correctly, efficiently.
  bool merge(std::size_t lo, std::size_t up) {
    assert(lo < up);
    auto first = lookup(lo);
    auto last = lookup(up);
    if (!first && !last)
      insert(lo, up, 1);
    else if (!first)
      find(up).setStart(lo);
    else if (!last)
      find(lo).setStop(up);
    else
      return false;
    return true;
  }
};
} // namespace

namespace {
/// This helper class is for sorting the symbols in the symbol table. We want
/// the symbols in an order such that a symbol will be visited after those it
/// depends upon. Otherwise this sort is stable and preserves the order of the
/// symbol table, which is sorted by name.
struct SymbolDependenceDepth {
  explicit SymbolDependenceDepth(
      std::vector<std::vector<lower::pft::Variable>> &vars)
      : vars{vars} {}

  // Analyze the equivalence sets. This analysis need not be performed when the
  // scope has no equivalence sets.
  void analyzeAliases(const semantics::Scope &scope) {
    IntervalSet::Allocator allocator;
    IntervalSet intervals(allocator);

    // Collect the offset ranges which have aliasing.
    for (const auto &set : scope.equivalenceSets())
      for (const auto &eqv : set) {
        const auto &sym = eqv.symbol;
        aliasSyms.insert(&sym);
        intervals.merge(sym.offset(), sym.offset() + sym.size() - 1);
      }

    // Create a primary store for each aliased interval.
    adjustSize(1);
    for (auto i = intervals.begin(), end = intervals.end(); i != end; ++i) {
      vars[0].emplace_back(
          lower::pft::Variable::StoreInterval{i.start(),
                                              i.stop() - i.start() + 1},
          /*isGlobal=*/false);
      stores.emplace_back(i.start(), i.stop() + 1);
    }
  }

  // Recursively visit each symbol to determine the height of its dependence on
  // other symbols.
  int analyze(const semantics::Symbol &sym) {
    auto done = seen.insert(&sym);
    if (!done.second)
      return 0;
    if (semantics::IsProcedure(sym)) {
      // TODO: add declaration?
      return 0;
    }
    if (sym.has<semantics::UseDetails>() ||
        sym.has<semantics::HostAssocDetails>() ||
        sym.has<semantics::NamelistDetails>() ||
        sym.has<semantics::MiscDetails>()) {
      // FIXME: do we want to do anything with any of these?
      return 0;
    }

    // Symbol must be something lowering will have to allocate.
    bool global = semantics::IsSaved(sym);
    int depth = 0;
    const auto *symTy = sym.GetType();
    assert(symTy && "symbol must have a type");

    // Make sure an aliasing variable appears after its primary storage.
    if (!aliasSyms.empty())
      if (aliasSyms.find(&sym) != aliasSyms.end())
        depth = std::max(1, depth);

    // check CHARACTER's length
    if (symTy->category() == semantics::DeclTypeSpec::Character)
      if (auto e = symTy->characterTypeSpec().length().GetExplicit())
        for (const auto &s : evaluate::CollectSymbols(*e))
          depth = std::max(analyze(s) + 1, depth);

    if (const auto *details = sym.detailsIf<semantics::ObjectEntityDetails>()) {
      auto doExplicit = [&](const auto &bound) {
        if (bound.isExplicit()) {
          semantics::SomeExpr e{*bound.GetExplicit()};
          for (const auto &s : evaluate::CollectSymbols(e))
            depth = std::max(analyze(s) + 1, depth);
        }
      };
      // handle any symbols in array bound declarations
      for (const auto &subs : details->shape()) {
        doExplicit(subs.lbound());
        doExplicit(subs.ubound());
      }
      // handle any symbols in coarray bound declarations
      for (const auto &subs : details->coshape()) {
        doExplicit(subs.lbound());
        doExplicit(subs.ubound());
      }
      // handle any symbols in initialization expressions
      if (auto e = details->init()) {
        // A PARAMETER may not be marked as implicitly SAVE, so set the flag.
        global = true;
        for (const auto &s : evaluate::CollectSymbols(*e))
          depth = std::max(analyze(s) + 1, depth);
      }
    }
    adjustSize(depth + 1);
    vars[depth].emplace_back(sym, global, depth);
    if (semantics::IsAllocatable(sym))
      vars[depth].back().setHeapAlloc();
    if (semantics::IsPointer(sym))
      vars[depth].back().setPointer();
    if (sym.attrs().test(semantics::Attr::TARGET))
      vars[depth].back().setTarget();

    // If there are alias sets, then link the participating variables to their
    // primary stores when constructing the new variable on the list.
    if (!aliasSyms.empty())
      if (aliasSyms.find(&sym) != aliasSyms.end()) {
        if (global)
          llvm::report_fatal_error("TODO: EQUIVALENCE on global");
        // Expect the total number of EQUIVALENCE sets to be small for a typical
        // Fortran program.
        auto findStore = [&](std::size_t off) -> std::size_t {
          for (auto v : stores) {
            auto bot = std::get<0>(v);
            if (off >= bot && off < std::get<1>(v))
              return bot;
          }
          llvm_unreachable("the store must be present");
        };
        vars[depth].back().setAlias(findStore(sym.offset()));
      }
    return depth;
  }

  // Save the final list of symbols as a single vector and free the rest.
  void finalize() {
    for (int i = 1, end = vars.size(); i < end; ++i)
      vars[0].insert(vars[0].end(), vars[i].begin(), vars[i].end());
    vars.resize(1);
  }

private:
  // Make sure the table is of appropriate size.
  void adjustSize(std::size_t size) {
    if (vars.size() < size)
      vars.resize(size);
  }

  llvm::SmallSet<const semantics::Symbol *, 32> seen;
  std::vector<std::vector<lower::pft::Variable>> &vars;
  llvm::SmallSet<const semantics::Symbol *, 32> aliasSyms;
  std::vector<std::tuple<std::size_t, std::size_t>> stores;
};
} // namespace

void Fortran::lower::pft::FunctionLikeUnit::processSymbolTable(
    const semantics::Scope &scope) {
  SymbolDependenceDepth sdd{varList};
  if (!scope.equivalenceSets().empty())
    sdd.analyzeAliases(scope);
  for (const auto &iter : scope)
    sdd.analyze(iter.second.get());
  sdd.finalize();
}

Fortran::lower::pft::FunctionLikeUnit::FunctionLikeUnit(
    const parser::MainProgram &func, const lower::pft::ParentVariant &parent,
    const semantics::SemanticsContext &semanticsContext)
    : ProgramUnit{func, parent}, endStmt{
                                     getFunctionStmt<parser::EndProgramStmt>(
                                         func)} {
  const auto &programStmt{
      std::get<std::optional<parser::Statement<parser::ProgramStmt>>>(func.t)};
  if (programStmt.has_value()) {
    beginStmt = programStmt.value();
    auto symbol = getSymbol(beginStmt);
    entryPointList[0].first = symbol;
    processSymbolTable(*symbol->scope());
  } else {
    processSymbolTable(semanticsContext.FindScope(
        std::get<parser::Statement<parser::EndProgramStmt>>(func.t).source));
  }
}

Fortran::lower::pft::FunctionLikeUnit::FunctionLikeUnit(
    const parser::FunctionSubprogram &func,
    const lower::pft::ParentVariant &parent,
    const semantics::SemanticsContext &)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<parser::FunctionStmt>(func)},
      endStmt{getFunctionStmt<parser::EndFunctionStmt>(func)} {
  auto symbol = getSymbol(beginStmt);
  entryPointList[0].first = symbol;
  processSymbolTable(*symbol->scope());
}

Fortran::lower::pft::FunctionLikeUnit::FunctionLikeUnit(
    const parser::SubroutineSubprogram &func,
    const lower::pft::ParentVariant &parent,
    const semantics::SemanticsContext &)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<parser::SubroutineStmt>(func)},
      endStmt{getFunctionStmt<parser::EndSubroutineStmt>(func)} {
  auto symbol = getSymbol(beginStmt);
  entryPointList[0].first = symbol;
  processSymbolTable(*symbol->scope());
}

Fortran::lower::pft::FunctionLikeUnit::FunctionLikeUnit(
    const parser::SeparateModuleSubprogram &func,
    const lower::pft::ParentVariant &parent,
    const semantics::SemanticsContext &)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<parser::MpSubprogramStmt>(func)},
      endStmt{getFunctionStmt<parser::EndMpSubprogramStmt>(func)} {
  auto symbol = getSymbol(beginStmt);
  entryPointList[0].first = symbol;
  processSymbolTable(*symbol->scope());
}

Fortran::lower::pft::ModuleLikeUnit::ModuleLikeUnit(
    const parser::Module &m, const lower::pft::ParentVariant &parent)
    : ProgramUnit{m, parent}, beginStmt{getModuleStmt<parser::ModuleStmt>(m)},
      endStmt{getModuleStmt<parser::EndModuleStmt>(m)} {}

Fortran::lower::pft::ModuleLikeUnit::ModuleLikeUnit(
    const parser::Submodule &m, const lower::pft::ParentVariant &parent)
    : ProgramUnit{m, parent}, beginStmt{getModuleStmt<parser::SubmoduleStmt>(
                                  m)},
      endStmt{getModuleStmt<parser::EndSubmoduleStmt>(m)} {}

Fortran::lower::pft::BlockDataUnit::BlockDataUnit(
    const parser::BlockData &bd, const lower::pft::ParentVariant &parent,
    const semantics::SemanticsContext &semanticsContext)
    : ProgramUnit{bd, parent},
      symTab{semanticsContext.FindScope(
          std::get<parser::Statement<parser::EndBlockDataStmt>>(bd.t).source)} {
}

std::unique_ptr<lower::pft::Program>
Fortran::lower::createPFT(const parser::Program &root,
                          const semantics::SemanticsContext &semanticsContext) {
  PFTBuilder walker(semanticsContext);
  Walk(root, walker);
  return walker.result();
}

void Fortran::lower::dumpPFT(llvm::raw_ostream &outputStream,
                             const lower::pft::Program &pft) {
  PFTDumper{}.dumpPFT(outputStream, pft);
}

void Fortran::lower::pft::Program::dump() const {
  dumpPFT(llvm::errs(), *this);
}

void Fortran::lower::pft::Evaluation::dump() const {
  PFTDumper{}.dumpEvaluation(llvm::errs(), *this);
}

void Fortran::lower::pft::Variable::dump() const {
  if (auto *sym = std::get_if<const Fortran::semantics::Symbol *>(&u))
    llvm::errs() << "symbol: " << (*sym)->name();
  else if (auto *store = std::get_if<StoreInterval>(&u))
    llvm::errs() << "interval[" << std::get<0>(*store) << ", "
                 << std::get<1>(*store) << "]:";
  else
    llvm_unreachable("not a Variable");
  
  llvm::errs() << " depth: " << depth;
  if (global)
    llvm::errs() << ", global";
  if (heapAlloc)
    llvm::errs() << ", allocatable";
  if (pointer)
    llvm::errs() << ", pointer";
  if (target)
    llvm::errs() << ", target";
  if (aliasee)
    llvm::errs() << ", equivalence(" << aliasOffset << ")";
  llvm::errs() << '\n';
}

void Fortran::lower::pft::FunctionLikeUnit::dump() const {
  PFTDumper{}.dumpFunctionLikeUnit(llvm::errs(), *this);
}

void Fortran::lower::pft::ModuleLikeUnit::dump() const {
  PFTDumper{}.dumpModuleLikeUnit(llvm::errs(), *this);
}

/// The BlockDataUnit dump is just the associated symbol table.
void Fortran::lower::pft::BlockDataUnit::dump() const {
  llvm::errs() << "block data {\n" << symTab << "\n}\n";
}
