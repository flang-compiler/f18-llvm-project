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
#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<bool> clDisableStructuredFir(
    "no-structured-fir", llvm::cl::desc("disable generation of structured FIR"),
    llvm::cl::init(false), llvm::cl::Hidden);

namespace Fortran::lower {
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
  PFTBuilder(const Fortran::semantics::SemanticsContext &semanticsContext)
      : pgm{std::make_unique<pft::Program>()}, parentVariantStack{*pgm.get()},
        semanticsContext{semanticsContext} {}

  /// Get the result
  std::unique_ptr<pft::Program> result() { return std::move(pgm); }

  template <typename A>
  constexpr bool Pre(const A &a) {
    if constexpr (pft::isFunctionLike<A>) {
      return enterFunction(a, semanticsContext);
    } else if constexpr (pft::isConstruct<A>) {
      return enterConstruct(a);
    } else if constexpr (UnwrapStmt<A>::isStmt) {
      using T = typename UnwrapStmt<A>::Type;
      // Node "a" being visited has one of the following types:
      // Statement<T>, Statement<Indirection<T>, UnlabeledStatement<T>,
      // or UnlabeledStatement<Indirection<T>>
      auto stmt{UnwrapStmt<A>(a)};
      if constexpr (pft::isConstructStmt<T> || pft::isOtherStmt<T>) {
        addEvaluation(pft::Evaluation{stmt.unwrapped, parentVariantStack.back(),
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
    if constexpr (pft::isFunctionLike<A>) {
      exitFunction();
    } else if constexpr (pft::isConstruct<A>) {
      exitConstruct();
    }
  }

  // Module like
  bool Pre(const parser::Module &node) { return enterModule(node); }
  bool Pre(const parser::Submodule &node) { return enterModule(node); }

  void Post(const parser::Module &) { exitModule(); }
  void Post(const parser::Submodule &) { exitModule(); }

  // Block data
  bool Pre(const parser::BlockData &node) {
    addUnit(pft::BlockDataUnit{node, parentVariantStack.back()});
    return false;
  }

  // Get rid of production wrapper
  bool Pre(const parser::UnlabeledStatement<parser::ForallAssignmentStmt>
               &statement) {
    addEvaluation(std::visit(
        [&](const auto &x) {
          return pft::Evaluation{
              x, parentVariantStack.back(), statement.source, {}};
        },
        statement.statement.u));
    return false;
  }
  bool Pre(const parser::Statement<parser::ForallAssignmentStmt> &statement) {
    addEvaluation(std::visit(
        [&](const auto &x) {
          return pft::Evaluation{x, parentVariantStack.back(), statement.source,
                                 statement.label};
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
              addEvaluation(pft::Evaluation{stmt.statement,
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
    auto &unit = addUnit(pft::ModuleLikeUnit{func, parentVariantStack.back()});
    functionList = &unit.nestedFunctions;
    parentVariantStack.emplace_back(unit);
    return true;
  }

  void exitModule() {
    parentVariantStack.pop_back();
    resetFunctionList();
  }

  /// Initialize a new function-like unit and make it the builder's focus.
  template <typename A>
  bool
  enterFunction(const A &func,
                const Fortran::semantics::SemanticsContext &semanticsContext) {
    auto &unit = addFunction(pft::FunctionLikeUnit{
        func, parentVariantStack.back(), semanticsContext});
    labelEvaluationMap = &unit.labelEvaluationMap;
    assignSymbolLabelMap = &unit.assignSymbolLabelMap;
    functionList = &unit.nestedFunctions;
    pushEvaluationList(&unit.evaluationList);
    parentVariantStack.emplace_back(unit);
    return true;
  }

  void exitFunction() {
    // Guarantee that there is a branch target after the last user statement.
    static const parser::ContinueStmt endTarget{};
    addEvaluation(
        pft::Evaluation{endTarget, parentVariantStack.back(), {}, {}});
    lastLexicalEvaluation = nullptr;
    analyzeBranches(nullptr, *evaluationListStack.back()); // add branch links
    popEvaluationList();
    labelEvaluationMap = nullptr;
    assignSymbolLabelMap = nullptr;
    parentVariantStack.pop_back();
    resetFunctionList();
  }

  /// Initialize a new construct and make it the builder's focus.
  template <typename A>
  bool enterConstruct(const A &construct) {
    auto &eval =
        addEvaluation(pft::Evaluation{construct, parentVariantStack.back()});
    eval.evaluationList.reset(new pft::EvaluationList);
    pushEvaluationList(eval.evaluationList.get());
    parentVariantStack.emplace_back(eval);
    constructStack.emplace_back(&eval);
    return true;
  }

  void exitConstruct() {
    popEvaluationList();
    parentVariantStack.pop_back();
    constructStack.pop_back();
  }

  /// Reset functionList to an enclosing function's functionList.
  void resetFunctionList() {
    if (!parentVariantStack.empty()) {
      std::visit(common::visitors{
                     [&](pft::FunctionLikeUnit *p) {
                       functionList = &p->nestedFunctions;
                     },
                     [&](pft::ModuleLikeUnit *p) {
                       functionList = &p->nestedFunctions;
                     },
                     [&](auto *) { functionList = nullptr; },
                 },
                 parentVariantStack.back().p);
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
  pft::Evaluation makeEvaluationAction(const parser::ActionStmt &statement,
                                       parser::CharBlock position,
                                       std::optional<parser::Label> label) {
    return std::visit(
        common::visitors{
            [&](const auto &x) {
              return pft::Evaluation{removeIndirection(x),
                                     parentVariantStack.back(), position,
                                     label};
            },
        },
        statement.u);
  }

  /// Append an Evaluation to the end of the current list.
  pft::Evaluation &addEvaluation(pft::Evaluation &&eval) {
    assert(functionList && "not in a function");
    assert(evaluationListStack.size() > 0);
    if (constructStack.size() > 0) {
      eval.parentConstruct = constructStack.back();
    }
    evaluationListStack.back()->emplace_back(std::move(eval));
    pft::Evaluation *p = &evaluationListStack.back()->back();
    if (p->isActionStmt() || p->isConstructStmt()) {
      if (lastLexicalEvaluation) {
        lastLexicalEvaluation->lexicalSuccessor = p;
        p->printIndex = lastLexicalEvaluation->printIndex + 1;
      } else {
        p->printIndex = 1;
      }
      lastLexicalEvaluation = p;
    }
    if (p->label.has_value()) {
      labelEvaluationMap->try_emplace(*p->label, p);
    }
    return evaluationListStack.back()->back();
  }

  /// push a new list on the stack of Evaluation lists
  void pushEvaluationList(pft::EvaluationList *eval) {
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
  void analyzeIoBranches(pft::Evaluation &eval, const A &stmt) {
    auto processIfLabel{[&](const auto &specs) {
      using LableNodes =
          std::tuple<parser::ErrLabel, parser::EorLabel, parser::EndLabel>;
      for (const auto &spec : specs) {
        const auto *label = std::visit(
            [](const auto &label) -> const parser::Label * {
              using B = std::decay_t<decltype(label)>;
              if constexpr (common::HasMember<B, LableNodes>) {
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
      processIfLabel(std::get<std::list<parser::InquireSpec>>(stmt.u));
    } else if constexpr (common::HasMember<A, OtherIOStmts>) {
      processIfLabel(stmt.v);
    } else {
      // Always crash if this is instantiated
      static_assert(!std::is_same_v<A, parser::ReadStmt>,
                    "Unexpected IO statement");
    }
  }

  /// Set the exit of a construct, possibly from multiple enclosing constructs.
  void setConstructExit(pft::Evaluation &eval) {
    eval.constructExit = eval.evaluationList->back().lexicalSuccessor;
    if (eval.constructExit && eval.constructExit->isNopConstructStmt()) {
      eval.constructExit = eval.constructExit->parentConstruct->constructExit;
    }
    assert(eval.constructExit && "missing construct exit");
  }

  void markBranchTarget(pft::Evaluation &sourceEvaluation,
                        pft::Evaluation &targetEvaluation) {
    sourceEvaluation.isUnstructured = true;
    if (!sourceEvaluation.controlSuccessor) {
      sourceEvaluation.controlSuccessor = &targetEvaluation;
    }
    targetEvaluation.isNewBlock = true;
  }
  void markBranchTarget(pft::Evaluation &sourceEvaluation,
                        parser::Label label) {
    assert(label && "missing branch target label");
    pft::Evaluation *targetEvaluation{labelEvaluationMap->find(label)->second};
    assert(targetEvaluation && "missing branch target evaluation");
    markBranchTarget(sourceEvaluation, *targetEvaluation);
  }

  /// Return the first non-nop successor of an evaluation, possibly exiting
  /// from one or more enclosing constructs.
  pft::Evaluation *exitSuccessor(pft::Evaluation &eval) {
    pft::Evaluation *successor{eval.lexicalSuccessor};
    if (successor && successor->isNopConstructStmt()) {
      successor = successor->parentConstruct->constructExit;
    }
    assert(successor && "missing exit successor");
    return successor;
  }

  /// Mark the exit successor of an Evaluation as a new block.
  void markExitSuccessorAsNewBlock(pft::Evaluation &eval) {
    exitSuccessor(eval)->isNewBlock = true;
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
      if (auto name{std::get<0>(stmt.t)}) {
        return name->ToString();
      }
    }
    return {};
  }

  /// \p parentConstruct can be null if this statement is at the highest
  /// level of a program.
  template <typename A>
  void insertConstructName(const A &stmt, pft::Evaluation *parentConstruct) {
    std::string name{getConstructName(stmt)};
    if (!name.empty()) {
      constructNameMap[name] = parentConstruct;
    }
  }

  /// Insert branch links for a list of Evaluations.
  /// \p parentConstruct can be null if the evaluationList contains the
  /// top-level statements of a program.
  void analyzeBranches(pft::Evaluation *parentConstruct,
                       std::list<pft::Evaluation> &evaluationList) {
    pft::Evaluation *lastConstructStmtEvaluation{nullptr};
    pft::Evaluation *lastIfStmtEvaluation{nullptr};
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
            pft::Evaluation *construct{name.empty() ? doConstructStack.back()
                                                    : constructNameMap[name]};
            assert(construct && "missing CYCLE construct");
            markBranchTarget(eval, construct->evaluationList->back());
          },
          [&](const parser::ExitStmt &s) {
            std::string name{getConstructName(s)};
            pft::Evaluation *construct{name.empty() ? doConstructStack.back()
                                                    : constructNameMap[name]};
            assert(construct && "missing EXIT construct");
            markBranchTarget(eval, *construct->constructExit);
          },
          [&](const parser::GotoStmt &s) { markBranchTarget(eval, s.v); },
          [&](const parser::IfStmt &) { lastIfStmtEvaluation = &eval; },
          [&](const parser::ReturnStmt &) { eval.isUnstructured = true; },
          [&](const parser::StopStmt &) { eval.isUnstructured = true; },
          [&](const parser::ComputedGotoStmt &s) {
            for (auto &label : std::get<std::list<parser::Label>>(s.t)) {
              markBranchTarget(eval, label);
            }
          },
          [&](const parser::ArithmeticIfStmt &s) {
            markBranchTarget(eval, std::get<1>(s.t));
            markBranchTarget(eval, std::get<2>(s.t));
            markBranchTarget(eval, std::get<3>(s.t));
          },
          [&](const parser::AssignStmt &s) { // legacy label assignment
            auto &label = std::get<parser::Label>(s.t);
            const auto *sym = std::get<parser::Name>(s.t).symbol;
            assert(sym && "missing AssignStmt symbol");
            pft::Evaluation *t{labelEvaluationMap->find(label)->second};
            if (!t->isA<parser::FormatStmt>()) {
              markBranchTarget(eval, label);
            }
            auto iter = assignSymbolLabelMap->find(*sym);
            if (iter == assignSymbolLabelMap->end()) {
              pft::LabelSet labelSet{};
              labelSet.insert(label);
              assignSymbolLabelMap->try_emplace(*sym, labelSet);
            } else {
              iter->second.insert(label);
            }
          },
          [&](const parser::AssignedGotoStmt &) {
            // Although this statement is a branch, it doesn't have any
            // explicit control successors.  So the code at the end of the
            // loop won't mark the exit successor.  Do that here.
            markExitSuccessorAsNewBlock(eval);
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
            eval.lexicalSuccessor->isNewBlock = true;
            lastConstructStmtEvaluation = &eval;
          },
          [&](const parser::CaseStmt &) {
            eval.isNewBlock = true;
            lastConstructStmtEvaluation->controlSuccessor = &eval;
            lastConstructStmtEvaluation = &eval;
          },
          [&](const parser::EndSelectStmt &) {
            eval.lexicalSuccessor->isNewBlock = true;
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
            eval.lexicalSuccessor->isNewBlock = true;
            eval.controlSuccessor = &evaluationList.back();
            if (std::holds_alternative<parser::ScalarLogicalExpr>(control->u)) {
              eval.isUnstructured = true; // while loop
            }
            // Defer additional processing for an unstructured concurrent loop
            // to the EndDoStmt, when the loop is known to be unstructured.
          },
          [&](const parser::EndDoStmt &) {
            pft::Evaluation &doEval{evaluationList.front()};
            eval.controlSuccessor = &doEval;
            doConstructStack.pop_back();
            if (parentConstruct->lowerAsStructured()) {
              return;
            }
            // Now that the loop is known to be unstructured, finish concurrent
            // loop processing, using NonLabelDoStmt information.
            parentConstruct->constructExit->isNewBlock = true;
            const auto &doStmt{doEval.getIf<parser::NonLabelDoStmt>()};
            assert(doStmt && "missing NonLabelDoStmt");
            auto &control{
                std::get<std::optional<parser::LoopControl>>(doStmt->t)};
            if (!control.has_value()) {
              return; // infinite loop
            }
            const auto *concurrent{
                std::get_if<parser::LoopControl::Concurrent>(&control->u)};
            if (!concurrent) {
              return;
            }
            // Unstructured concurrent loop.  NonLabelDoStmt code accounts
            // for one concurrent loop dimension.  Reserve preheader,
            // header, and latch blocks for the remaining dimensions, and
            // one block for a mask expression.
            const auto &header{
                std::get<parser::ConcurrentHeader>(concurrent->t)};
            auto dims{std::get<std::list<parser::ConcurrentControl>>(header.t)
                          .size()};
            for (; dims > 1; --dims) {
              doEval.localBlocks.emplace_back(nullptr); // preheader
              doEval.localBlocks.emplace_back(nullptr); // header
              eval.localBlocks.emplace_back(nullptr);   // latch
            }
            if (std::get<std::optional<parser::ScalarLogicalExpr>>(header.t)) {
              doEval.localBlocks.emplace_back(nullptr); // mask
            }
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
            if (parentConstruct->lowerAsUnstructured()) {
              parentConstruct->constructExit->isNewBlock = true;
            }
            if (lastConstructStmtEvaluation) {
              lastConstructStmtEvaluation->controlSuccessor =
                  parentConstruct->constructExit;
              lastConstructStmtEvaluation = nullptr;
            }
          },
          [&](const parser::SelectRankStmt &s) {
            insertConstructName(s, parentConstruct);
            eval.lexicalSuccessor->isNewBlock = true;
          },
          [&](const parser::SelectRankCaseStmt &) { eval.isNewBlock = true; },
          [&](const parser::SelectTypeStmt &s) {
            insertConstructName(s, parentConstruct);
            eval.lexicalSuccessor->isNewBlock = true;
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
      if (eval.evaluationList) {
        analyzeBranches(&eval, *eval.evaluationList);
      }

      // Insert branch links for an unstructured IF statement.
      if (lastIfStmtEvaluation && lastIfStmtEvaluation != &eval) {
        // eval is the action substatement of an IfStmt.
        if (eval.lowerAsUnstructured()) {
          eval.isNewBlock = true;
          markExitSuccessorAsNewBlock(eval);
          lastIfStmtEvaluation->isUnstructured = true;
        }
        lastIfStmtEvaluation->controlSuccessor = exitSuccessor(eval);
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

      // The lexical successor of a branch starts a new block.
      if (eval.controlSuccessor && eval.isActionStmt() &&
          eval.lowerAsUnstructured()) {
        markExitSuccessorAsNewBlock(eval);
      }
    }
  }

  std::unique_ptr<pft::Program> pgm;
  std::vector<pft::ParentVariant> parentVariantStack;
  const Fortran::semantics::SemanticsContext &semanticsContext;

  /// functionList points to the internal or module procedure function list
  /// of a FunctionLikeUnit or a ModuleLikeUnit.  It may be null.
  std::list<pft::FunctionLikeUnit> *functionList{nullptr};
  std::vector<pft::Evaluation *> constructStack{};
  std::vector<pft::Evaluation *> doConstructStack{};
  /// evaluationListStack is the current nested construct evaluationList state.
  std::vector<pft::EvaluationList *> evaluationListStack{};
  llvm::DenseMap<parser::Label, pft::Evaluation *> *labelEvaluationMap{nullptr};
  pft::SymbolLabelMap *assignSymbolLabelMap{nullptr};
  std::map<std::string, pft::Evaluation *> constructNameMap{};
  pft::Evaluation *lastLexicalEvaluation{nullptr};
};

class PFTDumper {
public:
  void dumpPFT(llvm::raw_ostream &outputStream, pft::Program &pft) {
    for (auto &unit : pft.getUnits()) {
      std::visit(common::visitors{
                     [&](pft::BlockDataUnit &unit) {
                       outputStream << getNodeIndex(unit) << " ";
                       outputStream << "BlockData: ";
                       outputStream << "\nEndBlockData\n\n";
                     },
                     [&](pft::FunctionLikeUnit &func) {
                       dumpFunctionLikeUnit(outputStream, func);
                     },
                     [&](pft::ModuleLikeUnit &unit) {
                       dumpModuleLikeUnit(outputStream, unit);
                     },
                 },
                 unit);
    }
  }

  llvm::StringRef evaluationName(pft::Evaluation &eval) {
    return eval.visit(common::visitors{
        [](const auto &parseTreeNode) {
          return parser::ParseTreeDumper::GetNodeName(parseTreeNode);
        },
    });
  }

  void dumpEvaluationList(llvm::raw_ostream &outputStream,
                          pft::EvaluationList &evaluationList, int indent = 1) {
    static const std::string white{"                                      ++"};
    std::string indentString{white.substr(0, indent * 2)};
    for (pft::Evaluation &eval : evaluationList) {
      llvm::StringRef name{evaluationName(eval)};
      std::string bang{eval.isUnstructured ? "!" : ""};
      if (eval.isConstruct()) {
        outputStream << indentString << "<<" << name << bang << ">>";
        if (eval.constructExit) {
          outputStream << " -> " << eval.constructExit->printIndex;
        }
        outputStream << '\n';
        dumpEvaluationList(outputStream, *eval.evaluationList, indent + 1);
        outputStream << indentString << "<<End " << name << bang << ">>\n";
        continue;
      }
      outputStream << indentString;
      if (eval.printIndex) {
        outputStream << eval.printIndex << ' ';
      }
      if (eval.isNewBlock) {
        outputStream << '^';
      }
      if (eval.localBlocks.size()) {
        outputStream << '*';
      }
      outputStream << name << bang;
      if (eval.isActionStmt() || eval.isConstructStmt()) {
        if (eval.controlSuccessor) {
          outputStream << " -> " << eval.controlSuccessor->printIndex;
        }
      }
      if (eval.position.size()) {
        outputStream << ": " << eval.position.ToString();
      }
      outputStream << '\n';
    }
  }

  void dumpFunctionLikeUnit(llvm::raw_ostream &outputStream,
                            pft::FunctionLikeUnit &functionLikeUnit) {
    outputStream << getNodeIndex(functionLikeUnit) << " ";
    llvm::StringRef unitKind{};
    std::string name{};
    std::string header{};
    if (functionLikeUnit.beginStmt) {
      functionLikeUnit.beginStmt->visit(common::visitors{
          [&](const parser::Statement<parser::ProgramStmt> &statement) {
            unitKind = "Program";
            name = statement.statement.v.ToString();
          },
          [&](const parser::Statement<parser::FunctionStmt> &statement) {
            unitKind = "Function";
            name = std::get<parser::Name>(statement.statement.t).ToString();
            header = statement.source.ToString();
          },
          [&](const parser::Statement<parser::SubroutineStmt> &statement) {
            unitKind = "Subroutine";
            name = std::get<parser::Name>(statement.statement.t).ToString();
            header = statement.source.ToString();
          },
          [&](const parser::Statement<parser::MpSubprogramStmt> &statement) {
            unitKind = "MpSubprogram";
            name = statement.statement.v.ToString();
            header = statement.source.ToString();
          },
          [&](const auto &) {},
      });
    } else {
      unitKind = "Program";
      name = "<anonymous>";
    }
    outputStream << unitKind << ' ' << name;
    if (header.size())
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
                          pft::ModuleLikeUnit &moduleLikeUnit) {
    outputStream << getNodeIndex(moduleLikeUnit) << " ";
    outputStream << "ModuleLike: ";
    outputStream << "\nContains\n";
    for (auto &func : moduleLikeUnit.nestedFunctions)
      dumpFunctionLikeUnit(outputStream, func);
    outputStream << "EndContains\nEndModuleLike\n\n";
  }

  template <typename T>
  std::size_t getNodeIndex(const T &node) {
    auto addr{static_cast<const void *>(&node)};
    auto it{nodeIndexes.find(addr)};
    if (it != nodeIndexes.end()) {
      return it->second;
    }
    nodeIndexes.try_emplace(addr, nextIndex);
    return nextIndex++;
  }
  std::size_t getNodeIndex(const pft::Program &) { return 0; }

private:
  llvm::DenseMap<const void *, std::size_t> nodeIndexes;
  std::size_t nextIndex{1}; // 0 is the root
};

} // namespace
} // namespace Fortran::lower

template <typename A, typename T>
static Fortran::lower::pft::FunctionLikeUnit::FunctionStatement
getFunctionStmt(const T &func) {
  return std::get<Fortran::parser::Statement<A>>(func.t);
}
template <typename A, typename T>
static Fortran::lower::pft::ModuleLikeUnit::ModuleStatement
getModuleStmt(const T &mod) {
  return std::get<Fortran::parser::Statement<A>>(mod.t);
}

static const Fortran::semantics::Symbol *getSymbol(
    std::optional<Fortran::lower::pft::FunctionLikeUnit::FunctionStatement>
        &beginStmt) {
  if (!beginStmt)
    return nullptr;

  const auto *symbol = beginStmt->visit(Fortran::common::visitors{
      [](const Fortran::parser::Statement<Fortran::parser::ProgramStmt> &stmt)
          -> const Fortran::semantics::Symbol * {
        return stmt.statement.v.symbol;
      },
      [](const Fortran::parser::Statement<Fortran::parser::FunctionStmt> &stmt)
          -> const Fortran::semantics::Symbol * {
        return std::get<Fortran::parser::Name>(stmt.statement.t).symbol;
      },
      [](const Fortran::parser::Statement<Fortran::parser::SubroutineStmt>
             &stmt) -> const Fortran::semantics::Symbol * {
        return std::get<Fortran::parser::Name>(stmt.statement.t).symbol;
      },
      [](const Fortran::parser::Statement<Fortran::parser::MpSubprogramStmt>
             &stmt) -> const Fortran::semantics::Symbol * {
        return stmt.statement.v.symbol;
      },
      [](const auto &) -> const Fortran::semantics::Symbol * {
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

Fortran::lower::pft::FunctionLikeUnit *
Fortran::lower::pft::Evaluation::getOwningProcedure() const {
  return std::visit(
      Fortran::common::visitors{
          [](Fortran::lower::pft::FunctionLikeUnit *c) { return c; },
          [&](Fortran::lower::pft::Evaluation *c) {
            return c->getOwningProcedure();
          },
          [](auto *) -> Fortran::lower::pft::FunctionLikeUnit * {
            return nullptr;
          },
      },
      parentVariant.p);
}

namespace {
/// This helper class is for sorting the symbols in the symbol table. We want
/// the symbols in an order such that a symbol will be visited after those it
/// depends upon. Otherwise this sort is stable and preserves the order of the
/// symbol table, which is sorted by name.
struct SymbolDependenceDepth {
  explicit SymbolDependenceDepth(
      std::vector<std::vector<Fortran::lower::pft::Variable>> &vars)
      : vars{vars} {}

  // Recursively visit each symbol to determine the height of its dependence on
  // other symbols.
  int analyze(const Fortran::semantics::Symbol &sym) {
    auto done = seen.insert(&sym);
    if (!done.second)
      return 0;
    if (Fortran::semantics::IsDummy(sym)) {
      // Trivial base case. Add to the list in case it's pass-by-value.
      adjustSize(1);
      vars[0].emplace_back(sym);
      return 0;
    }
    if (Fortran::semantics::IsProcedure(sym)) {
      // TODO: add declaration?
      return 0;
    }
    if (sym.has<Fortran::semantics::UseDetails>() ||
        sym.has<Fortran::semantics::HostAssocDetails>() ||
        sym.has<Fortran::semantics::NamelistDetails>() ||
        sym.has<Fortran::semantics::MiscDetails>()) {
      // FIXME: do we want to do anything with any of these?
      return 0;
    }

    // Symbol must be something lowering will have to allocate.
    bool global = Fortran::semantics::IsSaved(sym);
    int depth = 0;
    const auto *symTy = sym.GetType();
    assert(symTy && "symbol must have a type");

    // check CHARACTER's length
    if (symTy->category() == Fortran::semantics::DeclTypeSpec::Character)
      if (auto e = symTy->characterTypeSpec().length().GetExplicit())
        for (const auto &s : Fortran::evaluate::CollectSymbols(*e))
          depth = std::max(analyze(s) + 1, depth);

    if (const auto *details =
            sym.detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
      auto doExplicit = [&](const auto &bound) {
        if (bound.isExplicit()) {
          Fortran::semantics::SomeExpr e{*bound.GetExplicit()};
          for (const auto &s : Fortran::evaluate::CollectSymbols(e))
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
        assert(global && "should have been marked implicitly SAVE");
        for (const auto &s : Fortran::evaluate::CollectSymbols(*e))
          depth = std::max(analyze(s) + 1, depth);
      }
    }
    adjustSize(depth + 1);
    vars[depth].emplace_back(sym, global, depth);
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

  llvm::SmallSet<const Fortran::semantics::Symbol *, 32> seen;
  std::vector<std::vector<Fortran::lower::pft::Variable>> &vars;
};
} // namespace

void Fortran::lower::pft::FunctionLikeUnit::processSymbolTable(
    const Fortran::semantics::Scope &scope) {
  SymbolDependenceDepth sdd{varList};
  for (const auto &iter : scope) {
    sdd.analyze(iter.second.get());
    // llvm::outs() << iter.second.get() << '\n';
  }
  sdd.finalize();
}

Fortran::lower::pft::FunctionLikeUnit::FunctionLikeUnit(
    const Fortran::parser::MainProgram &func,
    const Fortran::lower::pft::ParentVariant &parent,
    const Fortran::semantics::SemanticsContext &semanticsContext)
    : ProgramUnit{func, parent},
      endStmt{getFunctionStmt<Fortran::parser::EndProgramStmt>(func)} {
  const auto &ps{std::get<
      std::optional<Fortran::parser::Statement<Fortran::parser::ProgramStmt>>>(
      func.t)};
  if (ps.has_value()) {
    beginStmt = ps.value();
    symbol = getSymbol(beginStmt);
    processSymbolTable(*symbol->scope());
  } else {
    processSymbolTable(semanticsContext.FindScope(
        std::get<Fortran::parser::Statement<Fortran::parser::EndProgramStmt>>(
            func.t)
            .source));
  }
}

Fortran::lower::pft::FunctionLikeUnit::FunctionLikeUnit(
    const Fortran::parser::FunctionSubprogram &func,
    const Fortran::lower::pft::ParentVariant &parent,
    const Fortran::semantics::SemanticsContext &)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<Fortran::parser::FunctionStmt>(func)},
      endStmt{getFunctionStmt<Fortran::parser::EndFunctionStmt>(func)},
      symbol{getSymbol(beginStmt)} {
  processSymbolTable(*symbol->scope());
}

Fortran::lower::pft::FunctionLikeUnit::FunctionLikeUnit(
    const Fortran::parser::SubroutineSubprogram &func,
    const Fortran::lower::pft::ParentVariant &parent,
    const Fortran::semantics::SemanticsContext &)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<Fortran::parser::SubroutineStmt>(func)},
      endStmt{getFunctionStmt<Fortran::parser::EndSubroutineStmt>(func)},
      symbol{getSymbol(beginStmt)} {
  processSymbolTable(*symbol->scope());
}

Fortran::lower::pft::FunctionLikeUnit::FunctionLikeUnit(
    const parser::SeparateModuleSubprogram &func,
    const Fortran::lower::pft::ParentVariant &parent,
    const Fortran::semantics::SemanticsContext &)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<Fortran::parser::MpSubprogramStmt>(func)},
      endStmt{getFunctionStmt<Fortran::parser::EndMpSubprogramStmt>(func)},
      symbol{getSymbol(beginStmt)} {
  processSymbolTable(*symbol->scope());
}

Fortran::lower::pft::ModuleLikeUnit::ModuleLikeUnit(
    const parser::Module &m, const Fortran::lower::pft::ParentVariant &parent)
    : ProgramUnit{m, parent},
      beginStmt{getModuleStmt<Fortran::parser::ModuleStmt>(m)},
      endStmt{getModuleStmt<Fortran::parser::EndModuleStmt>(m)} {}

Fortran::lower::pft::ModuleLikeUnit::ModuleLikeUnit(
    const Fortran::parser::Submodule &m,
    const Fortran::lower::pft::ParentVariant &parent)
    : ProgramUnit{m, parent},
      beginStmt{getModuleStmt<Fortran::parser::SubmoduleStmt>(m)},
      endStmt{getModuleStmt<Fortran::parser::EndSubmoduleStmt>(m)} {}

Fortran::lower::pft::BlockDataUnit::BlockDataUnit(
    const Fortran::parser::BlockData &bd,
    const Fortran::lower::pft::ParentVariant &parent)
    : ProgramUnit{bd, parent} {}

std::unique_ptr<Fortran::lower::pft::Program> Fortran::lower::createPFT(
    const Fortran::parser::Program &root,
    const Fortran::semantics::SemanticsContext &semanticsContext) {
  PFTBuilder walker(semanticsContext);
  Walk(root, walker);
  return walker.result();
}

void Fortran::lower::dumpPFT(llvm::raw_ostream &outputStream,
                             Fortran::lower::pft::Program &pft) {
  PFTDumper{}.dumpPFT(outputStream, pft);
}

void Fortran::lower::pft::Program::dump() { dumpPFT(llvm::errs(), *this); }
