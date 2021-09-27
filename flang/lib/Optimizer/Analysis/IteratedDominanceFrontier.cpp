//===- IteratedDominanceFrontier.cpp - Compute IDF ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compute iterated dominance frontiers using a linear time algorithm.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/IteratedDominanceFrontier.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include <queue>
#include <utility>

namespace fir {

template <class NodeTy, bool IsPostDom>
void IDFCalculator<NodeTy, IsPostDom>::calculate(
    llvm::SmallVectorImpl<NodeTy *> &phiBlocks) {
  // Use a priority queue keyed on dominator tree level so that inserted nodes
  // are handled from the bottom of the dominator tree upwards. We also augment
  // the level with a DFS number to ensure that the blocks are ordered in a
  // deterministic way.
  using UnsignedPair = std::pair<unsigned, unsigned>;
  using DomTreeNode = llvm::DomTreeNodeBase<NodeTy>;
  using DomTreeNodePair = std::pair<DomTreeNode *, UnsignedPair>;
  using IDFPriorityQueue =
      std::priority_queue<DomTreeNodePair,
                          llvm::SmallVector<DomTreeNodePair, 32>,
                          llvm::less_second>;
  IDFPriorityQueue pq;

  if (defBlocks->empty())
    return;

  dt.updateDFSNumbers();

  for (NodeTy *bb : *defBlocks) {
    if (DomTreeNode *node = dt.getNode(bb))
      pq.push({node, std::make_pair(node->getLevel(), node->getDFSNumIn())});
  }

  llvm::SmallVector<DomTreeNode *, 32> worklist;
  llvm::SmallPtrSet<DomTreeNode *, 32> visitedpq;
  llvm::SmallPtrSet<DomTreeNode *, 32> visitedWorklist;

  while (!pq.empty()) {
    DomTreeNodePair rootPair = pq.top();
    pq.pop();
    DomTreeNode *root = rootPair.first;
    unsigned rootLevel = rootPair.second.first;

    // Walk all dominator tree children of Root, inspecting their CFG edges with
    // targets elsewhere on the dominator tree. Only targets whose level is at
    // most Root's level are added to the iterated dominance frontier of the
    // definition set.

    worklist.clear();
    worklist.push_back(root);
    visitedWorklist.insert(root);

    while (!worklist.empty()) {
      DomTreeNode *node = worklist.pop_back_val();
      NodeTy *bb = node->getBlock();
      // Succ is the successor in the direction we are calculating IDF, so it is
      // successor for IDF, and predecessor for Reverse IDF.
      auto doWork = [&](NodeTy *succ) {
        DomTreeNode *succNode = dt.getNode(succ);

        const unsigned succLevel = succNode->getLevel();
        if (succLevel > rootLevel)
          return;

        if (!visitedpq.insert(succNode).second)
          return;

        NodeTy *succBB = succNode->getBlock();
        if (useLiveIn && !liveInBlocks->count(succBB))
          return;

        phiBlocks.emplace_back(succBB);
        if (!defBlocks->count(succBB))
          pq.push(std::make_pair(
              succNode, std::make_pair(succLevel, succNode->getDFSNumIn())));
      };

      for (auto *succ : bb->getSuccessors())
        doWork(succ);

      for (auto domChild : *node) {
        if (visitedWorklist.insert(domChild).second)
          worklist.push_back(domChild);
      }
    }
  }
}

template class IDFCalculator<mlir::Block, false>;

} // namespace fir
