// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_MATERIAL_COMPILER_V2_IR_H_INCLUDED__
#define __NBL_ASSET_MATERIAL_COMPILER_V2_IR_H_INCLUDED__

#include <nbl/core/IReferenceCounted.h>
#include <nbl/core/containers/refctd_dynamic_array.h>

namespace nbl::asset::material_compiler::v2::ir {

// Traverse the tree of nodes in reverse bfs order which
// gurantess we visit all nodes gets depended by upper level
// nodes are visited beforehand. (kahn's algorithm)
template <typename Visitor>
NBL_FORCE_INLINE void traverse_node_tree_topological(NodeHandle root,
                                                     Visitor visitor) {
  core::queue<NodeHandle> q;
  core::vector<NodeHandle> worklist;
  q.push(root);
  while (!q.empty()) {
    NodeHandle cur = q.front();
    q.pop();
    for (OffsetToValue value_offset : cur->get_value_offsets())
      if (cur->value(value_offset).is_node())
        worklist.push_back(cur->value(value_offset).get_node());
  }
  while (!worklist.empty()) {
    NodeHandle cur = worklist.back();
    worklist.pop_back();
    visitor(cur);
  }
}

using INodeCacheSet =
    core::unordered_set<NodeHandle, NodeHandleHasher, NodeHandleComparor>;

class NBL_API IRDAG : public core::IReferenceCounted
{
public:
  // Insert subdag into this DAG
  NodeHandle insert_subdag(IRDAG &dag) {
    assert(dag.root_nodes.size() == 1); // we must have only one root
    INodeCacheSet cache;
    for (NodeHandle node : nodes.get_handles()) {
      cache.insert(node);
    }
    NodeHandle root = *dag.root_nodes.begin();
    traverse_node_tree_topological(root, [&](NodeHandle cur) {
      if (cache.count(cur))
        return;
      NodeHandle cloned = cur->clone(*this);
      for (OffsetToValue value_offset : cloned->get_value_offsets()) {
        Value &value = cloned->value(value_offset);
        if (value.is_node()) {
          auto it = cache.find(value.get_node());
          assert(it != cache.end()); // node must have been visited before
          value = Value(*it);
        }
        if (value.is_texture())
          value = Value(create_texture(*value.get_texture()));
      }
      cache.insert(cloned);
    });
    return *cache.find(root);
  }

  NodeHandle clone_node(NodeHandle node) { return node->clone(*this); }

  template <typename ST, typename... Args>
  NodeHandle create_node(Args &&...args) {
    return nodes.create<ST>(std::forward<Args>(args)...);
  }

  };


NBL_FORCE_INLINE void subexpression_elimination_pass(IRDAG &dag) {
  INodeCacheSet cache;
  for (NodeHandle root : dag.get_roots()) {
    // traverse tree in topological order to properly add first known node of
    // the same hash into cache and replace children with cached node
    traverse_node_tree_topological(root, [&](NodeHandle cur) {
      if (cache.count(cur))
        return;
      for (OffsetToValue value_offset : cur->get_value_offsets()) {
        Value &value = cur->value(value_offset);
        if (!value.is_node())
          continue;
        auto it = cache.find(value.get_node());
        assert(it != cache.end()); // node must have been visited before
        value = Value(*it);
      }
      cache.insert(cur);
    });
  }
  dead_strip_pass(dag); // we can now remove unused nodes
}

} // namespace nbl::asset::material_compiler::v2::ir

#endif