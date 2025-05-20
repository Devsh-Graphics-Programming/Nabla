// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_MATERIAL_COMPILER_V2_IR_H_INCLUDED__
#define __NBL_ASSET_MATERIAL_COMPILER_V2_IR_H_INCLUDED__

#include "IRNode.h"
#include <nbl/asset/ICPUImageView.h>
#include <nbl/asset/ICPUSampler.h>
#include <nbl/core/IReferenceCounted.h>
#include <nbl/core/containers/refctd_dynamic_array.h>
#include <nbl/core/math/fnvhash.h>

namespace nbl::asset::material_compiler::v2::ir {

// Visit the tree of nodes with specified node as root in DFS order.
template <typename Visitor>
NBL_FORCE_INLINE void traverse_node_tree_dfs(NodeHandle root, Visitor visitor) {
  core::vector<NodeHandle> worklist;
  worklist.push_back(root);
  while (!worklist.empty()) {
    NodeHandle cur = worklist.back();
    worklist.pop_back();
    visitor(cur);
    for (OffsetToValue value_offset : cur->get_value_offsets())
      if (cur->value(value_offset).is_node())
        worklist.push_back(cur->value(value_offset).get_node());
  }
}

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

class NodeHandleHasher {
public:
  size_t operator()(const NodeHandle &key) const { return key->get_hash(); }
};

class NodeHandleComparor {
public:
  bool operator()(const NodeHandle &lhs, const NodeHandle &rhs) const {
    return *lhs == *rhs;
  }
};

using INodeCacheSet =
    core::unordered_set<NodeHandle, NodeHandleHasher, NodeHandleComparor>;

class NBL_API IRDAG : public core::IReferenceCounted {
public:
  IRDAG() {}

  ~IRDAG() {}

  INodeHash get_hash() const {
    INodeHash hash_value = 0;
    for (NodeHandle root : root_nodes) {
      hash_value ^= root->get_hash();
    }
    return hash_value;
  }

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

  TextureHandle create_texture(const TextureSource &texture) {
    auto it = texture_cache.find(texture);
    if (it != texture_cache.end())
      return it->second;
    return texture_cache[texture] = textures.create<TextureSource>(
               texture.image, texture.sampler, texture.scale);
  }

  void add_root(NodeHandle node) { root_nodes.push_back(node); }

  void remove_node(NodeHandle node) { nodes.remove(node); }

  const core::set<NodeHandle> &get_nodes() const { return nodes.get_handles(); }

  const core::vector<NodeHandle> &get_roots() const { return root_nodes; }

private:
  template <typename T, typename MemMgr> class CHandleManager {
  public:
    using HandleId = Handle<T>::HandleId;

    ~CHandleManager() {
      while (!handles.empty())
        remove(*handles.rbegin());
    }

    template <typename ST, typename... Args> Handle<T> create(Args &&...args) {
      ST *ptr = alloc<ST>(std::forward<Args>(args)...);
      ptrdiff_t offset = reinterpret_cast<uint8_t *>(ptr) - mem_mgr.get_base();
      HandleId new_id = offset;
      return Handle<T>(reinterpret_cast<uint8_t *>(mem_mgr.get_base()), new_id);
    }

    void remove(Handle<T> handle) {
      T &obj = *handle;
      obj.~T();
      handles.erase(handle);
    }

    const core::set<Handle<T>> &get_handles() const { return handles; }

  private:
    template <typename ST, typename... Args> ST *alloc(Args &&...args) {
      uint8_t *ptr = mem_mgr.alloc(sizeof(ST));
      return new (ptr) ST(std::forward<Args>(args)...);
    }

    core::set<Handle<T>> handles;
    MemMgr mem_mgr;
  };

  class SBackingMemManager {
    _NBL_STATIC_INLINE_CONSTEXPR size_t INITIAL_MEM_SIZE = 1ull << 20;
    _NBL_STATIC_INLINE_CONSTEXPR size_t MAX_MEM_SIZE = 1ull << 20;
    _NBL_STATIC_INLINE_CONSTEXPR size_t ALIGNMENT = _NBL_SIMD_ALIGNMENT;

    uint8_t *mem;
    size_t currSz;
    using addr_alctr_t = core::LinearAddressAllocator<uint32_t>;
    addr_alctr_t addrAlctr;

  public:
    SBackingMemManager()
        : mem(nullptr), currSz(INITIAL_MEM_SIZE),
          addrAlctr(nullptr, 0u, 0u, ALIGNMENT, MAX_MEM_SIZE) {
      mem = reinterpret_cast<uint8_t *>(_NBL_ALIGNED_MALLOC(currSz, ALIGNMENT));
    }
    ~SBackingMemManager() { _NBL_ALIGNED_FREE(mem); }

    uint8_t *get_base() const { return mem; }

    uint8_t *alloc(size_t bytes) {
      auto addr = addrAlctr.alloc_addr(bytes, ALIGNMENT);
      assert(addr != addr_alctr_t::invalid_address);
      // TODO reallocation will invalidate all pointers to nodes, so...
      // 1) never reallocate (just have reasonably big buffer for nodes)
      // 2) make some node_handle class that will work as pointer but is based
      // on offset instead of actual address
      if (addr + bytes > currSz) {
        size_t newSz = currSz << 1;
        if (newSz > MAX_MEM_SIZE) {
          addrAlctr.free_addr(addr, bytes);
          return nullptr;
        }

        void *newMem = _NBL_ALIGNED_MALLOC(newSz, ALIGNMENT);
        memcpy(newMem, mem, currSz);
        _NBL_ALIGNED_FREE(mem);
        mem = reinterpret_cast<uint8_t *>(newMem);
        currSz = newSz;
      }

      return mem + addr;
    }

    uint32_t getAllocatedSize() const { return addrAlctr.get_allocated_size(); }

    void freeLastAllocatedBytes(uint32_t _bytes) {
      assert(addrAlctr.get_allocated_size() >= _bytes);
      const uint32_t newCursor = addrAlctr.get_allocated_size() - _bytes;
      addrAlctr.reset(newCursor);
    }
  };

  core::vector<NodeHandle> root_nodes;
  CHandleManager<INode, SBackingMemManager> nodes;
  CHandleManager<TextureSource, SBackingMemManager> textures;
  core::unordered_map<TextureSource, TextureHandle, TextureSourceHasher,
                      TextureSourceComparor>
      texture_cache;
};

NBL_FORCE_INLINE void dead_strip_pass(IRDAG &dag) {
  core::set<NodeHandle> dead = dag.get_nodes();
  for (NodeHandle root : dag.get_roots())
    traverse_node_tree_dfs(root, [&](NodeHandle node) { dead.erase(node); });
  for (NodeHandle node : dead)
    dag.remove_node(node);
}

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