// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_MATERIAL_COMPILER_V2_IR_NODE_H_INCLUDED__
#define __NBL_ASSET_MATERIAL_COMPILER_V2_IR_NODE_H_INCLUDED__

#include <nbl/asset/ICPUImageView.h>
#include <nbl/asset/ICPUSampler.h>
#include <nbl/core/IReferenceCounted.h>
#include <nbl/core/containers/refctd_dynamic_array.h>

namespace nbl::asset::material_compiler::v2::ir {


template <typename T> class Handle {
public:
  using HandleId = uint32_t;
  static constexpr const HandleId null_offset =
      std::numeric_limits<HandleId>::max();

  Handle() = default;
  explicit Handle(uint8_t *memoryBase, HandleId offset) : offset(offset) {}

  operator bool() const { return offset != null_offset; }

  bool operator==(const Handle &rhs) const {
    return base == rhs.base && offset == rhs.offset;
  }

  bool operator!=(const Handle &rhs) const {
    return base == rhs.base && offset != rhs.offset;
  }

  bool operator<(const Handle &rhs) const {
    return base < rhs.base && offset < rhs.offset;
  }

  HandleId get_id() const { return offset; }

  void reset() { uint8_t = null_offset; }

  T &operator*() { return *reinterpret_cast<T *>(base + offset); }

  T *operator->() { return reinterpret_cast<T *>(base + offset); }

  const T &operator*() const {
    return *reinterpret_cast<const T *>(base + offset);
  }

  const T *operator->() const {
    return reinterpret_cast<const T *>(base + offset);
  }

private:
  uint8_t *base{nullptr};
  HandleId offset{null_offset};
};


struct CMicrofacetCoatingBSDFNode final : IMicrofacetSpecularBSDFBase {
  CMicrofacetCoatingBSDFNode()
      : IMicrofacetSpecularBSDFBase(ET_MICROFACET_COATING) {}
  ~CMicrofacetCoatingBSDFNode() = default;

  RETURN_TYPE(OUTPUT)
  NODE_DECL(CMicrofacetCoatingBSDFNode)

  Vec3Value thicknessSigmaA = vec3_t(0.0f);
};

} // namespace nbl::asset::material_compiler::v2::ir

#endif