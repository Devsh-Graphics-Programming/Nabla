// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_MATERIAL_COMPILER_V2_IR_NODE_H_INCLUDED__
#define __NBL_ASSET_MATERIAL_COMPILER_V2_IR_NODE_H_INCLUDED__

#include <nbl/asset/ICPUImageView.h>
#include <nbl/asset/ICPUSampler.h>
#include <nbl/core/ApplyMacro.h>
#include <nbl/core/IReferenceCounted.h>
#include <nbl/core/containers/refctd_dynamic_array.h>
#include <nbl/core/math/fnvhash.h>

namespace nbl::asset::material_compiler::v2::ir {

using vec3_t = core::vector3df_SIMD;

class IRDAG;

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

struct TextureSource {
  TextureSource() = default;
  ~TextureSource() = default;
  TextureSource(const core::smart_refctd_ptr<ICPUImageView> &image,
                const core::smart_refctd_ptr<ICPUSampler> &sampler, float scale)
      : image(image), sampler(sampler), scale(scale) {}
  core::smart_refctd_ptr<ICPUImageView> image;
  core::smart_refctd_ptr<ICPUSampler> sampler;
  float scale;

  uint64_t get_hash() const {
    return core::fnv_hash(image.get()) ^ core::fnv_hash(sampler.get()) ^
           core::fnv_hash(scale);
  }
  bool operator==(const TextureSource &rhs) const {
    return image == rhs.image && sampler == rhs.sampler && scale == rhs.scale;
  }
};

class TextureSourceHasher {
public:
  size_t operator()(const TextureSource &key) const { return key.get_hash(); }
};

class TextureSourceComparor {
public:
  bool operator()(const TextureSource &lhs, const TextureSource &rhs) const {
    return lhs == rhs;
  }
};

//! Type of a value in the IR
/*!
 * NONE: null type
 * NODE: the result of the node
 * OUTPUT: the final output of BSDF graph -- could be BSDF value, PDF, quotient
 * remaindar, or AOV depending on instruction stream type VEC3: vector with 3
 * float elements TEXTURE: handle pointing to texture
 */
enum class ValueType : uint8_t {
  NONE = 0,
  NODE,
  OUTPUT,
  VEC3,
  FLOAT,
  TEXTURE,
};

using INodeHash = uint64_t;

struct INode;
using TextureHandle = Handle<TextureSource>;
using NodeHandle = Handle<INode>;

//! Value in the IR
/** A value is either constant or result of node.
 */
class Value {
public:
  Value() : type(ValueType::NONE) {}
  ~Value() {}

  explicit Value(NodeHandle node) : type(ValueType::NODE), node(node) {}
  explicit Value(vec3_t value) : type(ValueType::VEC3), imm_vec3(value) {}
  explicit Value(float value) : type(ValueType::FLOAT), imm_float(value) {}
  explicit Value(TextureHandle value)
      : type(ValueType::TEXTURE), texture(texture) {}

  bool is_node() const { return type == ValueType::NODE; }
  bool is_const() const { return !is_none() && !is_node(); }
  bool is_none() const { return type == ValueType::NONE; }
  bool is_texture() const { return type == ValueType::TEXTURE; }

  vec3_t get_vec3() const {
    assert(type == ValueType::VEC3);
    return imm_vec3;
  }
  float get_float() const {
    assert(type == ValueType::FLOAT);
    return imm_float;
  }
  NodeHandle get_node() const {
    assert(type == ValueType::NODE);
    return node;
  }
  TextureHandle get_texture() const {
    assert(type == ValueType::TEXTURE);
    return texture;
  }
  // Return type of immediate or return type of node
  ValueType get_type() const;
  // Return the hash of the value
  INodeHash get_hash() const;

  Value(const Value &other) { std::memcpy(this, &other, sizeof(Value)); }
  Value &operator=(const Value &other) {
    std::memcpy(this, &other, sizeof(Value));
    return *this;
  }

  bool operator==(const Value &other) const;

private:
  ValueType type;
  union {
    NodeHandle node;
    TextureHandle texture;
    vec3_t imm_vec3;
    float imm_float;
  };
};

//! Value with type checking
/** Type checking will disabled when assertions are disabled.
 */
template <ValueType type_, typename T> class TypedValue : public Value {
public:
  TypedValue() : Value() {}
  ~TypedValue() {}

  template <typename F = T,
            typename std::enable_if<!std::is_same<F, void>::value, int>::type
                * = nullptr>
  TypedValue(F value) : Value(value) {}

  explicit TypedValue(const Value &value) : Value(value) {
    assert(value.get_type() == type_);
  }

  explicit TypedValue(const NodeHandle &node);
};

using OutputValue = TypedValue<ValueType::OUTPUT, void>;
using Vec3Value = TypedValue<ValueType::VEC3, vec3_t>;
using FloatValue = TypedValue<ValueType::FLOAT, float>;
using TextureValue = TypedValue<ValueType::TEXTURE, TextureHandle>;

using OffsetToValue = std::ptrdiff_t;

#define RETURN_TYPE(RT)                                                        \
  ValueType get_return_type() const override { return ValueType::RT; }

#define FIELD_TO_PTR(CLS_NAME, X) offsetof(CLS_NAME, X)

#define NODE_VALUES_DEF(CLS_NAME, ...)                                         \
  const core::vector<OffsetToValue> CLS_NAME::value_offsets = {                \
      APPLY_2(FIELD_TO_PTR, CLS_NAME, __VA_ARGS__)};

// FIXME: remove this
// we get error C2760: syntax error: unexpected token 'End of Token Stream',
// expected 'id-expression' without this
#define NODE_NO_VALUES_DEF(CLS_NAME)                                           \
  const core::vector<OffsetToValue> CLS_NAME::value_offsets = {};

template <typename T>
NBL_FORCE_INLINE bool field_compare(const T &lhs, const T &rhs) {
  return lhs == rhs;
}

template <>
NBL_FORCE_INLINE bool field_compare(const vec3_t &lhs, const vec3_t &rhs) {
  return (lhs == rhs).all();
}

#define FIELD_TO_COMPARE(OTHER, X) field_compare(X, OTHER.X)

#define NODE_FIELDS_DEF(CLS_NAME, ...)                                         \
  NodeHandle CLS_NAME::clone(IRDAG &dag) const {                               \
    NodeHandle node = dag.create_node<CLS_NAME>();                             \
    reinterpret_cast<CLS_NAME &>(*node) = *this;                               \
    return node;                                                               \
  }                                                                            \
  INodeHash CLS_NAME::get_hash() const {                                       \
    return hash_fields(symbol, __VA_ARGS__);                                   \
  }                                                                            \
  bool CLS_NAME::equals(const INode &other_) const {                           \
    const CLS_NAME &other = static_cast<const CLS_NAME &>(other_);             \
    bool comparison[] = {APPLY_2(FIELD_TO_COMPARE, other, __VA_ARGS__)};       \
    bool res = true;                                                           \
    for (bool c : comparison)                                                  \
      res &= c;                                                                \
    return res;                                                                \
  }

#define NODE_DECL(NT)                                                          \
  static const core::vector<OffsetToValue> value_offsets;                      \
  const core::vector<OffsetToValue> &get_value_offsets() override {            \
    return value_offsets;                                                      \
  }                                                                            \
  INodeHash get_hash() const override;                                         \
  NodeHandle clone(IRDAG &dag) const override;                                 \
                                                                               \
protected:                                                                     \
  bool equals(const INode &other) const override;                              \
                                                                               \
public:

template <typename T> INodeHash node_hash(const T &value) {
  return core::fnv_hash(value);
}

template <> INodeHash node_hash(const Value &value) { return value.get_hash(); }

struct INode {
  enum E_SYMBOL : uint8_t {
    ES_NORMAL_MODIFIER,
    ES_TEXTURE_PREFETCH,
    ES_EMISSION,
    ES_OPACITY,
    ES_BSDF,
    ES_BSDF_COMBINER
  };

  virtual ~INode() = default;

  // Get the hash of this subtree
  virtual INodeHash get_hash() const = 0;
  // Get the return type of this node; either OUTPUT or VEC3
  virtual ValueType get_return_type() const = 0;
  // Get the values of this node
  virtual const core::vector<OffsetToValue> &get_value_offsets() = 0;
  // Clone this node
  virtual NodeHandle clone(IRDAG &dag) const = 0;

  bool operator==(const INode &other) const {
    if (!compare_node_type(other))
      return false;
    return equals(other);
  }

  virtual bool compare_node_type(const INode &other) const {
    return symbol == other.symbol;
  }

  Value &value(OffsetToValue offset) {
    return *reinterpret_cast<Value *>(reinterpret_cast<uint8_t *>(this) +
                                      offset);
  }

  E_SYMBOL symbol;

protected:
  INode(E_SYMBOL s) : symbol(s) {}

  // Deep compare node of same type
  virtual bool equals(const INode &other) const = 0;

  template <typename... Args>
  NBL_FORCE_INLINE INodeHash hash_fields(Args &&...args) const {
    return 0 ^ (node_hash(std::forward<Args>(args)) ^ ...);
  }
};

#define COMBINE_NODE_FIELDS_DEF(CLS_NAME, ...)                                 \
  NODE_FIELDS_DEF(CLS_NAME, type, ##__VA_ARGS__)
#define COMBINE_NODE_VALUES_DEF(CLS_NAME, ...)                                 \
  NODE_VALUES_DEF(CLS_NAME, __VA_ARGS__)

struct IBSDFCombinerNode : INode {
  enum E_TYPE {
    // mix of N BSDFs
    ET_MIX,
    // blend of 2 BSDFs weighted by constant or texture
    ET_WEIGHT_BLEND,
    // for support of nvidia MDL's df::fresnel_layer
    ET_LOL_MDL_SUX_BROKEN_FRESNEL_BLEND,
    // blend of 2 BSDFs weighted by custom direction-based curve
    ET_CUSTOM_CURVE_BLEND
  };

  bool compare_node_type(const INode &other) const override {
    if (!INode::compare_node_type(other))
      return false;
    return type == static_cast<const IBSDFCombinerNode &>(other).type;
  }

  E_TYPE type;

protected:
  IBSDFCombinerNode(E_TYPE t) : INode(ES_BSDF_COMBINER), type(t) {}
};

#define BSDF_FIELDS_DEF(CLS_NAME, ...)                                         \
  NODE_FIELDS_DEF(CLS_NAME, type, normal, ##__VA_ARGS__)
#define BSDF_VALUES_DEF(CLS_NAME, ...)                                         \
  NODE_VALUES_DEF(CLS_NAME, normal, ##__VA_ARGS__)

struct IBSDFNode : INode {
  ~IBSDFNode() = default;
  enum E_TYPE {
    ET_MICROFACET_DIFFTRANS,
    ET_MICROFACET_DIFFUSE,
    ET_MICROFACET_SPECULAR,
    ET_MICROFACET_COATING,
    ET_MICROFACET_DIELECTRIC,
    ET_DELTA_TRANSMISSION
    // ET_SHEEN,
  };

  bool compare_node_type(const INode &other) const override {
    if (!INode::compare_node_type(other))
      return false;
    return type == static_cast<const IBSDFNode &>(other).type;
  }

  E_TYPE type;
  // normal will be lowered to normal cache update operations in backend
  Vec3Value normal = vec3_t(.0f);

protected:
  IBSDFNode(E_TYPE t) : INode(ES_BSDF), type(t) {}
};

#define MS_BSDF_FIELDS_DEF(CLS_NAME, ...)                                      \
  BSDF_FIELDS_DEF(CLS_NAME, ndf, shadowing, alpha_u, alpha_v, ##__VA_ARGS__)
#define MS_BSDF_VALUES_DEF(CLS_NAME, ...)                                      \
  BSDF_VALUES_DEF(CLS_NAME, alpha_u, alpha_v, ##__VA_ARGS__)

struct IMicrofacetSpecularBSDFBase : IBSDFNode {
  enum E_NDF { ENDF_BECKMANN, ENDF_GGX, ENDF_ASHIKHMIN_SHIRLEY, ENDF_PHONG };
  // TODO: Remove, the NDF fixes the geometrical shadowing and masking function.
  enum E_SHADOWING_TERM { EST_SMITH, EST_VCAVITIES };

  ~IMicrofacetSpecularBSDFBase() = default;

  void setSmooth(E_NDF _ndf = ENDF_GGX) {
    ndf = _ndf;
    alpha_u = 0.f;
    alpha_v = 0.f;
  }

  E_NDF ndf = ENDF_GGX;
  E_SHADOWING_TERM shadowing = EST_SMITH;
  FloatValue alpha_u = 0.f;
  FloatValue alpha_v = 0.f;

protected:
  IMicrofacetSpecularBSDFBase(E_TYPE t) : IBSDFNode(t) {}
};

#define MD_BSDF_FIELDS_DEF(CLS_NAME, ...)                                      \
  BSDF_FIELDS_DEF(CLS_NAME, alpha_u, alpha_v, ##__VA_ARGS__)
#define MD_BSDF_VALUES_DEF(CLS_NAME, ...)                                      \
  BSDF_VALUES_DEF(CLS_NAME, alpha_u, alpha_v, ##__VA_ARGS__)

struct IMicrofacetDiffuseBxDFBase : IBSDFNode {
  ~IMicrofacetDiffuseBxDFBase() = default;

  void setSmooth() {
    alpha_u = 0.f;
    alpha_v = 0.f;
  }

  FloatValue alpha_u = 0.f;
  FloatValue alpha_v = 0.f;

protected:
  IMicrofacetDiffuseBxDFBase(E_TYPE t) : IBSDFNode(t) {}
};

struct CTexturePrefetchNode final : INode {
  enum E_TEXCEL_TYPE : uint8_t { ETT_VEC3, ETT_FLOAT };

  CTexturePrefetchNode() : INode(ES_TEXTURE_PREFETCH) {}
  ~CTexturePrefetchNode() = default;

  ValueType get_return_type() const override {
    return texcel_type == ETT_VEC3 ? ValueType::VEC3 : ValueType::FLOAT;
  }
  NODE_DECL(CTexturePrefetchNode)

  E_TEXCEL_TYPE texcel_type = ETT_VEC3;
  TextureValue texture;
};

struct CNomralModifierNode final : INode {
  enum E_TYPE : uint8_t {
    ET_DISPLACEMENT,
    ET_HEIGHT,
    ET_NORMAL,
    ET_DERIVATIVE
  };

  enum E_SOURCE : uint8_t { ESRC_UV_FUNCTION, ESRC_TEXTURE };

  CNomralModifierNode() : INode(ES_NORMAL_MODIFIER) {}
  CNomralModifierNode(E_TYPE t) : INode(ES_NORMAL_MODIFIER), type(t) {}
  ~CNomralModifierNode() = default;

  NODE_DECL(CNomralModifierNode)
  RETURN_TYPE(OUTPUT)

  E_TYPE type = ET_DISPLACEMENT;
  // no other (than texture) source supported for now (uncomment in the future)
  // [far future TODO] E_SOURCE source;
  // TODO when source==ESRC_UV_FUNCTION, use the proper Value representation
  // from some node in fact when we want to translate MDL function of (u,v) into
  // this IR we could just create an image being a 2D plot of this function with
  // some reasonable quantization (pixel dimensions)
  Vec3Value value = vec3_t(0.f);
};

struct CEmissionNode final : INode {
  CEmissionNode() : INode(ES_EMISSION) {}
  ~CEmissionNode() = default;

  NODE_DECL(CEmissionNode)
  RETURN_TYPE(OUTPUT)

  vec3_t intensity = vec3_t(1.f);
};

struct COpacityNode final : INode {
  COpacityNode() : INode(ES_OPACITY) {}
  ~COpacityNode() = default;

  NODE_DECL(COpacityNode)
  RETURN_TYPE(OUTPUT)

  Vec3Value opacity = vec3_t(0.f);
  OutputValue bxdf;
};

struct CBSDFBlendNode final : IBSDFCombinerNode {
  CBSDFBlendNode() : IBSDFCombinerNode(ET_WEIGHT_BLEND) {}
  ~CBSDFBlendNode() = default;

  RETURN_TYPE(OUTPUT)
  NODE_DECL(CBSDFBlendNode)

  Vec3Value weight = vec3_t(0.f);
  OutputValue bxdf;
};

struct CBSDFMixNode final : IBSDFCombinerNode {
  CBSDFMixNode() : IBSDFCombinerNode(ET_MIX) {}
  ~CBSDFMixNode() = default;

  RETURN_TYPE(OUTPUT)
  NODE_DECL(CBSDFMixNode)

  OutputValue bxdf1;
  OutputValue bxdf2;
};

struct CMicrofacetSpecularBSDFNode final : IMicrofacetSpecularBSDFBase {
  CMicrofacetSpecularBSDFNode()
      : IMicrofacetSpecularBSDFBase(ET_MICROFACET_SPECULAR) {}
  ~CMicrofacetSpecularBSDFNode() = default;

  RETURN_TYPE(OUTPUT)
  NODE_DECL(CMicrofacetSpecularBSDFNode)
};

struct CMicrofacetCoatingBSDFNode final : IMicrofacetSpecularBSDFBase {
  CMicrofacetCoatingBSDFNode()
      : IMicrofacetSpecularBSDFBase(ET_MICROFACET_COATING) {}
  ~CMicrofacetCoatingBSDFNode() = default;

  RETURN_TYPE(OUTPUT)
  NODE_DECL(CMicrofacetCoatingBSDFNode)

  Vec3Value thicknessSigmaA = vec3_t(0.0f);
};

struct CMicrofacetDielectricBSDFNode final : IMicrofacetSpecularBSDFBase {
  CMicrofacetDielectricBSDFNode()
      : IMicrofacetSpecularBSDFBase(ET_MICROFACET_DIELECTRIC) {}
  ~CMicrofacetDielectricBSDFNode() = default;

  RETURN_TYPE(OUTPUT)
  NODE_DECL(CMicrofacetDielectricBSDFNode)

  bool thin = false;
};

struct CMicrofacetDiffuseBSDFNode final : IMicrofacetDiffuseBxDFBase {
  CMicrofacetDiffuseBSDFNode()
      : IMicrofacetDiffuseBxDFBase(ET_MICROFACET_DIFFUSE) {}
  ~CMicrofacetDiffuseBSDFNode() = default;

  RETURN_TYPE(OUTPUT)
  NODE_DECL(CMicrofacetDiffuseBSDFNode)

  Vec3Value reflectance = vec3_t(1.f);
};

struct CMicrofacetDifftransBSDFNode final : IMicrofacetDiffuseBxDFBase {
  CMicrofacetDifftransBSDFNode()
      : IMicrofacetDiffuseBxDFBase(ET_MICROFACET_DIFFTRANS) {}
  ~CMicrofacetDifftransBSDFNode() = default;

  RETURN_TYPE(OUTPUT)
  NODE_DECL(CMicrofacetDifftransBSDFNode)

  Vec3Value transmittance = vec3_t(0.5f);
};

} // namespace nbl::asset::material_compiler::v2::ir

#endif