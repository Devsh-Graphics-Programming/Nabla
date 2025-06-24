#include <nbl/asset/material_compiler/v2/IR.h>
#include <nbl/asset/material_compiler/v2/IRNode.h>

using namespace nbl::asset::material_compiler::v2::ir;
using namespace nbl;

ValueType Value::get_type() const {
  if (is_node()) {
    return node->get_return_type();
  }
  return type;
}

INodeHash Value::get_hash() const {
  switch (type) {
  case ValueType::NODE: {
    return node->get_hash();
  }
  case ValueType::TEXTURE: {
    return texture->get_hash();
  }
  case ValueType::FLOAT: {
    return core::fnv_hash(imm_float);
  }
  case ValueType::VEC3: {
    return core::fnv_hash(imm_vec3);
  }
  case ValueType::NONE: {
    assert(false && "Hashing of none value");
  }
  }
}

bool Value::operator==(const Value &other) const {
  if (type != other.type)
    return false;
  switch (type) {
  case ValueType::NODE: {
    return *node == *other.node;
  }
  case ValueType::TEXTURE: {
    return *texture == *other.texture;
  }
  case ValueType::FLOAT: {
    return imm_float == other.imm_float;
  }
  case ValueType::VEC3: {
    return (imm_vec3 == other.imm_vec3).all();
  }
  case ValueType::NONE: {
    assert(false && "Comparison of none value");
  }
  }
}

template <ValueType type_, typename T>
TypedValue<type_, T>::TypedValue(const NodeHandle &node) : Value(node) {
  assert(node->get_return_type() == type_);
}

NODE_FIELDS_DEF(CTexturePrefetchNode, texcel_type, texture)
NODE_VALUES_DEF(CTexturePrefetchNode, texture)

NODE_FIELDS_DEF(CNomralModifierNode, type, value)
NODE_VALUES_DEF(CNomralModifierNode, value)

NODE_FIELDS_DEF(CEmissionNode, intensity)
NODE_NO_VALUES_DEF(CEmissionNode)

NODE_FIELDS_DEF(COpacityNode, opacity, bxdf)
NODE_VALUES_DEF(COpacityNode, bxdf)

COMBINE_NODE_FIELDS_DEF(CBSDFBlendNode, weight, bxdf)
COMBINE_NODE_VALUES_DEF(CBSDFBlendNode, weight, bxdf)

COMBINE_NODE_FIELDS_DEF(CBSDFMixNode, bxdf1, bxdf2)
COMBINE_NODE_VALUES_DEF(CBSDFMixNode, bxdf1, bxdf2)

MS_BSDF_FIELDS_DEF(CMicrofacetSpecularBSDFNode)
MS_BSDF_VALUES_DEF(CMicrofacetSpecularBSDFNode)

MS_BSDF_FIELDS_DEF(CMicrofacetCoatingBSDFNode, thicknessSigmaA)
MS_BSDF_VALUES_DEF(CMicrofacetCoatingBSDFNode, thicknessSigmaA)

MS_BSDF_FIELDS_DEF(CMicrofacetDielectricBSDFNode, thin)
MS_BSDF_VALUES_DEF(CMicrofacetDielectricBSDFNode)

MD_BSDF_FIELDS_DEF(CMicrofacetDiffuseBSDFNode)
MD_BSDF_VALUES_DEF(CMicrofacetDiffuseBSDFNode)

MD_BSDF_FIELDS_DEF(CMicrofacetDifftransBSDFNode, transmittance)
MD_BSDF_VALUES_DEF(CMicrofacetDifftransBSDFNode, transmittance)
