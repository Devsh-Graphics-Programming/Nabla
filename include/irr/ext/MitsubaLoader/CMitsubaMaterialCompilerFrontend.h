#ifndef __C_MITSUBA_MATERIAL_COMPILER_FRONTEND_H_INCLUDED__
#define __C_MITSUBA_MATERIAL_COMPILER_FRONTEND_H_INCLUDED__

#include "irr/ext/MitsubaLoader/CElementBSDF.h"
#include <irr/asset/material_compiler/IR.h>

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{
    struct SContext;

class CMitsubaMaterialCompilerFrontend
{
    using tex_ass_type = std::tuple<core::smart_refctd_ptr<asset::ICPUImageView>, core::smart_refctd_ptr<asset::ICPUSampler>, float>;

    const SContext* m_loaderContext;

    tex_ass_type getDerivMap(const CElementTexture* _element) const;
    tex_ass_type getBlendWeightTex(const CElementTexture* _element) const;

    std::pair<const CElementTexture*, float> getTexture_common(const CElementTexture* _element) const;

    tex_ass_type getTexture(const CElementTexture* _element) const;
    tex_ass_type getTexture(const std::string& _key, const CElementTexture* _element, float _scale) const;

public:
    CMitsubaMaterialCompilerFrontend(const SContext* _ctx) : m_loaderContext(_ctx) {}

    asset::material_compiler::IR::INode* compileToIRTree(asset::material_compiler::IR* ir, const CElementBSDF* _bsdf);
};

}}}

#endif