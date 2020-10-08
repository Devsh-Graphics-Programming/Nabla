#ifndef __C_MITSUBA_MATERIAL_COMPILER_FRONTEND_H_INCLUDED__
#define __C_MITSUBA_MATERIAL_COMPILER_FRONTEND_H_INCLUDED__

#include "irr/ext/MitsubaLoader/CElementBSDF.h"
#include <irr/asset/material_compiler/IR.h>

#define DERIV_MAP_FLOAT32

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{
    struct SContext;

class CMitsubaMaterialCompilerFrontend
{
    const SContext* m_loaderContext;

    using tex_ass_type = std::tuple<core::smart_refctd_ptr<asset::ICPUImageView>, core::smart_refctd_ptr<asset::ICPUSampler>, float>;
    core::unordered_map<core::smart_refctd_ptr<asset::ICPUImage>, core::smart_refctd_ptr<asset::ICPUImageView>> m_treeCache;

    tex_ass_type getTexture(const CElementTexture* _element) const;

public:
    CMitsubaMaterialCompilerFrontend(const SContext* _ctx) : m_loaderContext(_ctx) {}

    asset::material_compiler::IR::INode* compileToIRTree(asset::material_compiler::IR* ir, const CElementBSDF* _bsdf);
};

}}}

#endif