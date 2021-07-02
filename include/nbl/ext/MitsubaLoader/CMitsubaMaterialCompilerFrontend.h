// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __C_MITSUBA_MATERIAL_COMPILER_FRONTEND_H_INCLUDED__
#define __C_MITSUBA_MATERIAL_COMPILER_FRONTEND_H_INCLUDED__

#include "nbl/ext/MitsubaLoader/CElementBSDF.h"
#include <nbl/asset/material_compiler/IR.h>

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{
    struct SContext;

class CMitsubaMaterialCompilerFrontend
{
    using IRNode = asset::material_compiler::IR::INode;
    using tex_ass_type = std::tuple<core::smart_refctd_ptr<asset::ICPUImageView>, core::smart_refctd_ptr<asset::ICPUSampler>, float>;

    const SContext* m_loaderContext;

    tex_ass_type getDerivMap(const CElementBSDF::BumpMap& _bump) const;
    tex_ass_type getBlendWeightTex(const CElementTexture* _element) const;

    std::pair<const CElementTexture*, float> getTexture_common(const CElementTexture* _element) const;

    tex_ass_type getTexture(const CElementTexture* _element) const;
    tex_ass_type getTexture(const std::string& _key, const CElementTexture* _element, float _scale) const;

    IRNode* createIRNode(asset::material_compiler::IR* ir, const CElementBSDF* _bsdf);

public:
    struct front_and_back_t
    {
        IRNode* front;
        IRNode* back;
    };

    CMitsubaMaterialCompilerFrontend(const SContext* _ctx) : m_loaderContext(_ctx) {}

    front_and_back_t compileToIRTree(asset::material_compiler::IR* ir, const CElementBSDF* _bsdf);
};

}}}

#endif