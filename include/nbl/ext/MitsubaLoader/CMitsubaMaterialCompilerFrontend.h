// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __C_MITSUBA_MATERIAL_COMPILER_FRONTEND_H_INCLUDED__
#define __C_MITSUBA_MATERIAL_COMPILER_FRONTEND_H_INCLUDED__

#include "nbl/core/Types.h"

#include "nbl/asset/material_compiler/IR.h"

#include "nbl/ext/MitsubaLoader/CElementBSDF.h"

namespace nbl::ext::MitsubaLoader
{

struct SContext;

class CMitsubaMaterialCompilerFrontend
{
    public:
        using IRNode = asset::material_compiler::IR::INode;
        enum E_IMAGE_VIEW_SEMANTIC : uint8_t
        {
            EIVS_IDENTITIY,
            EIVS_BLEND_WEIGHT,
            EIVS_NORMAL_MAP,
            EIVS_BUMP_MAP,
            EIVS_COUNT
        };

        struct front_and_back_t
        {
            IRNode* front;
            IRNode* back;
        };

        explicit CMitsubaMaterialCompilerFrontend(const SContext* _ctx) : m_loaderContext(_ctx) {}

        front_and_back_t compileToIRTree(asset::material_compiler::IR* ir, const CElementBSDF* _bsdf);

    private:
        using tex_ass_type = std::tuple<core::smart_refctd_ptr<asset::ICPUImageView>,core::smart_refctd_ptr<asset::ICPUSampler>,float>;

        const SContext* m_loaderContext;

        std::pair<const CElementTexture*,float> unwindTextureScale(const CElementTexture* _element) const;

        tex_ass_type getTexture(const CElementTexture* _element, const E_IMAGE_VIEW_SEMANTIC semantic=EIVS_IDENTITIY) const;

        tex_ass_type getErrorTexture(const E_IMAGE_VIEW_SEMANTIC semantic) const;

        IRNode* createIRNode(asset::material_compiler::IR* ir, const CElementBSDF* _bsdf);
};

}

#endif