// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _C_MITSUBA_MATERIAL_COMPILER_FRONTEND_H_INCLUDED_
#define _C_MITSUBA_MATERIAL_COMPILER_FRONTEND_H_INCLUDED_


//#include "nbl/asset/material_compiler/IR.h"

#include "nbl/ext/MitsubaLoader/CElementBSDF.h"
#include "nbl/ext/MitsubaLoader/CElementEmitter.h"

#include "nbl/asset/interchange/CIESProfileLoader.h"


namespace nbl::ext::MitsubaLoader
{

class CMitsubaMaterialCompilerFrontend
{
    public:
#ifdef 0
        enum E_IMAGE_VIEW_SEMANTIC : uint8_t
        {
            EIVS_IDENTITIY,
            EIVS_BLEND_WEIGHT,
            EIVS_NORMAL_MAP,
            EIVS_BUMP_MAP,
            EIVS_COUNT
        };

        front_and_back_t compileToIRTree(asset::material_compiler::IR* ir, const CElementBSDF* _bsdf);

    private:
        using tex_ass_type = std::tuple<core::smart_refctd_ptr<asset::ICPUImageView>,core::smart_refctd_ptr<asset::ICPUSampler>,float>;

        const SContext* m_loaderContext;

        std::pair<const CElementTexture*,float> unwindTextureScale(const CElementTexture* _element) const;

        tex_ass_type getTexture(const CElementTexture* _element, const E_IMAGE_VIEW_SEMANTIC semantic=EIVS_IDENTITIY) const;
#endif
};

}
#endif