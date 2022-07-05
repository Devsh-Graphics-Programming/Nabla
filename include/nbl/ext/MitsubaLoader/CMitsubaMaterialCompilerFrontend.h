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

class CMitsubaMaterialCompilerFrontend final
{
    public:
        using node_handle_t = asset::material_compiler::IR::node_handle_t;
        enum E_IMAGE_VIEW_SEMANTIC : uint8_t
        {
            EIVS_IDENTITIY,
            EIVS_BLEND_WEIGHT,
            EIVS_NORMAL_MAP,
            EIVS_BUMP_MAP,
            EIVS_COUNT
        };

        // TODO: embed hash val in the element for speed
        struct MerkleTree
        {
            struct hash
            {
                std::size_t operator()(const CElementBSDF* node) const;
            };
            struct equal_to
            {
                bool operator()(const CElementBSDF* node) const;
            };
        };

        struct SContext
        {
            const ext::MitsubaLoader::SContext* m_loaderContext;
            asset::material_compiler::IR* m_ir;
            core::unordered_map<const CElementBSDF*,node_handle_t,MerkleTree::hash,MerkleTree::equal_to> m_hashCons;
        };
        struct front_and_back_t
        {
            node_handle_t front;
            node_handle_t back;
        };
        static front_and_back_t compileToIRTree(SContext& ctx, const CElementBSDF* _root);

    private:
        static node_handle_t createIRNode(SContext& ctx, const CElementBSDF* _bsdf);

        using tex_ass_type = std::tuple<core::smart_refctd_ptr<asset::ICPUImageView>, core::smart_refctd_ptr<asset::ICPUSampler>, float>;
        static tex_ass_type getTexture(const ext::MitsubaLoader::SContext* _loaderContext, const CElementTexture* _element, const E_IMAGE_VIEW_SEMANTIC semantic=EIVS_IDENTITIY);
        static tex_ass_type getErrorTexture(const ext::MitsubaLoader::SContext* _loaderContext);
};

}

#endif