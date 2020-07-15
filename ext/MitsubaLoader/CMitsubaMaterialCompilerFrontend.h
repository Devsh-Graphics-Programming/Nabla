#ifndef __C_MITSUBA_MATERIAL_COMPILER_FRONTEND_H_INCLUDED__
#define __C_MITSUBA_MATERIAL_COMPILER_FRONTEND_H_INCLUDED__

#include "../../ext/MitsubaLoader/CElementBSDF.h"
#include "../../ext/MitsubaLoader/CMitsubaLoader.h"
#include <irr/asset/material_compiler/IR.h>

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class CMitsubaMaterialCompilerFrontend
{
    const CMitsubaLoader::SContext* m_loaderContext;
    core::unordered_map<const CElementBSDF*, asset::material_compiler::IR::INode*> m_nodeMap;

    inline CMitsubaLoader::SContext::tex_ass_type getTexture(const CElementTexture* _element) const
    {
        auto found = m_loaderContext->textureCache.find(_element);
        if (found == m_loaderContext->textureCache.end())
            return CMitsubaLoader::SContext::tex_ass_type(nullptr, nullptr, 0.f);

        return found->second;
    }

public:
    asset::material_compiler::IR::INode* compileToIRTree(asset::material_compiler::IR* ir, const CElementBSDF* _bsdf);
};

}}}

#endif