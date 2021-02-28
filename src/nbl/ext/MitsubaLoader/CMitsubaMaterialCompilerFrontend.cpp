// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/MitsubaLoader/CMitsubaMaterialCompilerFrontend.h"
#include "nbl/ext/MitsubaLoader/SContext.h"
#include "nbl/core/Types.h"

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{

    auto CMitsubaMaterialCompilerFrontend::getDerivMap(const CElementTexture* _element) const -> tex_ass_type
    {
        float scale = 1.f;
        std::tie(_element, scale) = getTexture_common(_element);
        std::string key = m_loaderContext->derivMapCacheKey(_element);
        if (_element->type != CElementTexture::BITMAP)
            return { nullptr, nullptr, 0.f };

        return getTexture(key, _element, scale);
    }

    auto CMitsubaMaterialCompilerFrontend::getBlendWeightTex(const CElementTexture* _element) const -> tex_ass_type
    {
        std::string key = m_loaderContext->blendWeightImageCacheKey(_element);
        float scale = 1.f;
        std::tie(_element, scale) = getTexture_common(_element);

        return getTexture(key, _element, scale);
    }

    std::pair<const CElementTexture*, float> CMitsubaMaterialCompilerFrontend::getTexture_common(const CElementTexture* _element) const
    {
        float scale = 1.f;
        while (_element->type == CElementTexture::SCALE)
        {
            scale *= _element->scale.scale;
            _element = _element->scale.texture;
        }
        _NBL_DEBUG_BREAK_IF(_element->type != CElementTexture::BITMAP);

        return { _element, scale };
    }

    auto CMitsubaMaterialCompilerFrontend::getTexture(const CElementTexture* _element) const -> tex_ass_type
    {
        float scale = 1.f;
        std::tie(_element, scale) = getTexture_common(_element);
        if (_element->type != CElementTexture::BITMAP)
            return { nullptr, nullptr, 0.f };

        return getTexture(_element->bitmap.filename.svalue, _element, scale);
    }

    auto CMitsubaMaterialCompilerFrontend::getTexture(const std::string& _key, const CElementTexture* _element, float _scale) const -> tex_ass_type
    {
        const std::string samplerKey = m_loaderContext->samplerCacheKey(_element);
        const std::string viewKey = m_loaderContext->imageViewCacheKey(_key);

        asset::IAsset::E_TYPE types[2]{ asset::IAsset::ET_IMAGE_VIEW, asset::IAsset::ET_TERMINATING_ZERO };
        auto viewBundle = m_loaderContext->override_->findCachedAsset(viewKey, types, m_loaderContext->inner, 0u);
        if (!viewBundle.getContents().empty())
        {
            auto view = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(viewBundle.getContents().begin()[0]);
            types[0] = asset::IAsset::ET_SAMPLER;
            auto samplerBundle = m_loaderContext->override_->findCachedAsset(samplerKey, types, m_loaderContext->inner, 0u);
            assert(!samplerBundle.getContents().empty());
            auto sampler = core::smart_refctd_ptr_static_cast<asset::ICPUSampler>(samplerBundle.getContents().begin()[0]);

            return {view, sampler, _scale};
        }
        return { nullptr, nullptr, _scale };
    }

    auto CMitsubaMaterialCompilerFrontend::createIRNode(asset::material_compiler::IR* ir, const CElementBSDF* _bsdf) -> IRNode*
    {
        using namespace asset;
        using namespace material_compiler;

        auto getFloatOrTexture = [this](const CElementTexture::FloatOrTexture& src, IR::INode::SParameter<float>& dst)
        {
            if (src.value.type == SPropertyElementData::INVALID)
            {
                IR::INode::STextureSource tex;
                std::tie(tex.image, tex.sampler, tex.scale) = getTexture(src.texture);
                _NBL_DEBUG_BREAK_IF(!tex.image);
                if (tex.image)
                    dst = std::move(tex);
                else
                    dst = 0.f; // 0 in case of no texture

            }
            else dst = src.value.fvalue;
        };
        auto getSpectrumOrTexture = [this](const CElementTexture::SpectrumOrTexture& src, IR::INode::SParameter<IR::INode::color_t>& dst)
        {
            if (src.value.type == SPropertyElementData::INVALID)
            {
                IR::INode::STextureSource tex;
                std::tie(tex.image, tex.sampler, tex.scale) = getTexture(src.texture);
                _NBL_DEBUG_BREAK_IF(!tex.image);
                if (tex.image)
                    dst = std::move(tex);
                else
                    dst = IR::INode::color_t(0.5f, 0.5f, 0.5f); // red in case of no texture
            }
            else dst = src.value.vvalue;
        };

        constexpr IR::CMicrofacetSpecularBSDFNode::E_NDF ndfMap[4]{
            IR::CMicrofacetSpecularBSDFNode::ENDF_BECKMANN,
            IR::CMicrofacetSpecularBSDFNode::ENDF_GGX,
            IR::CMicrofacetSpecularBSDFNode::ENDF_PHONG,
            IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY
        };

        const auto type = _bsdf->type;
        IRNode* ir_node = nullptr;
        switch (type)
        {
        case CElementBSDF::TWO_SIDED:
            //TWO_SIDED is not translated into IR node directly
            break;
        case CElementBSDF::MASK:
            ir_node = ir->allocNode<IR::COpacityNode>();
            ir_node->children.count = 1u;
            getSpectrumOrTexture(_bsdf->mask.opacity, static_cast<IR::COpacityNode*>(ir_node)->opacity);
            break;
        case CElementBSDF::DIFFUSE:
        case CElementBSDF::ROUGHDIFFUSE:
            ir_node = ir->allocNode<IR::CMicrofacetDiffuseBSDFNode>();
            getSpectrumOrTexture(_bsdf->diffuse.reflectance, static_cast<IR::CMicrofacetDiffuseBSDFNode*>(ir_node)->reflectance);
            if (type == CElementBSDF::ROUGHDIFFUSE)
            {
                getFloatOrTexture(_bsdf->diffuse.alpha, static_cast<IR::CMicrofacetDiffuseBSDFNode*>(ir_node)->alpha_u);
                static_cast<IR::CMicrofacetDiffuseBSDFNode*>(ir_node)->alpha_v = static_cast<IR::CMicrofacetDiffuseBSDFNode*>(ir_node)->alpha_u;
            }
            else
            {
                static_cast<IR::CMicrofacetDiffuseBSDFNode*>(ir_node)->setSmooth();
            }
            break;
        case CElementBSDF::CONDUCTOR:
        case CElementBSDF::ROUGHCONDUCTOR:
        {
            ir_node = ir->allocNode<IR::CMicrofacetSpecularBSDFNode>();
            auto* node = static_cast<IR::CMicrofacetSpecularBSDFNode*>(ir_node);
            node->shadowing = IR::CMicrofacetSpecularBSDFNode::EST_SMITH;
            const float extEta = _bsdf->conductor.extEta;
            node->eta = _bsdf->conductor.eta.vvalue/extEta;
            node->etaK = _bsdf->conductor.k.vvalue/extEta;

            if (type == CElementBSDF::ROUGHCONDUCTOR)
            {
                node->ndf = ndfMap[_bsdf->conductor.distribution];
                getFloatOrTexture(_bsdf->conductor.alphaU, node->alpha_u);
                if (node->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                    getFloatOrTexture(_bsdf->conductor.alphaV, node->alpha_v);
                else
                    node->alpha_v = node->alpha_u;
            }
            else
            {
                node->setSmooth();
            }
        }
        break;
        case CElementBSDF::DIFFUSE_TRANSMITTER:
        {
            ir_node = ir->allocNode<IR::CMicrofacetDifftransBSDFNode>();
            auto* node = static_cast<IR::CMicrofacetDifftransBSDFNode*>(ir_node);
            node->setSmooth();

            getSpectrumOrTexture(_bsdf->difftrans.transmittance, node->transmittance);
        }
        break;
        case CElementBSDF::PLASTIC:
        case CElementBSDF::ROUGHPLASTIC:
        {
            ir_node = ir->allocNode<IR::CMicrofacetCoatingBSDFNode>();
            auto* coat = static_cast<IR::CMicrofacetCoatingBSDFNode*>(ir_node);
            coat->children.count = 1u;

            auto& coated = ir_node->children[0];
            coated = ir->allocNode<IR::CMicrofacetDiffuseBSDFNode>();

            const float eta = _bsdf->plastic.intIOR/_bsdf->plastic.extIOR;

            coat->shadowing = IR::CMicrofacetSpecularBSDFNode::EST_SMITH;
            coat->eta = IR::INode::color_t(eta);
            if (type == CElementBSDF::ROUGHPLASTIC)
            {
                coat->ndf = ndfMap[_bsdf->plastic.distribution];
                getFloatOrTexture(_bsdf->plastic.alphaU, coat->alpha_u);
                if (coat->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                    getFloatOrTexture(_bsdf->plastic.alphaV, coat->alpha_v);
                else
                    coat->alpha_v = coat->alpha_u;
            }
            else coat->setSmooth();

            auto* node_diffuse = static_cast<IR::CMicrofacetDiffuseBSDFNode*>(coated);
            getSpectrumOrTexture(_bsdf->plastic.diffuseReflectance, node_diffuse->reflectance);
            node_diffuse->alpha_u = coat->alpha_u;
            node_diffuse->alpha_v = coat->alpha_v;
            node_diffuse->eta = coat->eta;
        }
        break;
        case CElementBSDF::DIELECTRIC:
        case CElementBSDF::THINDIELECTRIC:
        case CElementBSDF::ROUGHDIELECTRIC:
        {
            auto* dielectric = ir->allocNode<IR::CMicrofacetDielectricBSDFNode>();
            ir_node = dielectric;

            const float eta = _bsdf->dielectric.intIOR/_bsdf->dielectric.extIOR;
            _NBL_DEBUG_BREAK_IF(eta==1.f);
            if (eta==1.f)
                os::Printer::log("WARNING: Dielectric with IoR=1.0!", _bsdf->id, ELL_ERROR);

            dielectric->shadowing = IR::CMicrofacetSpecularBSDFNode::EST_SMITH;
            dielectric->eta = IR::INode::color_t(eta);
            if (type == CElementBSDF::ROUGHDIELECTRIC)
            {
                dielectric->ndf = ndfMap[_bsdf->dielectric.distribution];
                getFloatOrTexture(_bsdf->dielectric.alphaU, dielectric->alpha_u);
                if (dielectric->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                    getFloatOrTexture(_bsdf->dielectric.alphaV, dielectric->alpha_v);
                else
                    dielectric->alpha_v = dielectric->alpha_u;
            }
            else
            {
                dielectric->setSmooth();
            }

            dielectric->thin = (type == CElementBSDF::THINDIELECTRIC);
        }
        break;
        case CElementBSDF::BUMPMAP:
        {
            ir_node = ir->allocNode<IR::CGeomModifierNode>(IR::CGeomModifierNode::ET_DERIVATIVE);
            ir_node->children.count = 1u;

            auto* node = static_cast<IR::CGeomModifierNode*>(ir_node);
            //no other source supported for now (uncomment in the future) [far future TODO]
            //node->source = IR::CGeomModifierNode::ESRC_TEXTURE;

            std::tie(node->texture.image, node->texture.sampler, node->texture.scale) = getDerivMap(_bsdf->bumpmap.texture);
        }
        break;
        case CElementBSDF::COATING:
        case CElementBSDF::ROUGHCOATING:
        {
            ir_node = ir->allocNode<IR::CMicrofacetCoatingBSDFNode>();
            ir_node->children.count = 1u;

            const float eta = _bsdf->dielectric.intIOR/_bsdf->dielectric.extIOR;

            auto* node = static_cast<IR::CMicrofacetCoatingBSDFNode*>(ir_node);

            const float thickness = _bsdf->coating.thickness;
            getSpectrumOrTexture(_bsdf->coating.sigmaA, node->thicknessSigmaA);
            if (node->thicknessSigmaA.source == IR::INode::EPS_CONSTANT)
                node->thicknessSigmaA.value.constant *= thickness;
            else
                node->thicknessSigmaA.value.texture.scale *= thickness;

            node->eta = IR::INode::color_t(eta);
            node->shadowing = IR::CMicrofacetCoatingBSDFNode::EST_SMITH;
            if (type == CElementBSDF::ROUGHCOATING)
            {
                node->ndf = ndfMap[_bsdf->coating.distribution];
                getFloatOrTexture(_bsdf->coating.alphaU, node->alpha_u);
                if (node->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                    getFloatOrTexture(_bsdf->coating.alphaV, node->alpha_v);
                else
                    node->alpha_v = node->alpha_u;
            }
            else
            {
                node->setSmooth();
            }
        }
        break;
        case CElementBSDF::BLEND_BSDF:
        {
            ir_node = ir->allocNode<IR::CBSDFBlendNode>();
            ir_node->children.count = 2u;

            auto* node = static_cast<IR::CBSDFBlendNode*>(ir_node);
            if (_bsdf->blendbsdf.weight.value.type == SPropertyElementData::INVALID)
            {
                std::tie(node->weight.value.texture.image, node->weight.value.texture.sampler, node->weight.value.texture.scale) =
                    getBlendWeightTex(_bsdf->blendbsdf.weight.texture);
                node->weight.source = IR::INode::EPS_TEXTURE;
            }
            else
            {
                node->weight.value.constant = IR::INode::color_t(_bsdf->blendbsdf.weight.value.fvalue);
                node->weight.source = IR::INode::EPS_CONSTANT;
            }
        }
        break;
        case CElementBSDF::MIXTURE_BSDF:
        {
            ir_node = ir->allocNode<IR::CBSDFMixNode>();
            auto* node = static_cast<IR::CBSDFMixNode*>(ir_node);
            const size_t cnt = _bsdf->mixturebsdf.childCount;
            ir_node->children.count = cnt;
            for (int32_t i = cnt-1u; i >= 0; --i)
                node->weights[i] = _bsdf->mixturebsdf.weights[i];
        }
        break;
        }

        return ir_node;
    }

auto CMitsubaMaterialCompilerFrontend::compileToIRTree(asset::material_compiler::IR* ir, const CElementBSDF* _bsdf) -> front_and_back_t
{
    using namespace asset;
    using namespace material_compiler;

    auto getFloatOrTexture = [this](const CElementTexture::FloatOrTexture& src, IR::INode::SParameter<float>& dst)
    {
        if (src.value.type == SPropertyElementData::INVALID)
        {
            IR::INode::STextureSource tex;
            std::tie(tex.image, tex.sampler, tex.scale) = getTexture(src.texture);
            _NBL_DEBUG_BREAK_IF(!tex.image);
            if (tex.image)
                dst = std::move(tex);
            else
                dst = 0.f; // 0 in case of no texture

        }
        else dst = src.value.fvalue;
    };
    auto getSpectrumOrTexture = [this](const CElementTexture::SpectrumOrTexture& src, IR::INode::SParameter<IR::INode::color_t>& dst)
    {
        if (src.value.type == SPropertyElementData::INVALID)
        {
            IR::INode::STextureSource tex;
            std::tie(tex.image, tex.sampler, tex.scale) = getTexture(src.texture);
            _NBL_DEBUG_BREAK_IF(!tex.image);
            if (tex.image)
                dst = std::move(tex);
            else
                dst = IR::INode::color_t(1.f, 0.f, 0.f); // red in case of no texture
        }
        else dst = src.value.vvalue;
    };

    struct SNode
    {
        const CElementBSDF* bsdf;
        IRNode* ir_node = nullptr;
        uint32_t parent_ix = static_cast<uint32_t>(-1);
        uint32_t child_num = 0u;
        bool twosided = false;
        bool front = true;

        CElementBSDF::Type type() const { return bsdf->type; }
    };
    auto node_parent = [](const SNode& node, core::vector<SNode>& traversal) { return &traversal[node.parent_ix]; };

    core::vector<SNode> bfs;
    {
        core::queue<SNode> q;
        {
            SNode root{ _bsdf };
            root.twosided = (root.type() == CElementBSDF::TWO_SIDED);
            q.push(root);
        }

        while (q.size())
        {
            SNode parent = q.front();
            q.pop();
            //node.ir_node = createIRNode(node.bsdf);

            if (parent.bsdf->isMeta())
            {
                const auto& meta = parent.bsdf->meta_common;
                for (uint32_t i = 0u; i < meta.childCount; ++i)
                {
                    SNode child_node;
                    child_node.bsdf = meta.bsdf[i];
                    child_node.parent_ix = parent.type() == CElementBSDF::TWO_SIDED ? parent.parent_ix : bfs.size();
                    child_node.twosided = (child_node.type() == CElementBSDF::TWO_SIDED) || parent.twosided;
                    child_node.child_num = (parent.type() == CElementBSDF::TWO_SIDED) ? parent.child_num : i;
                    child_node.front = parent.front;
                    if (parent.type() == CElementBSDF::TWO_SIDED && i == 1u)
                        child_node.front = false;
                    q.push(child_node);
                }
            }
            if (parent.type() != CElementBSDF::TWO_SIDED)
                bfs.push_back(parent);
        }
    }

    auto createBackfaceNodeFromFrontface = [&ir](const IRNode* front) -> IRNode*
    {
        switch (front->symbol)
        {
        case IRNode::ES_BSDF_COMBINER: [[fallthrough]];
        case IRNode::ES_OPACITY: [[fallthrough]];
        case IRNode::ES_GEOM_MODIFIER: [[fallthrough]];
        case IRNode::ES_EMISSION:
            return ir->copyNode(front);
        case IRNode::ES_BSDF:
        {
            auto* bsdf = static_cast<const IR::CBSDFNode*>(front);
            if (bsdf->type == IR::CBSDFNode::ET_MICROFACET_DIELECTRIC)
            {
                auto* dielectric = static_cast<const IR::CMicrofacetDielectricBSDFNode*>(bsdf);
                auto* copy = static_cast<IR::CMicrofacetDielectricBSDFNode*>(ir->copyNode(front));
                if (!copy->thin) //we're always outside in case of thin dielectric
                    copy->eta = IRNode::color_t(1.f) / copy->eta;

                return copy;
            }
            else if (bsdf->type == IR::CBSDFNode::ET_MICROFACET_DIFFTRANS)
                return ir->copyNode(front);
        }
        [[fallthrough]]; // intentional
        default:
        {
            // black diffuse otherwise
            auto* invalid = ir->allocNode<IR::CMicrofacetDiffuseBSDFNode>();
            invalid->setSmooth();
            invalid->reflectance = IR::INode::color_t(0.f);

            return invalid;
        }
        }
    };

    //create frontface IR
    IRNode* frontroot = nullptr;
    for (auto& node : bfs)
    {
        if (!node.front)
            continue;

        IRNode** dst = nullptr;
        if (node.parent_ix >= bfs.size())
            dst = &frontroot;
        else
            dst = const_cast<IRNode**>(&node_parent(node, bfs)->ir_node->children[node.child_num]);

        node.ir_node = *dst = createIRNode(ir, node.bsdf);
    }
    IRNode* backroot = nullptr;
    for (uint32_t i = 0u; i < bfs.size(); ++i)
    {
        SNode& node = bfs[i];

        IRNode* ir_node = nullptr;
        if (!node.twosided)
            ir_node = createBackfaceNodeFromFrontface(node.ir_node);
        else
        {
            if (node.front)
            {
                if ((i+1u) < bfs.size() && bfs[i+1u].twosided && !bfs[i+1u].front)
                    continue; // will take backface node in next iteration
                //otherwise copy the one from front (same bsdf on both sides_
                ir_node = ir->copyNode(node.ir_node);
            }
            else
                ir_node = createIRNode(ir, node.bsdf);
        }
        node.ir_node = ir_node;

        IRNode** dst = nullptr;
        if (node.parent_ix >= bfs.size())
            dst = &backroot;
        else
            dst = const_cast<IRNode**>(&node_parent(node, bfs)->ir_node->children[node.child_num]);

        *dst = ir_node;
    }

    ir->addRootNode(frontroot);
    ir->addRootNode(backroot);

    return { frontroot, backroot };
}

}}}