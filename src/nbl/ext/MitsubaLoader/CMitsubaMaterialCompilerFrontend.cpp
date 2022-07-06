// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/MitsubaLoader/CMitsubaMaterialCompilerFrontend.h"
#include "nbl/ext/MitsubaLoader/SContext.h"

namespace nbl::ext::MitsubaLoader
{

auto CMitsubaMaterialCompilerFrontend::getTexture(const ext::MitsubaLoader::SContext* _loaderContext, const CElementTexture* _element, const E_IMAGE_VIEW_SEMANTIC semantic) -> tex_ass_type
{
    // first unwind the texture Scales
    float scale = 1.f;
    while (_element && _element->type==CElementTexture::SCALE)
    {
        scale *= _element->scale.scale;
        _element = _element->scale.texture;
    }
    _NBL_DEBUG_BREAK_IF(_element && _element->type!=CElementTexture::BITMAP);
    if (!_element)
    {
        os::Printer::log("[ERROR] Could Not Find Texture, dangling reference after scale unroll, substituting 2x2 Magenta Checkerboard Error Texture.", ELL_ERROR);
        return getErrorTexture(_loaderContext);
    }

    asset::IAsset::E_TYPE types[2]{ asset::IAsset::ET_IMAGE_VIEW, asset::IAsset::ET_TERMINATING_ZERO };
    const auto key = _loaderContext->imageViewCacheKey(_element->bitmap,semantic);
    auto viewBundle = _loaderContext->override_->findCachedAsset(key,types,_loaderContext->inner,0u);
    if (!viewBundle.getContents().empty())
    {
        auto view = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(viewBundle.getContents().begin()[0]);

        // TODO: here for the bumpmap bug
        auto found = _loaderContext->derivMapCache.find(view->getCreationParameters().image);
        if (found!=_loaderContext->derivMapCache.end())
        {
            const float normalizationFactor = found->second;
            scale *= normalizationFactor;
        }

        types[0] = asset::IAsset::ET_SAMPLER;
        const std::string samplerKey = _loaderContext->samplerCacheKey(_loaderContext->computeSamplerParameters(_element->bitmap));
        auto samplerBundle = _loaderContext->override_->findCachedAsset(samplerKey, types, _loaderContext->inner, 0u);
        assert(!samplerBundle.getContents().empty());
        auto sampler = core::smart_refctd_ptr_static_cast<asset::ICPUSampler>(samplerBundle.getContents().begin()[0]);

        return {view, sampler, scale};
    }
    return { nullptr, nullptr, core::nan<float>()};
}

auto CMitsubaMaterialCompilerFrontend::getErrorTexture(const ext::MitsubaLoader::SContext* _loaderContext) -> tex_ass_type
{
    constexpr const char* ERR_TEX_CACHE_NAME = "nbl/builtin/image_view/dummy2d";
    constexpr const char* ERR_SMPLR_CACHE_NAME = "nbl/builtin/sampler/default";

    asset::IAsset::E_TYPE types[2]{ asset::IAsset::ET_IMAGE_VIEW, asset::IAsset::ET_TERMINATING_ZERO };
    auto bundle = _loaderContext->override_->findCachedAsset(ERR_TEX_CACHE_NAME, types, _loaderContext->inner, 0u);
    assert(!bundle.getContents().empty()); // this shouldnt ever happen since ERR_TEX_CACHE_NAME is builtin asset
        
    auto view = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(bundle.getContents().begin()[0]);

    types[0] = asset::IAsset::ET_SAMPLER;
    auto sbundle = _loaderContext->override_->findCachedAsset(ERR_SMPLR_CACHE_NAME, types, _loaderContext->inner, 0u);
    assert(!sbundle.getContents().empty()); // this shouldnt ever happen since ERR_SMPLR_CACHE_NAME is builtin asset

    auto smplr = core::smart_refctd_ptr_static_cast<asset::ICPUSampler>(sbundle.getContents().begin()[0]);

    return { view, smplr, 1.f };
}

auto CMitsubaMaterialCompilerFrontend::createIRNode(SContext& ctx, const CElementBSDF* _bsdf) -> node_handle_t
{
    using namespace asset::material_compiler;

    auto transparent = [](const float eta) -> bool {return eta>0.99999f&&eta<1.000001f;};

    auto getFloatOrTexture = [&ctx](const CElementTexture::FloatOrTexture& src) -> IR::INode::SParameter<float>
    {
        if (src.value.type == SPropertyElementData::INVALID)
        {
            IR::INode::STextureSource tex;
            std::tie(tex.image, tex.sampler, tex.scale) = getTexture(ctx.m_loaderContext,src.texture);
            return tex;
        }
        else
            return src.value.fvalue;
    };
    auto getSpectrumOrTexture = [&ctx](const CElementTexture::SpectrumOrTexture& src, const E_IMAGE_VIEW_SEMANTIC semantic=EIVS_IDENTITIY) -> IR::INode::SParameter<IR::INode::color_t>
    {
        if (src.value.type == SPropertyElementData::INVALID)
        {
            IR::INode::STextureSource tex;
            std::tie(tex.image, tex.sampler, tex.scale) = getTexture(ctx.m_loaderContext,src.texture,semantic);
            return tex;
        }
        else
            return src.value.vvalue;
    };

    auto setAlpha = [&getFloatOrTexture](IR::IMicrofacetBSDFNode* node, const bool rough, const auto& _bsdfEl) -> void
    {
        if (rough)
        {
            using bsdf_t = std::remove_const_t<std::remove_reference_t<decltype(_bsdfEl)>>;
            if constexpr (std::is_same_v<bsdf_t,CElementBSDF::AllDiffuse> || std::is_same_v<bsdf_t,CElementBSDF::DiffuseTransmitter>)
                getFloatOrTexture(_bsdfEl.alpha,node->alpha_u);
            else
                getFloatOrTexture(_bsdfEl.alphaU,node->alpha_u);
            node->alpha_v = node->alpha_u;
            if constexpr (std::is_base_of_v<CElementBSDF::RoughSpecularBase,bsdf_t>)
            {
                constexpr IR::CMicrofacetSpecularBSDFNode::E_NDF ndfMap[4] =
                {
                    IR::CMicrofacetSpecularBSDFNode::ENDF_BECKMANN,
                    IR::CMicrofacetSpecularBSDFNode::ENDF_GGX,
                    IR::CMicrofacetSpecularBSDFNode::ENDF_PHONG,
                    IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY
                };
                auto& ndf = static_cast<IR::ICookTorranceBSDFNode*>(node)->ndf;
                ndf = ndfMap[_bsdfEl.distribution];
                if (ndf==IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                    getFloatOrTexture(_bsdfEl.alphaV,node->alpha_v);
            }
        }
        else
            node->setSmooth();
    };

    auto ir_node = IR::invalid_node;
    auto& hashCons = ctx.m_hashCons;
    auto findAndSetChild = [&hashCons,_bsdf](IR::INode* pNode, const uint32_t childIx) -> node_handle_t
    {
        auto child = _bsdf->type==CElementBSDF::COATING ? _bsdf->coating.bsdf[childIx]:_bsdf->meta_common.bsdf[childIx];
        return pNode->getChildrenArray()[childIx] = std::get<node_handle_t>(*hashCons.find(child));
    };
    auto* ir = ctx.m_ir;
    const auto type = _bsdf->type;
    switch (type)
    {
        case CElementBSDF::MASK:
        {
            IR::INode::SParameter<IR::INode::color_t> opacity = getSpectrumOrTexture(_bsdf->mask.opacity,EIVS_BLEND_WEIGHT);
            ir_node = ir->allocNode<IR::COpacityNode>(1);
            auto pNode = ir->getNode<IR::COpacityNode>(ir_node);
            pNode->opacity = std::move(opacity);
            findAndSetChild(pNode,0u);
            break;
        }
        case CElementBSDF::DIFFUSE:
        case CElementBSDF::ROUGHDIFFUSE:
        {
            ir_node = ir->allocNode<IR::CMicrofacetDiffuseBSDFNode>(0);
            auto pNode = ir->getNode<IR::CMicrofacetDiffuseBSDFNode>(ir_node);
            setAlpha(pNode,type==CElementBSDF::ROUGHDIFFUSE,_bsdf->diffuse);
            pNode->reflectance = getSpectrumOrTexture(_bsdf->diffuse.reflectance);
            break;
        }
        case CElementBSDF::CONDUCTOR:
        case CElementBSDF::ROUGHCONDUCTOR:
        {
            ir_node = ir->allocNode<IR::CMicrofacetSpecularBSDFNode>(0);
            auto pNode = ir->getNode<IR::CMicrofacetSpecularBSDFNode>(ir_node);
            setAlpha(pNode,type==CElementBSDF::ROUGHCONDUCTOR,_bsdf->conductor);
            const float extEta = _bsdf->conductor.extEta;
            pNode->eta = _bsdf->conductor.eta.vvalue/extEta;
            pNode->etaK = _bsdf->conductor.k.vvalue/extEta;
            // TODO: first check if eta=1 or etaK=INF (no idea actually what would cause pitch black fresnel), then replace with NOOP
            break;
        }
        case CElementBSDF::DIFFUSE_TRANSMITTER:
        {
            ir_node = ir->allocNode<IR::CMicrofacetDifftransBSDFNode>(0);
            auto pNode = ir->getNode<IR::CMicrofacetDifftransBSDFNode>(ir_node);
            pNode->setSmooth();
            pNode->transmittance = getSpectrumOrTexture(_bsdf->difftrans.transmittance);
            break;
        }
        case CElementBSDF::PLASTIC:
        case CElementBSDF::ROUGHPLASTIC:
        {
            const bool rough = type==CElementBSDF::ROUGHPLASTIC;

            ir_node = ir->allocNode<IR::CMicrofacetCoatingBSDFNode>(1);
            auto coat = ir->getNode<IR::CMicrofacetCoatingBSDFNode>(ir_node);
            setAlpha(coat,rough,_bsdf->plastic);
            coat->eta = IR::INode::color_t(_bsdf->plastic.intIOR/_bsdf->plastic.extIOR);

            {
                CElementBSDF tmp("impl_tmp_diffuse_element");
                tmp.type = rough ? CElementBSDF::ROUGHDIFFUSE:CElementBSDF::DIFFUSE;
                tmp.diffuse.alpha = _bsdf->plastic.alpha;
                tmp.diffuse.reflectance = _bsdf->plastic.diffuseReflectance;
                coat->getChildrenArray()[0] = createIRNode(ctx,&tmp);
            }
            break;
        }
        case CElementBSDF::DIELECTRIC:
        case CElementBSDF::THINDIELECTRIC:
        case CElementBSDF::ROUGHDIELECTRIC:
        {
            ir_node = ir->allocNode<IR::CMicrofacetDielectricBSDFNode>(0);
            auto dielectric = ir->getNode<IR::CMicrofacetDielectricBSDFNode>(ir_node);
            setAlpha(dielectric,type==CElementBSDF::ROUGHDIELECTRIC,_bsdf->dielectric);
            dielectric->eta = IR::INode::color_t(_bsdf->dielectric.intIOR/_bsdf->dielectric.extIOR);
            dielectric->thin = type==CElementBSDF::THINDIELECTRIC;
            break;
        }
        case CElementBSDF::BUMPMAP:
        {
            ir_node = ir->allocNode<IR::CGeomModifierNode>(1,IR::CGeomModifierNode::ET_DERIVATIVE);
            auto* pNode = ir->getNode<IR::CGeomModifierNode>(ir_node);
            //no other source supported for now (uncomment in the future) [far future TODO]
            //node->source = IR::CGeomModifierNode::ESRC_TEXTURE;

            std::tie(pNode->texture.image,pNode->texture.sampler,pNode->texture.scale) =
                getTexture(ctx.m_loaderContext,_bsdf->bumpmap.texture,_bsdf->bumpmap.wasNormal ? EIVS_NORMAL_MAP:EIVS_BUMP_MAP);
            
            findAndSetChild(pNode,0);
            break;
        }
        case CElementBSDF::COATING:
        case CElementBSDF::ROUGHCOATING:
        {
            const bool rough = type==CElementBSDF::ROUGHDIELECTRIC;

            ir_node = ir->allocNode<IR::CMicrofacetCoatingBSDFNode>(1u);
            auto* coat = ir->getNode<IR::CMicrofacetCoatingBSDFNode>(ir_node);
            setAlpha(coat,rough,_bsdf->coating);

            coat->eta = IR::INode::color_t(_bsdf->dielectric.intIOR/_bsdf->dielectric.extIOR);

            const float thickness = _bsdf->coating.thickness;
            coat->thicknessSigmaA = getSpectrumOrTexture(_bsdf->coating.sigmaA);
            if (coat->thicknessSigmaA.isConstant())
                coat->thicknessSigmaA.constant *= thickness;
            else
                coat->thicknessSigmaA.texture.scale *= thickness;
            
            findAndSetChild(coat,0);
            break;
        }
        break;
        case CElementBSDF::BLEND_BSDF:
        {
            ir_node = ir->allocNode<IR::CBSDFBlendNode>(2);
            auto* pNode = ir->getNode<IR::CBSDFBlendNode>(ir_node);
            if (_bsdf->blendbsdf.weight.value.type == SPropertyElementData::INVALID)
            {
                std::tie(pNode->weight.texture.image,pNode->weight.texture.sampler,pNode->weight.texture.scale) =
                    getTexture(ctx.m_loaderContext,_bsdf->blendbsdf.weight.texture,EIVS_BLEND_WEIGHT);
                assert(!core::isnan(pNode->weight.texture.scale));
            }
            else
                pNode->weight = IR::INode::color_t(_bsdf->blendbsdf.weight.value.fvalue);
            findAndSetChild(pNode,0);
            findAndSetChild(pNode,1);
        }
        break;
        case CElementBSDF::MIXTURE_BSDF:
        {
            ir_node = ir->allocNode<IR::CBSDFMixNode>(_bsdf->mixturebsdf.childCount);
            auto* pNode = ir->getNode<IR::CBSDFMixNode>(ir_node);
            const auto* weightIt = _bsdf->mixturebsdf.weights;
            for (size_t i=0u; i<pNode->getChildCount(); i++)
            {
                pNode->weights[i] = *(weightIt++);
                findAndSetChild(pNode,i);
            }
            break;
        }
        default:
            break;
    }
    assert(ir_node!=IR::invalid_node);
    return ir_node;
}

auto CMitsubaMaterialCompilerFrontend::compileToIRTree(SContext& ctx, const CElementBSDF* _root) -> front_and_back_t
{
    using namespace asset;
    using namespace material_compiler;

    auto frontroot = IR::invalid_node;
    auto backroot = IR::invalid_node;
    //create frontface IR
    struct DFSData
    {
        const CElementBSDF* bsdf;
        // most BxDFs have different appearance depending on NdotV, if "twosided" we behave as-if `NdotV` is always positive
        // since <twosided> is a nesting Meta-BxDF it affects all of its tree-branch, ergo the setting is propagated
        uint8_t twosided : 1;
        // to do Post-Order DFS we need to visit parent twice
        uint8_t visited : 1;

        CElementBSDF::Type type() const { return bsdf->type; }
    };
    core::stack<DFSData> dfs;
    auto pre = [&dfs,&frontroot,&ctx](const CElementBSDF* _bsdf, const bool twosidedParent)
    {
        DFSData el;
        // unwind twosided
        for (el.bsdf=_bsdf; el.bsdf->type==CElementBSDF::TWO_SIDED; _bsdf=_bsdf->meta_common.bsdf[0])
        {
            // sanity checks
            static_assert(_bsdf->twosided.MaxChildCount == 1);
            assert(_bsdf->meta_common.childCount==1);
            assert(_bsdf->twosided.childCount==1);
        }
        el.twosided = twosidedParent || el.bsdf!=_bsdf;
        // only meta nodes get pushed onto stack
        if (el.bsdf->isMeta())
        {
            el.visited = false;
            dfs.push(std::move(el));
        }
        else
            frontroot = createIRNode(ctx,el.bsdf);
    };
    pre(_root,false);
    while (!dfs.empty())
    {
        auto& parent = dfs.top();
        assert(parent.type()!=CElementBSDF::TWO_SIDED);
        assert(parent.bsdf->isMeta());

        if (parent.visited)
        {
            dfs.pop();
            frontroot = createIRNode(ctx,parent.bsdf);
        }
        else
        {
            parent.visited = true;

            const bool isCoating = parent.type()==CElementBSDF::COATING;
            const auto childCount = isCoating ? parent.bsdf->coating.childCount:parent.bsdf->meta_common.childCount;
            for (auto i=0u; i<childCount; i++)
            {
                const auto child = isCoating ? parent.bsdf->coating.bsdf[i]:parent.bsdf->meta_common.bsdf[i];
                pre(child,parent.twosided);
            }
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
                auto* copy = ir->copyNode<IR::CMicrofacetDielectricBSDFNode>(front);
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

    IRNode* backroot = nullptr;
    for (uint32_t i = 0u; i < bfs.size(); ++i)
    {
        SNode& node = bfs[i];

        IRNode* ir_node = nullptr;
        if (!node.twosided)
            ir_node = createBackfaceNodeFromFrontface(node.ir_node);
        else
        {
            ir_node = ir->copyNode(node.ir_node);
        }
        node.ir_node = ir_node;

        IRNode** dst = nullptr;
        if (node.parent_ix >= bfs.size())
            dst = &backroot;
        else
            dst = const_cast<IRNode**>(&node_parent(node, bfs)->ir_node->children[node.child_num]);

        *dst = ir_node;
    }

    ctx.m_ir->addRootNode(frontroot);
    ctx.m_ir->addRootNode(backroot);
    return {frontroot,backroot};
}

}