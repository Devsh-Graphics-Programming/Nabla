#include "irr/ext/MitsubaLoader/CMitsubaMaterialCompilerFrontend.h"
#include "irr/ext/MitsubaLoader/SContext.h"

#include <irr/core/Types.h>
#include <irr/asset/filters/kernels/CGaussianImageFilterKernel.h>
#include <irr/asset/filters/kernels/CDerivativeImageFilterKernel.h>
#include <irr/asset/filters/kernels/CBoxImageFilterKernel.h>
#include <irr/asset/filters/kernels/CChannelIndependentImageFilterKernel.h>
#include <irr/asset/filters/CMipMapGenerationImageFilter.h>

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

    auto CMitsubaMaterialCompilerFrontend::getDerivMap(const CElementTexture* _element) const -> tex_ass_type
    {
        std::string key = m_loaderContext->derivMapCacheKey(_element);
        float scale = 1.f;
        std::tie(_element, scale) = getTexture_common(_element);
        if (_element->type != CElementTexture::BITMAP)
            return { nullptr, nullptr, 0.f };

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
        _IRR_DEBUG_BREAK_IF(_element->type != CElementTexture::BITMAP);

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

        asset::IAsset::E_TYPE types[2]{ asset::IAsset::ET_IMAGE_VIEW, static_cast<asset::IAsset::E_TYPE>(0) };
        auto viewBundle = m_loaderContext->override_->findCachedAsset(viewKey, types, m_loaderContext->inner, 0u);
        assert(!viewBundle.isEmpty());
        auto view = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(viewBundle.getContents().begin()[0]);
        types[0] = asset::IAsset::ET_SAMPLER;
        auto samplerBundle = m_loaderContext->override_->findCachedAsset(samplerKey, types, m_loaderContext->inner, 0u);
        assert(!samplerBundle.isEmpty());
        auto sampler = core::smart_refctd_ptr_static_cast<asset::ICPUSampler>(samplerBundle.getContents().begin()[0]);

        return {view, sampler, _scale};
    }

    asset::material_compiler::IR::INode* CMitsubaMaterialCompilerFrontend::compileToIRTree(asset::material_compiler::IR* ir, const CElementBSDF* _bsdf)
{
    using namespace asset;
    using namespace material_compiler;

    auto getFloatOrTexture = [this](const CElementTexture::FloatOrTexture& src, IR::INode::SParameter<float>& dst)
    {
        if (src.value.type == SPropertyElementData::INVALID)
        {
            dst.source = IR::INode::EPS_TEXTURE;
            //making sure smart_refctd_ptr assignment wont try to drop() -- .value is union
            dst.value.constant = 0.f;
            std::tie(dst.value.texture.image, dst.value.texture.sampler, dst.value.texture.scale) = getTexture(src.texture);
            if (!dst.value.texture.image) {
                assert(!dst.value.texture.sampler);
                dst.source = IR::INode::EPS_CONSTANT;
                dst.value.constant = 0.f;//0 in case when didnt find texture
            }

        }
        else
        {
            dst.source = IR::INode::EPS_CONSTANT;
            dst.value.constant = src.value.fvalue;
        }
    };
    auto getSpectrumOrTexture = [this](const CElementTexture::SpectrumOrTexture& src, IR::INode::SParameter<IR::INode::color_t>& dst)
    {
        if (src.value.type == SPropertyElementData::INVALID)
        {
            dst.source = IR::INode::EPS_TEXTURE;
            //making sure smart_refctd_ptr assignment wont try to drop() -- .value is union
            dst.value.constant = IR::INode::color_t(0.f);
            std::tie(dst.value.texture.image, dst.value.texture.sampler, dst.value.texture.scale) = getTexture(src.texture);
            if (!dst.value.texture.image) {
                assert(!dst.value.texture.sampler);
                dst.source = IR::INode::EPS_CONSTANT;
                dst.value.constant = IR::INode::color_t(1.f, 0.f, 0.f);//red in case when didnt find texture
            }
        }
        else
        {
            dst.source = IR::INode::EPS_CONSTANT;
            dst.value.constant = src.value.vvalue;
        }
    };

    constexpr IR::CMicrofacetSpecularBSDFNode::E_NDF ndfMap[4]{
        IR::CMicrofacetSpecularBSDFNode::ENDF_BECKMANN,
        IR::CMicrofacetSpecularBSDFNode::ENDF_GGX,
        IR::CMicrofacetSpecularBSDFNode::ENDF_PHONG,
        IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY
    };

    IR::INode* root = ir->allocRootNode<IR::CMaterialNode>();
    root->children.count = 1u;

    bool twosided = false;
    IR::INode::SParameter<IR::INode::color_t> opacity;
    {
        opacity.source = IR::INode::EPS_CONSTANT;
        opacity.value.constant = IR::INode::color_t(1.f);
    }
    bool thin = false;

    const CElementBSDF* current = _bsdf;

    core::queue<const CElementBSDF*> bsdfQ;
    bsdfQ.push(_bsdf);
    core::queue<IR::INode*> nodeQ;
    //nodeQ.push(root);
    uint32_t childrenCountdown = 1u;
    IR::INode* parent = root;//nodeQ.front();

    while (!bsdfQ.empty())
    {
        current = bsdfQ.front();
        bsdfQ.pop();

        IR::INode* nextSym;
        const auto currType = current->type;
        switch (currType)
        {
        case CElementBSDF::TWO_SIDED:
            twosided = true;
            bsdfQ.push(current->twosided.bsdf[0]);
            continue;
            break;
        case CElementBSDF::MASK:
            getSpectrumOrTexture(current->mask.opacity, opacity);
            bsdfQ.push(current->mask.bsdf[0]);
            continue;
            break;
        case CElementBSDF::DIFFUSE:
        case CElementBSDF::ROUGHDIFFUSE:
            nextSym = ir->allocNode<IR::CMicrofacetDiffuseBSDFNode>();
            getSpectrumOrTexture(current->diffuse.reflectance, static_cast<IR::CMicrofacetDiffuseBSDFNode*>(nextSym)->reflectance);
            if (currType==CElementBSDF::ROUGHDIFFUSE)
            {
                getFloatOrTexture(current->diffuse.alpha, static_cast<IR::CMicrofacetDiffuseBSDFNode*>(nextSym)->alpha_u);
                static_cast<IR::CMicrofacetDiffuseBSDFNode*>(nextSym)->alpha_v = static_cast<IR::CMicrofacetDiffuseBSDFNode*>(nextSym)->alpha_u;
            }
            else
            {
                static_cast<IR::CMicrofacetDiffuseBSDFNode*>(nextSym)->setSmooth();
            }
            break;
        case CElementBSDF::CONDUCTOR:
        case CElementBSDF::ROUGHCONDUCTOR:
        {
            nextSym = ir->allocNode<IR::CMicrofacetSpecularBSDFNode>();
            auto* node = static_cast<IR::CMicrofacetSpecularBSDFNode*>(nextSym);
            node->shadowing = IR::CMicrofacetSpecularBSDFNode::EST_SMITH;
            node->scatteringMode = IR::CMicrofacetSpecularBSDFNode::ESM_REFLECT;
            const float extEta = current->conductor.extEta;
            node->eta = current->conductor.eta.vvalue/extEta;
            node->etaK = current->conductor.k.vvalue/extEta;

            if (currType == CElementBSDF::ROUGHCONDUCTOR)
            {
                node->ndf = ndfMap[current->conductor.distribution];
                getFloatOrTexture(current->conductor.alphaU, node->alpha_u);
                if (node->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                    getFloatOrTexture(current->conductor.alphaV, node->alpha_v);
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
            nextSym = ir->allocNode<IR::CDifftransBSDFNode>();
            auto* node = static_cast<IR::CDifftransBSDFNode*>(nextSym);

            getSpectrumOrTexture(current->difftrans.transmittance, node->transmittance);
        }
        break;
        case CElementBSDF::PLASTIC:
        case CElementBSDF::ROUGHPLASTIC:
        {
            nextSym = ir->allocNode<IR::CCoatingBSDFNode>();
            auto* coat = static_cast<IR::CCoatingBSDFNode*>(nextSym);
            coat->children.count = 1u;

            auto& coated = nextSym->children[0];
            coated = ir->allocNode<IR::CMicrofacetDiffuseBSDFNode>();

            const float eta = current->plastic.intIOR/current->plastic.extIOR;

            coat->shadowing = IR::CMicrofacetSpecularBSDFNode::EST_SMITH;
            coat->eta = IR::INode::color_t(eta);
            coat->scatteringMode = IR::CMicrofacetSpecularBSDFNode::ESM_REFLECT;
            if (currType == CElementBSDF::ROUGHPLASTIC)
            {
                coat->ndf = ndfMap[current->plastic.distribution];
                getFloatOrTexture(current->plastic.alphaU, coat->alpha_u);
                if (coat->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                    getFloatOrTexture(current->plastic.alphaV, coat->alpha_v);
                else
                    coat->alpha_v = coat->alpha_u;
            }
            else coat->setSmooth();

            auto* node_diffuse = static_cast<IR::CMicrofacetDiffuseBSDFNode*>(coated);
            getSpectrumOrTexture(current->plastic.diffuseReflectance, node_diffuse->reflectance);
            node_diffuse->alpha_u = coat->alpha_u;
            node_diffuse->alpha_v = coat->alpha_v;
            node_diffuse->eta = coat->eta;
        }
        break;
        case CElementBSDF::DIELECTRIC:
        case CElementBSDF::THINDIELECTRIC:
        case CElementBSDF::ROUGHDIELECTRIC:
        {
            auto* dielectric = ir->allocNode<IR::CDielectricBSDFNode>();
            nextSym = dielectric;

            const float eta = current->dielectric.intIOR/current->dielectric.extIOR;
            _IRR_DEBUG_BREAK_IF(eta==1.f);
            if (eta==1.f)
                os::Printer::log("WARNING: Dielectric with IoR=1.0!", current->id, ELL_ERROR);

            dielectric->scatteringMode = IR::CMicrofacetSpecularBSDFNode::ESM_REFLECT;
            dielectric->shadowing = IR::CMicrofacetSpecularBSDFNode::EST_SMITH;
            dielectric->eta = IR::INode::color_t(eta);
            if (currType == CElementBSDF::ROUGHDIELECTRIC)
            {
                dielectric->ndf = ndfMap[current->dielectric.distribution];
                getFloatOrTexture(current->dielectric.alphaU, dielectric->alpha_u);
                if (dielectric->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                    getFloatOrTexture(current->dielectric.alphaV, dielectric->alpha_v);
                else
                    dielectric->alpha_v = dielectric->alpha_u;
            }
            else
            {
                dielectric->setSmooth();
            }

            thin = (currType == CElementBSDF::THINDIELECTRIC);
        }
        break;
        case CElementBSDF::BUMPMAP:
        {
            nextSym = ir->allocNode<IR::CGeomModifierNode>(IR::CGeomModifierNode::ET_DERIVATIVE);
            nextSym->children.count = 1u;

            auto* node = static_cast<IR::CGeomModifierNode*>(nextSym);
            //no other source supported for now (uncomment in the future) [far future TODO]
            //node->source = IR::CGeomModifierNode::ESRC_TEXTURE;

            std::tie(node->texture.image, node->texture.sampler, node->texture.scale) = getDerivMap(current->bumpmap.texture);
            bsdfQ.push(current->bumpmap.bsdf[0]);
        }
        break;
        case CElementBSDF::COATING:
        case CElementBSDF::ROUGHCOATING:
        {
            nextSym = ir->allocNode<IR::CCoatingBSDFNode>();
            nextSym->children.count = 1u;

            const float eta = current->dielectric.intIOR/current->dielectric.extIOR;

            auto* node = static_cast<IR::CCoatingBSDFNode*>(nextSym);

            const float thickness = current->coating.thickness;
            getSpectrumOrTexture(current->coating.sigmaA, node->thicknessSigmaA);
            if (node->thicknessSigmaA.source == IR::INode::EPS_CONSTANT)
                node->thicknessSigmaA.value.constant *= thickness;
            else
                node->thicknessSigmaA.value.texture.scale *= thickness;

            node->eta = IR::INode::color_t(eta);
            node->shadowing = IR::CCoatingBSDFNode::EST_SMITH;
            if (currType == CElementBSDF::ROUGHCOATING)
            {
                node->ndf = ndfMap[current->coating.distribution];
                getFloatOrTexture(current->coating.alphaU, node->alpha_u);
                if (node->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                    getFloatOrTexture(current->coating.alphaV, node->alpha_v);
                else
                    node->alpha_v = node->alpha_u;
            }
            else
            {
                node->setSmooth();
            }
            bsdfQ.push(current->coating.bsdf[0]);
        }
        break;
        case CElementBSDF::BLEND_BSDF:
        {
            nextSym = ir->allocNode<IR::CBSDFBlendNode>();
            nextSym->children.count = 2u;

            auto* node = static_cast<IR::CBSDFBlendNode*>(nextSym);
            getFloatOrTexture(current->blendbsdf.weight, node->weight);

            bsdfQ.push(current->blendbsdf.bsdf[1]);
            bsdfQ.push(current->blendbsdf.bsdf[0]);
        }
        break;
        case CElementBSDF::MIXTURE_BSDF:
        {
            nextSym = ir->allocNode<IR::CBSDFMixNode>();
            auto* node = static_cast<IR::CBSDFMixNode*>(nextSym);
            const size_t cnt = current->mixturebsdf.childCount;
            nextSym->children.count = cnt;
            for (int32_t i = cnt-1u; i >= 0; --i)
                node->weights[i] = current->mixturebsdf.weights[i];

            for (int32_t i = cnt-1u; i >= 0; --i)
                bsdfQ.push(current->mixturebsdf.bsdf[i]);
        }
        break;
        }

        IR::INode* newParent = nullptr;
        if (nextSym->children)
            nodeQ.push(nextSym);
        if (!--childrenCountdown && !nodeQ.empty())
        {
            newParent = nodeQ.front();
            nodeQ.pop();
        }

        parent->children[childrenCountdown] = nextSym;
        if (newParent)
        {
            parent = newParent;
            childrenCountdown = parent->children.count;
        }
    }

    static_cast<IR::CMaterialNode*>(root)->opacity = opacity;
    static_cast<IR::CMaterialNode*>(root)->thin = thin;

    IR::INode* surfParent = root;
    if (surfParent->children[0]->symbol == IR::INode::ES_GEOM_MODIFIER)
        surfParent = surfParent->children[0];

    IR::INode::children_array_t surfaces;
    surfaces.count = twosided?2u:1u;
    surfaces[0] = ir->allocNode<IR::INode>(IR::INode::ES_FRONT_SURFACE);
    surfaces[0]->children = surfParent->children;
    if (surfaces.count>1u) {
        surfaces[1] = ir->allocNode<IR::INode>(IR::INode::ES_BACK_SURFACE);
        surfaces[1]->children = surfParent->children;
    }
    surfParent->children = surfaces;

    return root;
}

}}}