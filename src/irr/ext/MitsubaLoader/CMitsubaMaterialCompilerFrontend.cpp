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
    template<class Kernel>
    class MyKernel : public asset::CFloatingPointSeparableImageFilterKernelBase<MyKernel<Kernel>>
    {
        using Base = asset::CFloatingPointSeparableImageFilterKernelBase<MyKernel<Kernel>>;

        Kernel kernel;
        float multiplier;

    public:
        using value_type = typename Base::value_type;

        MyKernel(Kernel&& k, float _imgExtent) : Base(k.negative_support.x, k.positive_support.x), kernel(std::move(k)), multiplier(_imgExtent) {}

        // no special user data by default
        inline const asset::IImageFilterKernel::UserData* getUserData() const { return nullptr; }

        inline float weight(float x, int32_t channel) const
        {
            return kernel.weight(x, channel)*multiplier;
        }

        _IRR_STATIC_INLINE_CONSTEXPR bool has_derivative = false;

        IRR_DECLARE_DEFINE_CIMAGEFILTER_KERNEL_PASS_THROUGHS(Base)
    };

    static core::smart_refctd_ptr<asset::ICPUImage> createDerivMapFromHeightMap(asset::ICPUImage* _inImg, asset::ISampler::E_TEXTURE_CLAMP _uwrap, asset::ISampler::E_TEXTURE_CLAMP _vwrap, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor)
    {
        using namespace asset;

        auto getRGformat = [](asset::E_FORMAT f) -> asset::E_FORMAT {
            const uint32_t bytesPerChannel = (getBytesPerPixel(f) * core::rational(1, getFormatChannelCount(f))).getIntegerApprox();
            switch (bytesPerChannel)
            {
            case 1u:
#ifndef DERIV_MAP_FLOAT32
                return asset::EF_R8G8_UNORM;
#else
                _IRR_FALLTHROUGH;
#endif
            case 2u:
#ifndef DERIV_MAP_FLOAT32
                return asset::EF_R16G16_SFLOAT;
#else
                _IRR_FALLTHROUGH;
#endif
            case 4u:
                return asset::EF_R32G32_SFLOAT;
            case 8u:
                return asset::EF_R64G64_SFLOAT;
            default:
                return asset::EF_UNKNOWN;
            }
        };

        using ReconstructionKernel = CGaussianImageFilterKernel<>; // or Mitchell
        using DerivKernel_ = CDerivativeImageFilterKernel<ReconstructionKernel>;
        using DerivKernel = MyKernel<DerivKernel_>;
        using XDerivKernel = CChannelIndependentImageFilterKernel<DerivKernel, CBoxImageFilterKernel>;
        using YDerivKernel = CChannelIndependentImageFilterKernel<CBoxImageFilterKernel, DerivKernel>;
        using DerivativeMapFilter = CBlitImageFilter
        <
            false, false, DefaultSwizzle, IdentityDither, // (Criss, look at impl::CSwizzleAndConvertImageFilterBase)
            XDerivKernel,
            YDerivKernel,
            CBoxImageFilterKernel
        >;

        const auto extent = _inImg->getCreationParameters().extent;
        const float mlt = static_cast<float>( std::max(extent.width, extent.height) );
        XDerivKernel xderiv{ DerivKernel(DerivKernel_(ReconstructionKernel()), mlt), CBoxImageFilterKernel() };
        YDerivKernel yderiv{ CBoxImageFilterKernel(), DerivKernel(DerivKernel_(ReconstructionKernel()), mlt) };

        using swizzle_t = asset::ICPUImageView::SComponentMapping;
        DerivativeMapFilter::state_type state(std::move(xderiv), std::move(yderiv), CBoxImageFilterKernel());

        state.swizzle = { swizzle_t::ES_R, swizzle_t::ES_R, swizzle_t::ES_R, swizzle_t::ES_R };

        const auto& inParams = _inImg->getCreationParameters();
        auto outParams = inParams;
        outParams.format = getRGformat(outParams.format);
        auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(asset::getTexelOrBlockBytesize(outParams.format) * outParams.extent.width * outParams.extent.height);
        asset::ICPUImage::SBufferCopy region;
        region.imageOffset = { 0,0,0 };
        region.imageExtent = outParams.extent;
        region.imageSubresource.baseArrayLayer = 0u;
        region.imageSubresource.layerCount = 1u;
        region.imageSubresource.mipLevel = 0u;
        region.bufferRowLength = outParams.extent.width;
        region.bufferImageHeight = 0u;
        region.bufferOffset = 0u;
        auto outImg = asset::ICPUImage::create(std::move(outParams));
        outImg->setBufferAndRegions(std::move(buffer), core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(1ull, region));

        state.inOffset = { 0,0,0 };
        state.inBaseLayer = 0u;
        state.outOffset = { 0,0,0 };
        state.outBaseLayer = 0u;
        state.inExtent = inParams.extent;
        state.outExtent = state.inExtent;
        state.inLayerCount = 1u;
        state.outLayerCount = 1u;
        state.inMipLevel = 0u;
        state.outMipLevel = 0u;
        state.inImage = _inImg;
        state.outImage = outImg.get();
        state.axisWraps[0] = _uwrap;
        state.axisWraps[1] = _vwrap;
        state.axisWraps[2] = asset::ISampler::ETC_CLAMP_TO_EDGE;
        state.borderColor = _borderColor;
        state.scratchMemoryByteSize = DerivativeMapFilter::getRequiredScratchByteSize(&state);
        state.scratchMemory = reinterpret_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(state.scratchMemoryByteSize, _IRR_SIMD_ALIGNMENT));

        DerivativeMapFilter::execute(&state);

        _IRR_ALIGNED_FREE(state.scratchMemory);

        return outImg;
    }
    static core::smart_refctd_ptr<asset::ICPUImageView> createImageView(core::smart_refctd_ptr<asset::ICPUImage>&& _img)
    {
        const auto& iparams = _img->getCreationParameters();

        asset::ICPUImageView::SCreationParams params;
        params.format = iparams.format;
        params.subresourceRange.baseArrayLayer = 0u;
        params.subresourceRange.layerCount = iparams.arrayLayers;
        assert(params.subresourceRange.layerCount==1u);
        params.subresourceRange.baseMipLevel = 0u;
        params.subresourceRange.levelCount = iparams.mipLevels;
        params.viewType = asset::IImageView<asset::ICPUImage>::ET_2D;
        params.flags = static_cast<asset::IImageView<asset::ICPUImage>::E_CREATE_FLAGS>(0);
        params.image = std::move(_img);

        return asset::ICPUImageView::create(std::move(params));
    }
    static core::smart_refctd_ptr<asset::ICPUImageView> createDerivMap(asset::ICPUImage* _heightMap, asset::ICPUSampler* _smplr)
    {
        const auto& sp = _smplr->getParams();
        return createImageView(
            createDerivMapFromHeightMap(
                _heightMap, 
                static_cast<asset::ICPUSampler::E_TEXTURE_CLAMP>(sp.TextureWrapU),
                static_cast<asset::ICPUSampler::E_TEXTURE_CLAMP>(sp.TextureWrapV),
                static_cast<asset::ICPUSampler::E_TEXTURE_BORDER_COLOR>(sp.BorderColor)
            )
        );
    }

    auto CMitsubaMaterialCompilerFrontend::getTexture(const CElementTexture* _element) const -> tex_ass_type
    {
        static_assert(std::is_same_v<tex_ass_type, SContext::tex_ass_type>, "These types must be same!");

        auto found = m_loaderContext->textureCache.find(_element);
        if (found == m_loaderContext->textureCache.end())
            return tex_ass_type(nullptr, nullptr, 0.f);

        return found->second;
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
            //make sure smart_refctd_ptr assignment ptr wont try to drop() -- .value is union
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
            //make sure smart_refctd_ptr assignment ptr wont try to drop() -- .value is union
            dst.value.constant = IR::INode::color_t(0.f);
            std::tie(dst.value.texture.image, dst.value.texture.sampler, dst.value.texture.scale) = getTexture(src.texture);
            if (!dst.value.texture.image) { //TODO should be using loader override here
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
                getFloatOrTexture(current->diffuse.alpha, static_cast<IR::CMicrofacetDiffuseBSDFNode*>(nextSym)->alpha_v);
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
            nextSym = ir->allocNode<IR::CBSDFCombinerNode>(IR::CBSDFCombinerNode::ET_DIFFUSE_AND_SPECULAR);
            nextSym->children.count = 2u;

            const float eta = current->plastic.intIOR/current->plastic.extIOR;

            auto& diffuse = nextSym->children[0];
            diffuse = ir->allocNode<IR::CMicrofacetDiffuseBSDFNode>();
            auto& specular = nextSym->children[1];
            specular = ir->allocNode<IR::CMicrofacetSpecularBSDFNode>();

            auto* node_specular = static_cast<IR::CMicrofacetSpecularBSDFNode*>(specular);
            node_specular->shadowing = IR::CMicrofacetSpecularBSDFNode::EST_SMITH;
            node_specular->eta = IR::INode::color_t(eta);
            node_specular->scatteringMode = IR::CMicrofacetSpecularBSDFNode::ESM_REFLECT;
            if (currType == CElementBSDF::ROUGHPLASTIC)
            {
                node_specular->ndf = ndfMap[current->plastic.distribution];
                getFloatOrTexture(current->plastic.alphaU, node_specular->alpha_u);
                if (node_specular->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                    getFloatOrTexture(current->plastic.alphaV, node_specular->alpha_v);
                else
                    node_specular->alpha_v = node_specular->alpha_u;
            }
            else
            {
                node_specular->setSmooth();
            }
            auto* node_diffuse = static_cast<IR::CMicrofacetDiffuseBSDFNode*>(diffuse);
            getSpectrumOrTexture(current->plastic.diffuseReflectance, node_diffuse->reflectance);
            node_diffuse->alpha_u = node_specular->alpha_u;
            node_diffuse->alpha_v = node_specular->alpha_v;
            node_diffuse->eta = node_specular->eta;
        }
        break;
        case CElementBSDF::DIELECTRIC:
        case CElementBSDF::THINDIELECTRIC:
        case CElementBSDF::ROUGHDIELECTRIC:
        {
            nextSym = ir->allocNode<IR::CBSDFCombinerNode>(IR::CBSDFCombinerNode::ET_FRESNEL_BLEND);
            nextSym->children.count = 2u;

            const float eta = current->dielectric.intIOR/current->dielectric.extIOR;

            auto& refl = nextSym->children[0];
            refl = ir->allocNode<IR::CMicrofacetSpecularBSDFNode>();
            auto& trans = nextSym->children[1];
            trans = ir->allocNode<IR::CMicrofacetSpecularBSDFNode>();

            auto* node_refl = static_cast<IR::CMicrofacetSpecularBSDFNode*>(refl);
            auto* node_trans = static_cast<IR::CMicrofacetSpecularBSDFNode*>(trans);

            node_refl->scatteringMode = IR::CMicrofacetSpecularBSDFNode::ESM_REFLECT;
            node_refl->shadowing = IR::CMicrofacetSpecularBSDFNode::EST_SMITH;
            node_refl->eta = IR::INode::color_t(eta);
            if (currType == CElementBSDF::ROUGHDIELECTRIC)
            {
                node_refl->ndf = ndfMap[current->dielectric.distribution];
                getFloatOrTexture(current->dielectric.alphaU, node_refl->alpha_u);
                if (node_refl->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                    getFloatOrTexture(current->dielectric.alphaV, node_refl->alpha_v);
                else
                    node_refl->alpha_v = node_refl->alpha_u;
            }
            else
            {
                node_refl->setSmooth();
            }
            node_trans->scatteringMode = IR::CMicrofacetSpecularBSDFNode::ESM_TRANSMIT;
            node_trans->ndf = node_refl->ndf;
            node_trans->shadowing = node_refl->shadowing;
            node_trans->alpha_u = node_refl->alpha_u;
            node_trans->alpha_v = node_refl->alpha_v;
            node_trans->eta = node_refl->eta;

            thin = (currType == CElementBSDF::THINDIELECTRIC);
        }
        break;
        case CElementBSDF::BUMPMAP:
        {
            nextSym = ir->allocNode<IR::CGeomModifierNode>(IR::CGeomModifierNode::ET_DERIVATIVE);
            nextSym->children.count = 1u;

            auto* node = static_cast<IR::CGeomModifierNode*>(nextSym);
            node->source = IR::CGeomModifierNode::ESRC_TEXTURE;
            auto bm = getTexture(current->bumpmap.texture);

            auto& img = std::get<0>(bm)->getCreationParameters().image;
            auto smplr = std::get<1>(bm);

            std::tie(node->texture.image, node->texture.sampler, node->texture.scale) = std::make_tuple(createDerivMap(img.get(), smplr.get()), std::move(smplr), std::get<2>(bm));
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
            node->thickness = current->coating.thickness;
            getSpectrumOrTexture(current->coating.sigmaA, node->sigmaA);
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