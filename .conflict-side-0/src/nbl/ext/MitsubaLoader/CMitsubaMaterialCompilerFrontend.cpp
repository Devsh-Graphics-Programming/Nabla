// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/CMitsubaMaterialCompilerFrontend.h"

#if 0 // bump map handling in CTrueIR
            auto view = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(viewBundle.getContents().begin()[0]);

            auto found = m_loaderContext->derivMapCache.find(view->getCreationParameters().image);
            if (found!=m_loaderContext->derivMapCache.end())
            {
                const float normalizationFactor = found->second;
                scale *= normalizationFactor;
            }

            std::get<core::smart_refctd_ptr<asset::ICPUImageView>>(retval) = std::move(view);
#endif

        

#if 0 /// TODO when going CFrontendIR -> CTrueIR
                asset::material_compiler::IR::CEmitterNode::EmissionProfile profile;
            {
                auto worldSpaceIESTransform = core::concatenateBFollowedByA(transform, _emitter->transform.matrix);

                // fill .w component of 1 & 2 row with 0 to perform normmalization on .xyz vectors for these rows bellow to get up & view vectors
                worldSpaceIESTransform[1].w = 0;
                worldSpaceIESTransform[2].w = 0;

                const float THRESHOLD = core::exp2(-14.f);
                const auto det = core::determinant(worldSpaceIESTransform);

                if (abs(det) < THRESHOLD) // protect us from determinant = 0 where inverse transform doesn't exist because the matrix is singular, also we don't want to be too much close to 0 because of the matrix conditioning index becoming higher and higher
                {
                    os::Printer::log("ERROR: Emission profile rejected because determinant of transformation matrix does not exceed the minimum threshold and is too close or equal 0", ELL_ERROR);
                    return false; // exit the lambda
                }

                profile.right_hand = det > 0.0f;
                profile.up = core::normalize(worldSpaceIESTransform[1]);
                profile.view = core::normalize(worldSpaceIESTransform[2]);
            }
    


CMitsubaMaterialCompilerFrontend::tex_ass_type CMitsubaMaterialCompilerFrontend::getErrorTexture(const E_IMAGE_VIEW_SEMANTIC semantic) const
{
    tex_ass_type retval = { nullptr,nullptr,1.f };

    {
        const asset::IAsset::E_TYPE types[2]{ asset::IAsset::ET_SAMPLER, asset::IAsset::ET_TERMINATING_ZERO };
        constexpr const char* ERR_SMPLR_CACHE_NAME = "nbl/builtin/sampler/default";
        auto sbundle = m_loaderContext->override_->findCachedAsset(ERR_SMPLR_CACHE_NAME, types, m_loaderContext->inner, 0u);
        assert(!sbundle.getContents().empty()); // this shouldnt ever happen since ERR_SMPLR_CACHE_NAME is builtin asset
        std::get<core::smart_refctd_ptr<asset::ICPUSampler>>(retval) = core::smart_refctd_ptr_static_cast<asset::ICPUSampler>(sbundle.getContents().begin()[0]);
    }
    
    {
        auto format = asset::EF_R8G8B8A8_SRGB;
        uint32_t fill_value = 0x80808080u; // mid-gray
        std::string ERR_TEX_CACHE_NAME = "nbl/builtin/image_view/not_found";
        switch (semantic)
        {
            case CMitsubaMaterialCompilerFrontend::EIVS_BLEND_WEIGHT:
                ERR_TEX_CACHE_NAME += "?blend";
                break;
            case CMitsubaMaterialCompilerFrontend::EIVS_NORMAL_MAP:
                ERR_TEX_CACHE_NAME += "?deriv?n";
                format = asset::EF_R8G8_SNORM;
                std::get<float>(retval) = 0.f;
                break;
            case CMitsubaMaterialCompilerFrontend::EIVS_BUMP_MAP:
                ERR_TEX_CACHE_NAME += "?deriv?h";
                format = asset::EF_R8G8_SNORM;
                std::get<float>(retval) = 0.f;
                break;
            default:
                fill_value = 0xffff00ffu; // magenta
                break;
        }
        ERR_TEX_CACHE_NAME += "?view";

        const asset::IAsset::E_TYPE types[2]{ asset::IAsset::ET_IMAGE_VIEW, asset::IAsset::ET_TERMINATING_ZERO };
        auto bundle = m_loaderContext->override_->findCachedAsset(ERR_TEX_CACHE_NAME, types, m_loaderContext->inner, 0u);

        auto& outImageView = std::get<core::smart_refctd_ptr<asset::ICPUImageView>>(retval);
        if (bundle.getContents().empty())
        {
            constexpr uint32_t dummyTexPOTSize = 6;
            constexpr uint32_t resolution = 0x1u<<dummyTexPOTSize;
            
            auto image = asset::ICPUImage::create(asset::ICPUImage::SCreationParams{
                /*.flags = */static_cast<asset::IImage::E_CREATE_FLAGS>(0u),
                /*.type = */asset::IImage::ET_2D,
                /*.format = */format,
                /*.extent = */{resolution,resolution,1u},
                /*.mipLevels = */1u,
                /*.arrayLayers = */1u,
                /*.samples = */asset::IImage::ESCF_1_BIT
                //.tiling etc.
            });

            // set contents
            {
                const auto TexelSize = asset::getTexelOrBlockBytesize(format);
                const auto& info = image->getCreationParameters();
                auto buf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(TexelSize*info.extent.width*info.extent.height);

                // fill
                assert(asset::getTexelOrBlockBytesize(info.format)==TexelSize);
                std::fill_n(reinterpret_cast<uint32_t*>(buf->getPointer()), buf->getSize()/sizeof(uint32_t), fill_value);

                auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1u);
                asset::ICPUImage::SBufferCopy& region = regions->front();
                region.imageSubresource.mipLevel = 0u;
                region.imageSubresource.baseArrayLayer = 0u;
                region.imageSubresource.layerCount = 1u;
                region.bufferOffset = 0u;
                region.bufferRowLength = info.extent.width;
                region.bufferImageHeight = 0u;
                region.imageOffset = { 0u, 0u, 0u };
                region.imageExtent = { resolution, resolution, 1u };
                image->setBufferAndRegions(std::move(buf), regions);
            }

            outImageView = core::make_smart_refctd_ptr<asset::ICPUImageView>(asset::ICPUImageView::SCreationParams{
                /*.flags = */{},
                /*.image = */std::move(image),
                /*.viewType = */asset::ICPUImageView::ET_2D,
                /*.format = */format,
                /*.components = */{},
                /*.subresourceRange = */{
                    /*.aspectMask = */asset::IImage::EAF_COLOR_BIT,
                    /*.baseMipLevel = */0u,
                    /*.levelCount = */1u,
                    /*.baseArrayLayer = */0u,
                    /*.layerCount = */1u
                }
            });

            // TODO: shouldn't be using an override, should be using asset manager directly and setting the mutability to immutable
            m_loaderContext->override_->insertAssetIntoCache(asset::SAssetBundle(nullptr,{outImageView}), ERR_TEX_CACHE_NAME, m_loaderContext->inner, 0);
        }
        else
            outImageView = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(bundle.getContents().begin()[0]);
    }

    return retval;
}
#endif



        case CElementBSDF::BUMPMAP:
        {
            ir_node = ir->allocNode<IR::CGeomModifierNode>(IR::CGeomModifierNode::ET_DERIVATIVE);
            ir_node->children.count = 1u;

            auto* node = static_cast<IR::CGeomModifierNode*>(ir_node);
            //no other source supported for now (uncomment in the future) [far future TODO]
            //node->source = IR::CGeomModifierNode::ESRC_TEXTURE;

            std::tie(node->texture.image,node->texture.sampler,node->texture.scale) =
                getTexture(_bsdf->bumpmap.texture,_bsdf->bumpmap.wasNormal ? EIVS_NORMAL_MAP:EIVS_BUMP_MAP);
        }
        break;