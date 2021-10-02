#ifndef _NBL_EXT_OIT_H_INCLUDED_
#define _NBL_EXT_OIT_H_INCLUDED_

#include "nbl/builtin/glsl/ext/OIT/oit.glsl"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include <nabla.h>

namespace nbl::ext::OIT
{
    class COIT
    {
    public:
        static inline constexpr auto ColorImageFormat = NBL_GLSL_OIT_IMG_FORMAT_COLOR;
        static inline constexpr auto DepthImageFormat = NBL_GLSL_OIT_IMG_FORMAT_DEPTH;
        static inline constexpr auto VisImageFormat   = NBL_GLSL_OIT_IMG_FORMAT_VIS;

        static inline constexpr uint32_t BindingCount = 3u;
        static inline constexpr uint32_t DefaultSetNum          = 2u;
        static inline constexpr uint32_t DefaultColorImgBinding = 0u;
        static inline constexpr uint32_t DefaultDepthImgBinding = 1u;
        static inline constexpr uint32_t DefaultVisImgBinding   = 2u;

        struct images_t
        {
            core::smart_refctd_ptr<video::IGPUImageView> color;
            core::smart_refctd_ptr<video::IGPUImageView> depth;
            core::smart_refctd_ptr<video::IGPUImageView> vis;
        };

        struct proto_pipeline_t
        {
            core::smart_refctd_ptr<video::IGPUSpecializedShader> vs;
            core::smart_refctd_ptr<video::IGPUSpecializedShader> fs;
            asset::SVertexInputParams vtx;
            asset::SPrimitiveAssemblyParams primAsm;
            asset::SBlendParams blend;
            asset::SRasterizationParams raster;
        };

        bool initialize(video::ILogicalDevice* dev, uint32_t w, uint32_t h,
            video::IGPUObjectFromAssetConverter::SParams& c2gparams,
            uint32_t set = DefaultSetNum, uint32_t colorBnd = DefaultColorImgBinding, uint32_t depthBnd = DefaultDepthImgBinding, uint32_t visBnd = DefaultVisImgBinding)
        {
            auto createOITImage = [&dev,w,h](asset::E_FORMAT fmt) -> core::smart_refctd_ptr<video::IGPUImageView> {
                core::smart_refctd_ptr<video::IGPUImage> img;
                core::smart_refctd_ptr<video::IGPUImageView> view;
                {
                    video::IGPUImage::SCreationParams params;
                    params.arrayLayers = 1u;
                    params.extent = { w, h, 1u };
                    params.flags = static_cast<video::IGPUImage::E_CREATE_FLAGS>(0);
                    params.format = fmt;
                    params.initialLayout = asset::EIL_UNDEFINED;
                    params.mipLevels = 1u;
                    //indices
                    params.queueFamilyIndexCount = 0;
                    params.queueFamilyIndices = nullptr;
                    params.samples = asset::IImage::ESCF_1_BIT;
                    params.sharingMode = asset::ESM_CONCURRENT;
                    params.tiling = asset::IImage::ET_OPTIMAL;
                    params.type = asset::IImage::ET_2D;
                    params.usage = asset::IImage::EUF_STORAGE_BIT;

                    video::IDriverMemoryBacked::SDriverMemoryRequirements mreq; //ignored on GL

                    img = dev->createGPUImageOnDedMem(std::move(params), mreq);
                    assert(img);
                    if (!img)
                        return nullptr;

                    video::IGPUImageView::SCreationParams vparams;
                    vparams.format = params.format;
                    vparams.flags = static_cast<decltype(vparams.flags)>(0);
                    vparams.viewType = decltype(vparams.viewType)::ET_2D;
                    vparams.subresourceRange.baseArrayLayer = 0u;
                    vparams.subresourceRange.layerCount = 1u;
                    vparams.subresourceRange.baseMipLevel = 0u;
                    vparams.subresourceRange.levelCount = 1u;
                    vparams.image = img;
                    view = dev->createGPUImageView(std::move(vparams));
                    assert(view);
                }
                return view;
            };

            m_images.color = createOITImage(ColorImageFormat);
            if (!m_images.color)
                return false;
            m_images.depth = createOITImage(DepthImageFormat);
            if (!m_images.depth)
                return false;
            m_images.vis   = createOITImage(VisImageFormat);
            if (!m_images.vis)
                return false;

            std::string resolve_glsl = "#version 430 core\n\n";
            resolve_glsl += "#define NBL_GLSL_OIT_SET_NUM " + std::to_string(set) + "\n";
            resolve_glsl += "#define NBL_GLSL_COLOR_IMAGE_BINDING " + std::to_string(colorBnd) + "\n";
            resolve_glsl += "#define NBL_GLSL_DEPTH_IMAGE_BINDING " + std::to_string(depthBnd) + "\n";
            resolve_glsl += "#define NBL_GLSL_VIS_IMAGE_BINDING " + std::to_string(visBnd) + "\n";
            resolve_glsl += "#include <nbl/builtin/glsl/ext/OIT/resolve.frag>\n";

            auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(resolve_glsl.c_str());
            auto shader = dev->createGPUShader(std::move(cpushader));
            if (!shader)
                return false;

            std::tie(
                m_proto_pipeline.vs,
                m_proto_pipeline.vtx,
                m_proto_pipeline.primAsm,
                m_proto_pipeline.blend,
                m_proto_pipeline.raster
            ) = ext::FullScreenTriangle::createProtoPipeline(c2gparams);

            m_proto_pipeline.blend.blendParams[0].blendEnable = 1;
            m_proto_pipeline.blend.blendParams[0].srcColorFactor = asset::EBF_ONE;
            m_proto_pipeline.blend.blendParams[0].dstColorFactor = asset::EBF_SRC_ALPHA;
            m_proto_pipeline.blend.blendParams[0].srcAlphaFactor = asset::EBF_ONE;
            m_proto_pipeline.blend.blendParams[0].dstAlphaFactor = asset::EBF_ZERO;
            m_proto_pipeline.raster.depthWriteEnable = 0;
            m_proto_pipeline.raster.depthTestEnable = 0;

            if (!m_proto_pipeline.vs)
                return false;

            asset::ISpecializedShader::SInfo info;
            info.entryPoint = "main";
            info.m_filePathHint = "oit_resolve.frag";
            info.shaderStage = asset::ISpecializedShader::ESS_FRAGMENT;
            info.m_backingBuffer = nullptr;
            info.m_entries = nullptr;
            m_proto_pipeline.fs = dev->createGPUSpecializedShader(shader.get(), info);
            if (!m_proto_pipeline.fs)
                return false;

            return true;
        }

        template <typename DSLType>
        uint32_t getDSLayoutBindings(DSLType::SBinding* _out_bindings,
            uint32_t _colorBinding = DefaultColorImgBinding, uint32_t _depthBinding = DefaultDepthImgBinding, uint32_t _visBinding = DefaultVisImgBinding
        ) const
        {
            if (!_out_bindings)
                return BindingCount;

            const uint32_t b[BindingCount]{ _colorBinding, _depthBinding, _visBinding };
            for (uint32_t i = 0u; i < BindingCount; ++i)
            {
                auto& bnd = _out_bindings[i];
                bnd.binding = b[i];
                bnd.count = 1u;
                bnd.samplers = nullptr;
                bnd.stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
                bnd.type = asset::EDT_STORAGE_IMAGE;
            }

            return BindingCount;
        }
        uint32_t getDSWrites(video::IGPUDescriptorSet::SWriteDescriptorSet* _out_writes, video::IGPUDescriptorSet::SDescriptorInfo* _out_infos, video::IGPUDescriptorSet* dstset,
            uint32_t _colorBinding = DefaultColorImgBinding, uint32_t _depthBinding = DefaultDepthImgBinding, uint32_t _visBinding = DefaultVisImgBinding
        ) const
        {
            if (!_out_writes || !_out_infos)
                return BindingCount;

            const uint32_t b[BindingCount]{ _colorBinding, _depthBinding, _visBinding };
            core::smart_refctd_ptr<video::IGPUImageView> images[BindingCount]{ m_images.color, m_images.depth, m_images.vis };
            for (uint32_t i = 0u; i < BindingCount; ++i)
            {
                auto& w = _out_writes[i];
                auto& info = _out_infos[i];

                w.arrayElement = 0u;
                w.binding = b[i];
                w.count = 1u;
                w.descriptorType = asset::EDT_STORAGE_IMAGE;
                w.dstSet = dstset;
                w.info = &info;

                info.desc = images[i];
                info.image.sampler = nullptr;
                info.image.imageLayout = asset::EIL_GENERAL; // TODO for Vulkan
            }

            return BindingCount;
        }

        core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDSLayout(video::ILogicalDevice* dev, uint32_t _colorBinding = DefaultColorImgBinding, uint32_t _depthBinding = DefaultDepthImgBinding, uint32_t _visBinding = DefaultVisImgBinding) const
        {
            video::IGPUDescriptorSetLayout::SBinding b[BindingCount];
            getDSLayoutBindings<video::IGPUDescriptorSetLayout>(b, _colorBinding, _depthBinding, _visBinding);

            return dev->createGPUDescriptorSetLayout(b, b + BindingCount);
        }

        // should be required in first frame only
        void invalidateNodesVisibility(video::IGPUCommandBuffer* cmdbuf) const
        {
            asset::SClearColorValue clearval;
            clearval.float32[0] = 1.f;
            clearval.float32[1] = 1.f;
            clearval.float32[2] = 1.f;
            clearval.float32[3] = 1.f;
            asset::IImage::SSubresourceRange subres = m_images.vis->getCreationParameters().subresourceRange;
            cmdbuf->clearColorImage(m_images.vis->getCreationParameters().image.get(), asset::EIL_UNDEFINED, &clearval, 1u, &subres);

            // TODO barrier?
        }

        void barrierBetweenPasses(video::IGPUCommandBuffer* cmdbuf, uint32_t qfam) const
        {
            video::IGPUCommandBuffer::SImageMemoryBarrier imgbarrier[BindingCount];
            for (uint32_t i = 0u; i < BindingCount; ++i)
            {
                imgbarrier[i].barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
                imgbarrier[i].barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
                imgbarrier[i].dstQueueFamilyIndex = qfam;
                imgbarrier[i].srcQueueFamilyIndex = qfam;
                imgbarrier[i].oldLayout = asset::EIL_GENERAL;
                imgbarrier[i].newLayout = asset::EIL_GENERAL;
            }
            imgbarrier[0].image = m_images.color->getCreationParameters().image;
            imgbarrier[0].subresourceRange = m_images.color->getCreationParameters().subresourceRange;
            imgbarrier[1].image = m_images.depth->getCreationParameters().image;
            imgbarrier[1].subresourceRange = m_images.depth->getCreationParameters().subresourceRange;
            imgbarrier[2].image = m_images.vis->getCreationParameters().image;
            imgbarrier[2].subresourceRange = m_images.vis->getCreationParameters().subresourceRange;

            cmdbuf->pipelineBarrier(asset::EPSF_FRAGMENT_SHADER_BIT, asset::EPSF_FRAGMENT_SHADER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, BindingCount, imgbarrier);
        }

        void resolvePass(video::IGPUCommandBuffer* cmdbuf, video::IGPUGraphicsPipeline* gfx, video::IGPUDescriptorSet* ds, uint32_t set = DefaultSetNum)
        {
            cmdbuf->bindGraphicsPipeline(gfx);
            cmdbuf->bindDescriptorSets(asset::EPBP_GRAPHICS, gfx->getRenderpassIndependentPipeline()->getLayout(), set, 1u, &ds);
            cmdbuf->draw(3, 1, 0, 0);
        }

        void barrierAfterResolve(video::IGPUCommandBuffer* cmdbuf, uint32_t qfam) const
        {
            video::IGPUCommandBuffer::SImageMemoryBarrier imgbarrier;
            imgbarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
            imgbarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
            imgbarrier.dstQueueFamilyIndex = qfam;
            imgbarrier.srcQueueFamilyIndex = qfam;
            imgbarrier.oldLayout = asset::EIL_GENERAL;
            imgbarrier.newLayout = asset::EIL_GENERAL;
            imgbarrier.image = m_images.vis->getCreationParameters().image;
            imgbarrier.subresourceRange = m_images.vis->getCreationParameters().subresourceRange;

            cmdbuf->pipelineBarrier(asset::EPSF_FRAGMENT_SHADER_BIT, asset::EPSF_FRAGMENT_SHADER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &imgbarrier);
        }

        const proto_pipeline_t& getResolveProtoPipeline() const { return m_proto_pipeline; }

        const images_t& getImages() const { return m_images; }

    private:
        images_t m_images;
        proto_pipeline_t m_proto_pipeline;
    };
}

#endif// _NBL_EXT_OIT_H_INCLUDED_
