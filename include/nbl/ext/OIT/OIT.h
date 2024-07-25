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
        static inline constexpr auto SpinlockImageFormat   = asset::EF_R32_UINT;

        static inline constexpr uint32_t DefaultSetNum          = NBL_GLSL_OIT_SET_NUM;
        static inline constexpr uint32_t DefaultColorImgBinding = NBL_GLSL_COLOR_IMAGE_BINDING;
        static inline constexpr uint32_t DefaultDepthImgBinding = NBL_GLSL_DEPTH_IMAGE_BINDING;
        static inline constexpr uint32_t DefaultVisImgBinding   = NBL_GLSL_VIS_IMAGE_BINDING;
        static inline constexpr uint32_t DefaultSpinlockImgBinding   = NBL_GLSL_SPINLOCK_IMAGE_BINDING;
        static inline constexpr uint32_t MaxImgBindingCount   = 4u;

        struct images_t
        {
            core::smart_refctd_ptr<video::IGPUImageView> color;
            core::smart_refctd_ptr<video::IGPUImageView> depth;
            core::smart_refctd_ptr<video::IGPUImageView> vis;
            core::smart_refctd_ptr<video::IGPUImageView> spinlock;
        };

        struct proto_pipeline_t
        {
            core::smart_refctd_ptr<video::IGPUSpecializedShader> vs;
            core::smart_refctd_ptr<video::IGPUSpecializedShader> fs;
            asset::SVertexInputParams vtx;
            asset::SPrimitiveAssemblyParams primAsm;
            asset::SBlendParams blend;
            asset::SRasterizationParams raster;
            asset::SPushConstantRange pushConstants;
        };

        bool initialize(video::ILogicalDevice* dev, uint32_t w, uint32_t h,
            video::IGPUObjectFromAssetConverter::SParams& c2gparams,
            uint32_t set = DefaultSetNum, uint32_t colorBnd = DefaultColorImgBinding, uint32_t depthBnd = DefaultDepthImgBinding, uint32_t visBnd = DefaultVisImgBinding, uint32_t spinlockBnd = DefaultSpinlockImgBinding)
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
                    params.initialLayout = asset::IImage::EL_UNDEFINED;
                    params.mipLevels = 1u;
                    //indices
                    params.queueFamilyIndexCount = 0;
                    params.queueFamilyIndices = nullptr;
                    params.samples = asset::IImage::ESCF_1_BIT;
                    params.tiling = video::IGPUImage::TILING::OPTIMAL;
                    params.type = asset::IImage::ET_2D;
                    params.usage = asset::IImage::EUF_STORAGE_BIT;

                    img = dev->createImage(std::move(params));
                    assert(img);
                    auto mreq = img->getMemoryReqs();
                    mreq.memoryTypeBits &= dev->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
                    auto imgMem = dev->allocate(mreq, img.get());

                    if (!img || !imgMem.isValid())
                        return nullptr;

                    video::IGPUImageView::SCreationParams vparams = {};
                    vparams.format = params.format;
                    vparams.flags = static_cast<decltype(vparams.flags)>(0);
                    //vparams.subUsages = ? ? ? ; TODO
                    vparams.viewType = decltype(vparams.viewType)::ET_2D;
                    vparams.subresourceRange.baseArrayLayer = 0u;
                    vparams.subresourceRange.layerCount = 1u;
                    vparams.subresourceRange.baseMipLevel = 0u;
                    vparams.subresourceRange.levelCount = 1u;
                    vparams.image = img;
                    view = dev->createImageView(std::move(vparams));
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
            resolve_glsl += "#define NBL_GLSL_SPINLOCK_IMAGE_BINDING " + std::to_string(spinlockBnd) + "\n";
            resolve_glsl += "#include <nbl/builtin/glsl/ext/OIT/resolve.frag>\n";

            const bool hasInterlock = dev->getEnabledFeatures().fragmentShaderPixelInterlock;
            // TODO bring back
#if 0
            if (hasInterlock)
                m_images.spinlock = nullptr;
            else
#endif
                m_images.spinlock = createOITImage(SpinlockImageFormat);

            auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(resolve_glsl.c_str(), asset::IShader::ESS_FRAGMENT, asset::IShader::E_CONTENT_TYPE::ECT_GLSL, "oit_resolve.frag");
            auto shader = dev->createShader(std::move(cpushader));
            if (!shader)
                return false;

            std::tie(
                m_proto_pipeline.vs,
                m_proto_pipeline.vtx,
                m_proto_pipeline.primAsm,
                m_proto_pipeline.blend,
                m_proto_pipeline.raster,
                m_proto_pipeline.pushConstants
            ) = ext::FullScreenTriangle::createProtoPipeline(c2gparams, 0u);

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
            info.m_backingBuffer = nullptr;
            info.m_entries = nullptr;
            m_proto_pipeline.fs = dev->createSpecializedShader(shader.get(), info);
            if (!m_proto_pipeline.fs)
                return false;

            return true;
        }

        template <typename DSLType>
        uint32_t getDSLayoutBindings(typename DSLType::SBinding* _out_bindings,
            uint32_t _colorBinding = DefaultColorImgBinding, uint32_t _depthBinding = DefaultDepthImgBinding, uint32_t _visBinding = DefaultVisImgBinding, uint32_t _spinlockBinding = DefaultSpinlockImgBinding
        ) const
        {
            const uint32_t bindingCount = m_images.spinlock ? MaxImgBindingCount:(MaxImgBindingCount-1u);
            if (!_out_bindings)
                return bindingCount;

            const uint32_t b[MaxImgBindingCount]{ _colorBinding,_depthBinding,_visBinding,_spinlockBinding };
            for (uint32_t i = 0u; i < bindingCount; ++i)
            {
                auto& bnd = _out_bindings[i];
                bnd.binding = b[i];
                bnd.count = 1u;
                bnd.samplers = nullptr;
                bnd.stageFlags = asset::IShader::ESS_FRAGMENT;
                bnd.type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
            }

            return bindingCount;
        }
        uint32_t getDSWrites(video::IGPUDescriptorSet::SWriteDescriptorSet* _out_writes, video::IGPUDescriptorSet::SDescriptorInfo* _out_infos, video::IGPUDescriptorSet* dstset,
            uint32_t _colorBinding = DefaultColorImgBinding, uint32_t _depthBinding = DefaultDepthImgBinding, uint32_t _visBinding = DefaultVisImgBinding, uint32_t _spinlockBinding = DefaultSpinlockImgBinding
        ) const
        {
            const uint32_t bindingCount = m_images.spinlock ? MaxImgBindingCount:(MaxImgBindingCount-1u);
            if (!_out_writes || !_out_infos)
                return bindingCount;

            const uint32_t b[MaxImgBindingCount]{ _colorBinding, _depthBinding, _visBinding, _spinlockBinding };
            core::smart_refctd_ptr<video::IGPUImageView> images[MaxImgBindingCount]{ m_images.color, m_images.depth, m_images.vis, m_images.spinlock };
            for (uint32_t i = 0u; i < bindingCount; ++i)
            {
                auto& w = _out_writes[i];
                auto& info = _out_infos[i];

                w.arrayElement = 0u;
                w.binding = b[i];
                w.count = 1u;
                w.descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
                w.dstSet = dstset;
                w.info = &info;

                info.desc = images[i];
                info.image.sampler = nullptr;
                info.image.imageLayout = asset::IImage::EL_GENERAL; // TODO for Vulkan
            }

            return bindingCount;
        }

        core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDSLayout(video::ILogicalDevice* dev,
            uint32_t _colorBinding = DefaultColorImgBinding, uint32_t _depthBinding = DefaultDepthImgBinding, uint32_t _visBinding = DefaultVisImgBinding, uint32_t _spinlockBinding = DefaultSpinlockImgBinding
        ) const
        {
            video::IGPUDescriptorSetLayout::SBinding b[MaxImgBindingCount];
            const auto bindingCount = getDSLayoutBindings<video::IGPUDescriptorSetLayout>(b, _colorBinding, _depthBinding, _visBinding, _spinlockBinding);

            return dev->createDescriptorSetLayout(b,b+bindingCount);
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
            cmdbuf->clearColorImage(m_images.vis->getCreationParameters().image.get(), asset::IImage::EL_UNDEFINED, &clearval, 1u, &subres);

            if (m_images.spinlock)
            {
                clearval.uint32[0] = 0u;
                subres = m_images.spinlock->getCreationParameters().subresourceRange;
                cmdbuf->clearColorImage(m_images.spinlock->getCreationParameters().image.get(), asset::IImage::EL_UNDEFINED, &clearval, 1u, &subres);
            }
            // TODO barrier?
        }

        void barrierBetweenPasses(video::IGPUCommandBuffer* cmdbuf, uint32_t qfam) const
        {
            video::IGPUCommandBuffer::SImageMemoryBarrier imgbarrier[MaxImgBindingCount];
            for (uint32_t i = 0u; i < MaxImgBindingCount; ++i)
            {
                imgbarrier[i].barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
                imgbarrier[i].barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
                imgbarrier[i].dstQueueFamilyIndex = qfam;
                imgbarrier[i].srcQueueFamilyIndex = qfam;
                imgbarrier[i].oldLayout = asset::IImage::EL_GENERAL;
                imgbarrier[i].newLayout = asset::IImage::EL_GENERAL;
            }
            imgbarrier[0].image = m_images.color->getCreationParameters().image;
            imgbarrier[0].subresourceRange = m_images.color->getCreationParameters().subresourceRange;
            imgbarrier[1].image = m_images.depth->getCreationParameters().image;
            imgbarrier[1].subresourceRange = m_images.depth->getCreationParameters().subresourceRange;
            imgbarrier[2].image = m_images.vis->getCreationParameters().image;
            imgbarrier[2].subresourceRange = m_images.vis->getCreationParameters().subresourceRange;
            if (m_images.spinlock)
            {
                imgbarrier[3].image = m_images.spinlock->getCreationParameters().image;
                imgbarrier[3].subresourceRange = m_images.spinlock->getCreationParameters().subresourceRange;
            }

            const uint32_t bindingCount = m_images.spinlock ? MaxImgBindingCount:(MaxImgBindingCount-1u);
            cmdbuf->pipelineBarrier(asset::PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT, asset::PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT, asset::EDF_BY_REGION_BIT, 0u, nullptr, 0u, nullptr, bindingCount, imgbarrier);
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
            imgbarrier.oldLayout = asset::IImage::EL_GENERAL;
            imgbarrier.newLayout = asset::IImage::EL_GENERAL;
            imgbarrier.image = m_images.vis->getCreationParameters().image;
            imgbarrier.subresourceRange = m_images.vis->getCreationParameters().subresourceRange;

            cmdbuf->pipelineBarrier(asset::PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT, asset::PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &imgbarrier);
        }

        const proto_pipeline_t& getResolveProtoPipeline() const { return m_proto_pipeline; }

        const images_t& getImages() const { return m_images; }

    private:
        images_t m_images;
        proto_pipeline_t m_proto_pipeline;
    };
}

#endif// _NBL_EXT_OIT_H_INCLUDED_
