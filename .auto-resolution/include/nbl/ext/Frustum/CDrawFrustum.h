// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_FRUSTUM_DRAW_FRUSTUM_H_
#define _NBL_EXT_FRUSTUM_DRAW_FRUSTUM_H_

#include "nbl/video/declarations.h"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/ext/Frustum/builtin/hlsl/common.hlsl"

namespace nbl::ext::frustum
{
    class CDrawFrustum final : public core::IReferenceCounted
    {
    public:
        static constexpr inline uint32_t IndicesCount = 24u;

        enum DrawMode : uint16_t
        {
            DM_SINGLE = 0b01,
            DM_BATCH = 0b10,
            DM_BOTH = 0b11
        };

        struct SCachedCreationParameters
        {
            using streaming_buffer_t = video::StreamingTransientDataBufferST<core::allocator<uint8_t>>;
            static constexpr inline auto RequiredAllocateFlags = core::bitflag<video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS>(video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
            static constexpr inline auto RequiredUsageFlags = core::bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT) | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
            DrawMode drawMode = DM_BOTH;
            core::smart_refctd_ptr<video::IUtilities> utilities;
            core::smart_refctd_ptr<streaming_buffer_t> streamingBuffer = nullptr;
        };

        struct SCreationParameters : SCachedCreationParameters
        {
            video::IQueue* transfer = nullptr;
            core::smart_refctd_ptr<asset::IAssetManager> assetManager = nullptr;

            core::smart_refctd_ptr<video::IGPUPipelineLayout> singlePipelineLayout = nullptr;
            core::smart_refctd_ptr<video::IGPUPipelineLayout> batchPipelineLayout = nullptr;
            core::smart_refctd_ptr<video::IGPURenderpass> renderpass = nullptr;

            inline bool validate() const
            {
                const auto validation = std::to_array
                ({
                    std::make_pair(bool(assetManager), "Invalid `creationParams.assetManager` is nullptr!"),
                    std::make_pair(bool(utilities), "Invalid `creationParams.utilities` is nullptr!"),
                    std::make_pair(bool(transfer), "Invalid `creationParams.transfer` is nullptr!"),
                    std::make_pair(bool(renderpass), "Invalid `creationParams.renderpass` is nullptr!"),
                    std::make_pair(bool(utilities->getLogicalDevice()->getPhysicalDevice()->getQueueFamilyProperties()[transfer->getFamilyIndex()].queueFlags.hasFlags(video::IQueue::FAMILY_FLAGS::TRANSFER_BIT)), "Invalid `creationParams.transfer` is not capable of transfer operations!")
                    });
                system::logger_opt_ptr logger = utilities->getLogger();
                for (const auto& [ok, error] : validation)
                    if (!ok)
                    {
                        logger.log(error, system::ILogger::ELL_ERROR);
                        return false;
                    }

                assert(bool(assetManager->getSystem()));

                return true;
            }
        };

        struct DrawParameters
        {
            video::IGPUCommandBuffer* commandBuffer = nullptr;
            hlsl::float32_t4x4 viewProjectionMatrix;
            float lineWidth = 2.f;
        };

        static core::smart_refctd_ptr<CDrawFrustum> create(SCreationParameters&& params);

        static core::smart_refctd_ptr<video::IGPUPipelineLayout> createPipelineLayoutFromPCRange(video::ILogicalDevice* device, const asset::SPushConstantRange& pcRange);

        static core::smart_refctd_ptr<video::IGPUPipelineLayout> createDefaultPipelineLayout(video::ILogicalDevice* device, DrawMode mode = DM_BATCH);

        static const core::smart_refctd_ptr<system::IFileArchive> mount(core::smart_refctd_ptr<system::ILogger> logger, system::ISystem* system, video::ILogicalDevice* device, const std::string_view archiveAlias = "");

        inline const SCachedCreationParameters& getCreationParameters() const { return m_cachedCreationParams; }

        bool renderSingle(const DrawParameters& params, const hlsl::float32_t4x4& frustumTransform,const hlsl::float32_t4 & color);
        bool render(const DrawParameters& params, video::ISemaphore::SWaitInfo waitInfo, std::span<const InstanceData> frustumInstances);
    protected:

        struct ConstructorParams
        {
            SCachedCreationParameters creationParams;
            core::smart_refctd_ptr<video::IGPUGraphicsPipeline> singlePipeline = nullptr;
            core::smart_refctd_ptr<video::IGPUGraphicsPipeline> batchPipeline = nullptr;
            core::smart_refctd_ptr<video::IGPUBuffer> indicesBuffer = nullptr;
        };

        CDrawFrustum(ConstructorParams&& params) : 
            m_cachedCreationParams(std::move(params.creationParams)),
            m_singlePipeline(std::move(params.singlePipeline)),
            m_batchPipeline(std::move(params.batchPipeline)),
            m_indicesBuffer(std::move(params.indicesBuffer))
        {}
        ~CDrawFrustum() override {};
    private:
        static core::smart_refctd_ptr<video::IGPUGraphicsPipeline> createPipeline(SCreationParameters& params, const video::IGPUPipelineLayout* pipelineLayout, const DrawMode mode);
        static bool createStreamingBuffer(SCreationParameters& params);
        static core::smart_refctd_ptr<video::IGPUBuffer> createIndicesBuffer(SCreationParameters& params);

        core::smart_refctd_ptr<video::IGPUBuffer> m_indicesBuffer;

        SCachedCreationParameters m_cachedCreationParams;

        core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_singlePipeline;
        core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_batchPipeline;
    };
}
#endif
