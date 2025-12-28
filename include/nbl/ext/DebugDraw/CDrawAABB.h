// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_DEBUG_DRAW_DRAW_AABB_H_
#define _NBL_EXT_DEBUG_DRAW_DRAW_AABB_H_

#include "nbl/video/declarations.h"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/shapes/aabb.hlsl"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/ext/DebugDraw/builtin/hlsl/common.hlsl"

namespace nbl::ext::debug_draw
{
    class DrawAABB final : public core::IReferenceCounted
    {
    public:
        static constexpr inline uint32_t IndicesCount = 24u;

        enum DrawMode : uint16_t
        {
            ADM_DRAW_SINGLE = 0b01,
            ADM_DRAW_BATCH = 0b10,
            ADM_DRAW_BOTH = 0b11
        };

        struct SCachedCreationParameters
        {
            using streaming_buffer_t = video::StreamingTransientDataBufferST<core::allocator<uint8_t>>;

            static constexpr inline auto RequiredAllocateFlags = core::bitflag<video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS>(video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
            static constexpr inline auto RequiredUsageFlags = core::bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT) | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;

            DrawMode drawMode = ADM_DRAW_BOTH;

            core::smart_refctd_ptr<video::IUtilities> utilities;

            //! optional, default MDI buffer allocated if not provided
            core::smart_refctd_ptr<streaming_buffer_t> streamingBuffer = nullptr;
        };

        struct SCreationParameters : SCachedCreationParameters
        {
            video::IQueue* transfer = nullptr;  // only used to make the 24 element index buffer and instanced pipeline on create
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
            hlsl::float32_t4x4 cameraMat;
            float lineWidth = 1.f;
        };

        // creates an instance that can draw one AABB via push constant or multiple using streaming buffer
        static core::smart_refctd_ptr<DrawAABB> create(SCreationParameters&& params);

        // creates pipeline layout from push constant range
        static core::smart_refctd_ptr<video::IGPUPipelineLayout> createPipelineLayoutFromPCRange(video::ILogicalDevice* device, const asset::SPushConstantRange& pcRange);

        // creates default pipeline layout for pipeline specified by draw mode (note: if mode==BOTH, returns layout for BATCH mode)
        static core::smart_refctd_ptr<video::IGPUPipelineLayout> createDefaultPipelineLayout(video::ILogicalDevice* device, DrawMode mode = ADM_DRAW_BATCH);

        //! mounts the extension's archive to given system - useful if you want to create your own shaders with common header included
        static const core::smart_refctd_ptr<system::IFileArchive> mount(core::smart_refctd_ptr<system::ILogger> logger, system::ISystem* system, video::ILogicalDevice* device, const std::string_view archiveAlias = "");

        inline const SCachedCreationParameters& getCreationParameters() const { return m_cachedCreationParams; }

        // records draw command for single AABB, user has to set pipeline outside
        bool renderSingle(const DrawParameters& params, const hlsl::shapes::AABB<3, float>& aabb, const hlsl::float32_t4& color);

        // records draw command for rendering batch of AABB instances as InstanceData
        // user has to set span of filled-in InstanceData; camera matrix used in push constant
        inline bool render(const DrawParameters& params, video::ISemaphore::SWaitInfo waitInfo, std::span<const InstanceData> aabbInstances)
        {
            system::logger_opt_ptr logger = m_cachedCreationParams.utilities->getLogger();
            if (!(m_cachedCreationParams.drawMode & ADM_DRAW_BATCH))
            {
                logger.log("DrawAABB has not been enabled for draw batches!", system::ILogger::ELL_ERROR);
                return false;
            }

            using offset_t = SCachedCreationParameters::streaming_buffer_t::size_type;
            constexpr offset_t MaxAlignment = sizeof(InstanceData);
            // allocator initialization needs us to round up to PoT
            const auto MaxPOTAlignment = hlsl::roundUpToPoT(MaxAlignment);
            auto* streaming = m_cachedCreationParams.streamingBuffer.get();
            if (streaming->getAddressAllocator().max_alignment() < MaxPOTAlignment)
            {
                logger.log("Draw AABB Streaming Buffer cannot guarantee the alignments we require!");
                return false;
            }

            auto* const streamingPtr = reinterpret_cast<uint8_t*>(streaming->getBufferPointer());
            assert(streamingPtr);

            auto& commandBuffer = params.commandBuffer;
            commandBuffer->bindGraphicsPipeline(m_batchPipeline.get());
            commandBuffer->setLineWidth(params.lineWidth);
            asset::SBufferBinding<video::IGPUBuffer> indexBinding = { .offset = 0, .buffer = m_indicesBuffer };
            commandBuffer->bindIndexBuffer(indexBinding, asset::EIT_32BIT);

            auto srcIt = aabbInstances.begin();
            auto setInstancesRange = [&](InstanceData* data, uint32_t count) -> void {
                for (uint32_t i = 0; i < count; i++)
                {
                    auto inst = data + i;
                    *inst = *srcIt;
                    inst->transform = hlsl::mul(params.cameraMat, inst->transform);
                    srcIt++;

                    if (srcIt == aabbInstances.end())
                        break;
                }
            };

            const uint32_t numInstances = aabbInstances.size();
            uint32_t remainingInstancesBytes = numInstances * sizeof(InstanceData);
            while (srcIt != aabbInstances.end())
            {
                uint32_t blockByteSize = core::alignUp(remainingInstancesBytes, MaxAlignment);
                bool allocated = false;

                offset_t blockOffset = SCachedCreationParameters::streaming_buffer_t::invalid_value;
                const uint32_t smallestAlloc = hlsl::max<uint32_t>(core::alignUp(sizeof(InstanceData), MaxAlignment), streaming->getAddressAllocator().min_size());
                while (blockByteSize >= smallestAlloc)
                {
                    std::chrono::steady_clock::time_point waitTill = std::chrono::steady_clock::now() + std::chrono::milliseconds(1u);
                    if (streaming->multi_allocate(waitTill, 1, &blockOffset, &blockByteSize, &MaxAlignment) == 0u)
                    {
                        allocated = true;
                        break;
                    }

                    streaming->cull_frees();
                    blockByteSize >>= 1;
                }

                if (!allocated)
                {
                    logger.log("Failed to allocate a chunk from streaming buffer for the next drawcall batch.", system::ILogger::ELL_ERROR);
                    return false;
                }

                const uint32_t instanceCount = blockByteSize / sizeof(InstanceData);                
                auto* const streamingInstancesPtr = reinterpret_cast<InstanceData*>(streamingPtr + blockOffset);
                setInstancesRange(streamingInstancesPtr, instanceCount);

                if (streaming->needsManualFlushOrInvalidate())
                {
                    const video::ILogicalDevice::MappedMemoryRange flushRange(streaming->getBuffer()->getBoundMemory().memory, blockOffset, blockByteSize);
                    m_cachedCreationParams.utilities->getLogicalDevice()->flushMappedMemoryRanges(1, &flushRange);
                }

                remainingInstancesBytes -= instanceCount * sizeof(InstanceData);

                SInstancedPC pc;
                pc.pInstanceBuffer = m_cachedCreationParams.streamingBuffer->getBuffer()->getDeviceAddress() + blockOffset;

                commandBuffer->pushConstants(m_batchPipeline->getLayout(), asset::IShader::E_SHADER_STAGE::ESS_VERTEX, offsetof(ext::debug_draw::PushConstants, ipc), sizeof(SInstancedPC), &pc);
                commandBuffer->drawIndexed(IndicesCount, instanceCount, 0, 0, 0);

                streaming->multi_deallocate(1, &blockOffset, &blockByteSize, waitInfo);
            }

            return true;
        }

        static inline hlsl::float32_t3x4 getTransformFromAABB(const hlsl::shapes::AABB<3, float>& aabb)
        {
            const auto diagonal = aabb.getExtent();
            hlsl::float32_t3x4 transform;
            transform[0][3] = aabb.minVx.x;
            transform[1][3] = aabb.minVx.y;
            transform[2][3] = aabb.minVx.z;
            transform[0][0] = diagonal.x;
            transform[1][1] = diagonal.y;
            transform[2][2] = diagonal.z;
            return transform;
        }

        static hlsl::float32_t4x4 getTransformFromOBB(const hlsl::shapes::OBB<3, float>& aabb);

    protected:
        struct ConstructorParams
        {
            SCachedCreationParameters creationParams;
            core::smart_refctd_ptr<video::IGPUGraphicsPipeline> singlePipeline = nullptr;
            core::smart_refctd_ptr<video::IGPUGraphicsPipeline> batchPipeline = nullptr;
            core::smart_refctd_ptr<video::IGPUBuffer> indicesBuffer = nullptr;
        };

	    DrawAABB(ConstructorParams&& params) :
            m_cachedCreationParams(std::move(params.creationParams)),
            m_singlePipeline(std::move(params.singlePipeline)),
            m_batchPipeline(std::move(params.batchPipeline)),
            m_indicesBuffer(std::move(params.indicesBuffer))
        {}
	    ~DrawAABB() override {}

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
