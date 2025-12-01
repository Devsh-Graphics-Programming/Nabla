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

        // only used to make the 24 element index buffer and instanced pipeline on create
        struct SCreationParameters : SCachedCreationParameters
        {
            video::IQueue* transfer = nullptr;
            core::smart_refctd_ptr<asset::IAssetManager> assetManager = nullptr;

            core::smart_refctd_ptr<video::IGPUPipelineLayout> singlePipelineLayout;
            core::smart_refctd_ptr<video::IGPUPipelineLayout> batchPipelineLayout;
            core::smart_refctd_ptr<video::IGPURenderpass> renderpass = nullptr;

            inline bool validate() const
            {
                assert(bool(assetManager));
                assert(bool(assetManager->getSystem()));
                assert(bool(utilities));
                assert(bool(transfer));
                assert(bool(renderpass));

                system::logger_opt_ptr logger = utilities->getLogger();
                if (!bool(utilities->getLogicalDevice()->getPhysicalDevice()->getQueueFamilyProperties()[transfer->getFamilyIndex()].queueFlags.hasFlags(video::IQueue::FAMILY_FLAGS::TRANSFER_BIT)))
                {
                    logger.log("Invalid `creationParams.transfer` is not capable of transfer operations!", system::ILogger::ELL_ERROR);
                    return false;
                }

                return true;
            }
        };

        struct DrawParameters
        {
            video::IGPUCommandBuffer* commandBuffer = nullptr;
            hlsl::float32_t4x4 cameraMat = hlsl::float32_t4x4(1);
            float lineWidth = 1.f;
        };

        // creates an instance that can draw one AABB via push constant or multiple using streaming buffer
        static core::smart_refctd_ptr<DrawAABB> create(SCreationParameters&& params);

        // creates pipeline layout from push constant range
        static core::smart_refctd_ptr<video::IGPUPipelineLayout> createPipelineLayoutFromPCRange(video::ILogicalDevice* device, const asset::SPushConstantRange& pcRange);

        // creates default pipeline layout for pipeline specified by draw mode (note: if mode==BOTH, returns layout for BATCH mode)
        static core::smart_refctd_ptr<video::IGPUPipelineLayout> createDefaultPipelineLayout(video::ILogicalDevice* device, DrawMode mode = ADM_DRAW_BATCH);

        //! mounts the extension's archive to given system - useful if you want to create your own shaders with common header included
        static const core::smart_refctd_ptr<system::IFileArchive> mount(core::smart_refctd_ptr<system::ILogger> logger, system::ISystem* system, const std::string_view archiveAlias = "");

        inline const SCachedCreationParameters& getCreationParameters() const { return m_cachedCreationParams; }

        // records draw command for single AABB, user has to set pipeline outside
        bool renderSingle(const DrawParameters& params, const hlsl::shapes::AABB<3, float>& aabb, const hlsl::float32_t4& color);

        // records draw command for rendering batch of AABB instances as InstanceData
        // user has to set span of filled-in InstanceData; camera matrix used in push constant
        inline bool render(const DrawParameters& params, video::ISemaphore::SWaitInfo waitInfo, std::span<const InstanceData> aabbInstances)
        {
            if (!(m_cachedCreationParams.drawMode & ADM_DRAW_BATCH))
            {
                m_cachedCreationParams.utilities->getLogger()->log("DrawAABB has not been enabled for draw batches!", system::ILogger::ELL_ERROR);
                return false;
            }

            using offset_t = SCachedCreationParameters::streaming_buffer_t::size_type;
            constexpr auto MdiSizes = std::to_array<offset_t>({ sizeof(hlsl::float32_t3), sizeof(InstanceData) });
            // shared nPoT alignment needs to be divisible by all smaller ones to satisfy an allocation from all
            constexpr offset_t MaxAlignment = std::reduce(MdiSizes.begin(), MdiSizes.end(), 1, [](const offset_t a, const offset_t b)->offset_t {return std::lcm(a, b); });
            // allocator initialization needs us to round up to PoT
            const auto MaxPOTAlignment = hlsl::roundUpToPoT(MaxAlignment);

            auto* streaming = m_cachedCreationParams.streamingBuffer.get();

            auto* const streamingPtr = reinterpret_cast<uint8_t*>(streaming->getBufferPointer());
            assert(streamingPtr);

            auto& commandBuffer = params.commandBuffer;
            commandBuffer->bindGraphicsPipeline(m_batchPipeline.get());
            commandBuffer->setLineWidth(params.lineWidth);
            asset::SBufferBinding<video::IGPUBuffer> indexBinding = { .offset = 0, .buffer = m_indicesBuffer };
            commandBuffer->bindIndexBuffer(indexBinding, asset::EIT_32BIT);

            auto setInstancesRange = [&](InstanceData* data, uint32_t count) -> void {
                for (uint32_t i = 0; i < count; i++)
                {
                    auto inst = data + i;
                    *inst = aabbInstances[i];
                    inst->transform = hlsl::mul(params.cameraMat, inst->transform);
                }
                };

            const uint32_t numInstances = aabbInstances.size();
            const uint32_t instancesPerIter = streaming->getBuffer()->getSize() / sizeof(InstanceData);
            using suballocator_t = core::LinearAddressAllocatorST<offset_t>;
            uint32_t beginOffset = 0;
            while (beginOffset < numInstances)
            {
                const uint32_t instanceCount = hlsl::min<uint32_t>(instancesPerIter, numInstances);
                offset_t inputOffset = 0u;
                offset_t ImaginarySizeUpperBound = 0x1 << 30;
                suballocator_t imaginaryChunk(nullptr, inputOffset, 0, hlsl::roundUpToPoT(MaxAlignment), ImaginarySizeUpperBound);
                uint32_t instancesByteOffset = imaginaryChunk.alloc_addr(sizeof(InstanceData) * instanceCount, sizeof(InstanceData));
                const uint32_t totalSize = imaginaryChunk.get_allocated_size();

                inputOffset = SCachedCreationParameters::streaming_buffer_t::invalid_value;
                std::chrono::steady_clock::time_point waitTill = std::chrono::steady_clock::now() + std::chrono::milliseconds(1u);
                streaming->multi_allocate(waitTill, 1, &inputOffset, &totalSize, &MaxAlignment);

                auto* const streamingInstancesPtr = reinterpret_cast<InstanceData*>(streamingPtr + instancesByteOffset);
                setInstancesRange(streamingInstancesPtr, instanceCount);
                beginOffset += instanceCount;

                assert(!streaming->needsManualFlushOrInvalidate());

                SPushConstants pc;
                pc.pInstanceBuffer = m_cachedCreationParams.streamingBuffer->getBuffer()->getDeviceAddress() + instancesByteOffset;

                commandBuffer->pushConstants(m_batchPipeline->getLayout(), asset::IShader::E_SHADER_STAGE::ESS_VERTEX, 0, sizeof(SPushConstants), &pc);
                commandBuffer->drawIndexed(IndicesCount, instanceCount, 0, 0, 0);

                streaming->multi_deallocate(1, &inputOffset, &totalSize, waitInfo);
            }

            return true;
        }

        static hlsl::float32_t3x4 getTransformFromAABB(const hlsl::shapes::AABB<3, float>& aabb);

    protected:
	    DrawAABB(SCreationParameters&& _params, core::smart_refctd_ptr<video::IGPUGraphicsPipeline> singlePipeline, core::smart_refctd_ptr<video::IGPUGraphicsPipeline> batchPipeline,
            core::smart_refctd_ptr<video::IGPUBuffer> indicesBuffer);
	    ~DrawAABB() override;

    private:
        //static bool validateCreationParameters(SCreationParameters& params);
        static core::smart_refctd_ptr<video::IGPUGraphicsPipeline> createPipeline(SCreationParameters& params, const video::IGPUPipelineLayout* pipelineLayout, const std::string& vsPath, const std::string& fsPath);
        static bool createStreamingBuffer(SCreationParameters& params);
        static core::smart_refctd_ptr<video::IGPUBuffer> createIndicesBuffer(SCreationParameters& params);

        core::smart_refctd_ptr<video::IGPUBuffer> m_indicesBuffer;

        SCachedCreationParameters m_cachedCreationParams;

        core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_singlePipeline;
        core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_batchPipeline;
};
}

#endif
