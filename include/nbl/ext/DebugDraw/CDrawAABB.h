// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_DRAW_AABB_H_
#define _NBL_EXT_DRAW_AABB_H_

#include "nbl/video/declarations.h"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/shapes/aabb.hlsl"
#include "nbl/ext/DebugDraw/builtin/hlsl/common.hlsl"

namespace nbl::ext::debug_draw
{
class DrawAABB final : public core::IReferenceCounted
{
    public:
        static constexpr inline uint32_t IndicesCount = 24u;
        static constexpr inline uint32_t VerticesCount = 8u;

        struct SCachedCreationParameters
        {
            using streaming_buffer_t = video::StreamingTransientDataBufferST<core::allocator<uint8_t>>;

            static constexpr inline auto RequiredAllocateFlags = core::bitflag<video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS>(video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
            static constexpr inline auto RequiredUsageFlags = core::bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT) | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;

            core::smart_refctd_ptr<video::IUtilities> utilities;

            //! optional, default MDI buffer allocated if not provided
            core::smart_refctd_ptr<streaming_buffer_t> streamingBuffer = nullptr;
        };
        
        struct SCreationParameters : SCachedCreationParameters
        {
            video::IQueue* transfer = nullptr;
            core::smart_refctd_ptr<asset::IAssetManager> assetManager = nullptr;

            core::smart_refctd_ptr<video::IGPUPipelineLayout> singlePipelineLayout;
            core::smart_refctd_ptr<video::IGPUPipelineLayout> batchPipelineLayout;
            core::smart_refctd_ptr<video::IGPURenderpass> renderpass = nullptr;
        };

        // creates an instance that can draw one AABB via push constant or multiple using streaming buffer
        static core::smart_refctd_ptr<DrawAABB> create(SCreationParameters&& params);

        // creates pipeline layout from push constant range
        static core::smart_refctd_ptr<video::IGPUPipelineLayout> createPipelineLayoutFromPCRange(video::ILogicalDevice* device, const asset::SPushConstantRange& pcRange);

        // creates default pipeline layout for streaming version
        static core::smart_refctd_ptr<video::IGPUPipelineLayout> createDefaultPipelineLayout(video::ILogicalDevice* device);

        //! mounts the extension's archive to given system - useful if you want to create your own shaders with common header included
        static const core::smart_refctd_ptr<system::IFileArchive> mount(core::smart_refctd_ptr<system::ILogger> logger, system::ISystem* system, const std::string_view archiveAlias = "");

        inline const SCachedCreationParameters& getCreationParameters() const { return m_cachedCreationParams; }

        // records draw command for single AABB, user has to set pipeline outside
        bool renderSingle(video::IGPUCommandBuffer* commandBuffer, const hlsl::shapes::AABB<3, float>& aabb, const hlsl::float32_t4& color, const hlsl::float32_t4x4& cameraMat);

        bool render(video::IGPUCommandBuffer* commandBuffer, video::ISemaphore::SWaitInfo waitInfo, const hlsl::float32_t4x4& cameraMat);

        static hlsl::float32_t4x4 getTransformFromAABB(const hlsl::shapes::AABB<3, float>& aabb);

        void addAABB(const hlsl::shapes::AABB<3,float>& aabb, const hlsl::float32_t4& color = { 1,0,0,1 });
        void addOBB(const hlsl::shapes::AABB<3, float>& aabb, const hlsl::float32_t4x4& transform, const hlsl::float32_t4& color = { 1,0,0,1 });
        void clearAABBs();

    protected:
	    DrawAABB(SCreationParameters&& _params, core::smart_refctd_ptr<video::IGPUGraphicsPipeline> singlePipeline, core::smart_refctd_ptr<video::IGPUGraphicsPipeline> batchPipeline,
            core::smart_refctd_ptr<video::IGPUBuffer> indicesBuffer, core::smart_refctd_ptr<video::IGPUBuffer> verticesBuffer);
	    ~DrawAABB() override;

    private:
        static core::smart_refctd_ptr<video::IGPUGraphicsPipeline> createPipeline(SCreationParameters& params, const video::IGPUPipelineLayout* pipelineLayout, const std::string& vsPath, const std::string& fsPath);
        static bool createStreamingBuffer(SCreationParameters& params);
        static core::smart_refctd_ptr<video::IGPUBuffer> createIndicesBuffer(SCreationParameters& params);
        static core::smart_refctd_ptr<video::IGPUBuffer> createVerticesBuffer(SCreationParameters& params);

        std::vector<debug_draw::InstanceData> m_instances;
        core::smart_refctd_ptr<video::IGPUBuffer> m_indicesBuffer;
        core::smart_refctd_ptr<video::IGPUBuffer> m_verticesBuffer;

        SCachedCreationParameters m_cachedCreationParams;

        core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_singlePipeline;
        core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_batchPipeline;
};
}

#endif
