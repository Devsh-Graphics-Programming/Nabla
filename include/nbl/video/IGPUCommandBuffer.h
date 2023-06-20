#ifndef _NBL_VIDEO_I_GPU_COMMAND_BUFFER_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_COMMAND_BUFFER_H_INCLUDED_

#include "nbl/core/declarations.h"

#include "nbl/asset/asset.h"

#include "nbl/video/IGPUShader.h"
#include "nbl/video/IGPUCommandPool.h"
#include "nbl/video/IGPUQueue.h"


#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <type_traits>


namespace nbl::video
{

class NBL_API2 IGPUCommandBuffer : public core::IReferenceCounted, public IBackendObject
{
    public:
        enum class LEVEL : uint8_t
        {
            PRIMARY = 0u,
            SECONDARY
        };
        inline LEVEL getLevel() const { return m_level; }

        inline IGPUCommandPool* getPool() const { return m_cmdpool.get(); }
        inline uint32_t getQueueFamilyIndex() const { return m_cmdpool->getQueueFamilyIndex(); }

        /*
        CommandBuffer Lifecycle Tracking in Nabla:
            * We say a command buffer is "resettable" If it was allocated from a command pool which was created with `ECF_RESET_COMMAND_BUFFER_BIT` flag.
            - ES_INITIAL
                -> When a command buffer is allocated, it is in the ES_INITIAL state.
                -> If a command buffer is "resettable", Calling `reset()` on a command buffer will change it's state to ES_INITIAL If it's not PENDING
            - ES_RECORDING
                -> Calling `begin()` on a command buffer will change it's state to `ES_RECORDING` If It's not already RECORDING, and should be INITIAL for non-resettable command buffers.
            - ES_EXECUTABLE
                -> Calling `end()` on a command buffer will change it's state to `ES_EXECUTABLE` If it's RECORDING
                -> After submission for non-resettable command buffers.
            - ES_PENDING
                * ES_PENDING Is impossible to track correctly without a fence. So `ES_PENDING` actually means the engine is in the process of SUBMITTING and It will be changed to either `ES_EXECUTABLE` or `ES_INVALID` after SUBMISSION.
                * So the convention here is different than Vulkan's command buffer lifecycle and therefore contains false negatives (It is not PENDING but actually is PENDING and working on GPU)
            - ES_INVALID
                -> After submission for resettable command buffers.
        */
        enum class STATE : uint8_t
        {
            INITIAL,
            RECORDING,
            EXECUTABLE,
            PENDING,
            INVALID
        };
        inline STATE getState() const { return m_state; }
        inline bool isResettable() const
        {
            return m_cmdpool->getCreationFlags().hasFlags(IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
        }
        inline bool canReset() const
        {
            if(isResettable())
                return m_state!=STATE::PENDING;
            return false;
        }

        //! Begin, Reset, End
        enum class USAGE : uint8_t
        {
            NONE = 0x00,
            ONE_TIME_SUBMIT_BIT = 0x01,
            RENDER_PASS_CONTINUE_BIT = 0x02,
            SIMULTANEOUS_USE_BIT = 0x04
        };
        inline core::bitflag<USAGE> getRecordingFlags() const { return m_recordingFlags; }

        enum class QUERY_CONTROL_FLAGS : uint8_t
        {
            NONE = 0x00u,
            PRECISE_BIT = 0x01u
        };
        struct SInheritanceInfo
        {
            bool occlusionQueryEnable = false;
            core::bitflag<QUERY_CONTROL_FLAGS> queryFlags = QUERY_CONTROL_FLAGS::NONE;
            // the rest are only used if you begin with the `EU_RENDER_PASS_CONTINUE_BIT` flag
            core::smart_refctd_ptr<const IGPURenderpass> renderpass = nullptr;
            uint32_t subpass = IGPURenderpass::SCreationParams::SSubpassDependency::External;
            // optional metadata
            core::smart_refctd_ptr<const IGPUFramebuffer> framebuffer = nullptr;
        };
        bool begin(const core::bitflag<USAGE> flags, const SInheritanceInfo* inheritanceInfo = nullptr);
        inline SInheritanceInfo getCachedInheritanceInfo() const { return m_cachedInheritanceInfo; }

        enum class RESET_FLAGS : uint8_t
        {
            NONE = 0x00,
            RELEASE_RESOURCES_BIT = 0x01
        };
        bool reset(const core::bitflag<RESET_FLAGS> flags);
        bool end();

        // no multi-device yet
        //bool setDeviceMask(uint32_t deviceMask);

        //! Barriers and Sync
        struct SResourceMemoryBarrier
        {
            asset::SMemoryBarrier barrier = {};
            // If both indices are equal there will be no ownership transfer
            uint32_t srcQueueFamilyIndex = 0u;
            uint32_t dstQueueFamilyIndex = 0u;
        };
        struct SBufferMemoryBarrier
        {
            SResourceMemoryBarrier barrier = {};
            asset::SBufferRange<const IGPUBuffer> range = {0ull,IGPUDescriptorSet::SDescriptorInfo::SBufferInfo::WholeBuffer,nullptr};
        };
        struct SImageMemoryBarrier
        {
            SResourceMemoryBarrier barrier = {};
            core::smart_refctd_ptr<const IGPUImage> image = nullptr;
            IGPUImage::SSubresourceRange subresourceRange = {};
            // If both layouts match, no transition is performed, so this is our default
            IGPUImage::LAYOUT oldLayout = IGPUImage::LAYOUT::UNDEFINED;
            IGPUImage::LAYOUT newLayout = IGPUImage::LAYOUT::UNDEFINED;
        };
        struct SDependencyInfo
        {
            // no dependency flags because they must be 0 per the spec
            uint32_t memBarrierCount = 0;
            const asset::SMemoryBarrier* memBarriers = nullptr;
            uint32_t bufBarrierCount = 0;
            const SBufferMemoryBarrier* bufBarriers = nullptr;
            uint32_t imgBarrierCount = 0;
            const SImageMemoryBarrier* imgBarriers = nullptr;
        };
        using stage_flags_t = asset::PIPELINE_STAGE_FLAGS;
        bool setEvent(IGPUEvent* const _event, const SDependencyInfo& depInfo);
        bool resetEvent(IGPUEvent* _event, const core::bitflag<stage_flags_t> stageMask);
        bool waitEvents(const uint32_t eventCount, IGPUEvent* const* const pEvents, const SDependencyInfo* depInfos);

        bool pipelineBarrier(const core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags, const SDependencyInfo& depInfo);

        //! buffer transfers
        bool fillBuffer(IGPUBuffer* const dstBuffer, const size_t dstOffset, const size_t size, const uint32_t data);
        bool updateBuffer(IGPUBuffer* const dstBuffer, const size_t dstOffset, const size_t dataSize, const void* const pData);
        struct SBufferCopy
        {
            size_t srcOffset;
            size_t dstOffset;
            size_t size;
        };
        bool copyBuffer(const IGPUBuffer* srcBuffer, IGPUBuffer* dstBuffer, uint32_t regionCount, const SBufferCopy* const pRegions);

        //! image transfers
        union SClearColorValue
        {
            float float32[4];
            int32_t int32[4];
            uint32_t uint32[4];
        };
        struct SClearDepthStencilValue
        {
            float depth;
            uint32_t stencil;
        };
        union SClearValue
        {
            SClearColorValue color;
            SClearDepthStencilValue depthStencil;
        };
        bool clearColorImage(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearColorValue* const pColor, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges);
        bool clearDepthStencilImage(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearDepthStencilValue* const pDepthStencil, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges);
        bool copyBufferToImage(const IGPUBuffer* const srcBuffer, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions);
        bool copyImageToBuffer(const IGPUBuffer* const srcImage, const IGPUImage::LAYOUT srcImageLayout, const IGPUBuffer* const dstBuffer, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions);
        bool copyImage(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SImageCopy* const pRegions);

        //! acceleration structure transfers
        bool copyAccelerationStructure(const IGPUAccelerationStructure::CopyInfo& copyInfo);
        bool copyAccelerationStructureToMemory(const IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo);
        bool copyAccelerationStructureFromMemory(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo);

        //! acceleration structure builds
        bool buildAccelerationStructures(const core::SRange<const IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const video::IGPUAccelerationStructure::BuildRangeInfo* const* const ppBuildRangeInfos);
        bool buildAccelerationStructuresIndirect(const core::SRange<const IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const core::SRange<const IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses, const uint32_t* const pIndirectStrides, const uint32_t* const* const ppMaxPrimitiveCounts);

        //! state setup
        bool bindComputePipeline(const IGPUComputePipeline* const pipeline);
        bool bindGraphicsPipeline(const IGPUGraphicsPipeline* const pipeline);
        bool bindDescriptorSets(
            const asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const IGPUPipelineLayout* const layout,
            const uint32_t firstSet, const uint32_t descriptorSetCount, const IGPUDescriptorSet* const* const pDescriptorSets,
            const uint32_t dynamicOffsetCount=0u, const uint32_t* const dynamicOffsets=nullptr
        );
        bool pushConstants(const IGPUPipelineLayout* const layout, const core::bitflag<IGPUShader::E_SHADER_STAGE> stageFlags, const uint32_t offset, const uint32_t size, const void* const pValues);
        bool bindVertexBuffers(const uint32_t firstBinding, const uint32_t bindingCount, const IGPUBuffer* const* const pBuffers, const size_t* const pOffsets);
        bool bindIndexBuffer(const IGPUBuffer* const buffer, const size_t offset, const asset::E_INDEX_TYPE indexType);

        //! queries
        bool resetQueryPool(IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount);
        bool beginQuery(IQueryPool* const queryPool, const uint32_t query, const core::bitflag<QUERY_CONTROL_FLAGS> flags=QUERY_CONTROL_FLAGS::NONE);
        bool endQuery(IQueryPool* const queryPool, const uint32_t query);
        bool writeTimestamp(const asset::PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* const queryPool, const uint32_t query);
        bool writeAccelerationStructureProperties(const core::SRange<const IGPUAccelerationStructure>& pAccelerationStructures, const IQueryPool::E_QUERY_TYPE queryType, IQueryPool* const queryPool, const uint32_t firstQuery);
        bool copyQueryPoolResults(
            const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount,
            IGPUBuffer* const dstBuffer, const size_t dstOffset, const size_t stride, const core::bitflag<IQueryPool::E_QUERY_RESULTS_FLAGS> flags
        );

        //! dispatches
        bool dispatch(const uint32_t wgCountX, const uint32_t wgCountY=1, const uint32_t wgCountZ=1);
        bool dispatchIndirect(const IGPUBuffer* buffer, size_t offset);

# if 0
        //! Begin/End RenderPasses
        struct SRenderpassBeginInfo
        {
            core::smart_refctd_ptr<const IGPURenderpass> renderpass;
            core::smart_refctd_ptr<IGPUFramebuffer> framebuffer;
            VkRect2D renderArea;
            uint32_t clearValueCount;
            const SClearValue* clearValues;
        };
        bool beginRenderPass(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content);
        bool nextSubpass(const asset::E_SUBPASS_CONTENTS contents);
        bool endRenderPass();

        //! draws
        bool drawIndirect(const IGPUBuffer* buffer, size_t offset, uint32_t drawCount, uint32_t stride);
        bool drawIndexedIndirect(const IGPUBuffer* buffer, size_t offset, uint32_t drawCount, uint32_t stride);
        bool drawIndirectCount(const IGPUBuffer* buffer, size_t offset, const IGPUBuffer* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride);
        bool drawIndexedIndirectCount(const IGPUBuffer* buffer, size_t offset, const IGPUBuffer* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride);
        /* soon: [[deprecated]] */ bool drawMeshBuffer(const IGPUMeshBuffer* meshBuffer);
#endif

        struct SImageBlit
        {
            IGPUImage::SSubresourceLayers srcSubresource;
            asset::VkOffset3D srcOffsets[2];
            IGPUImage::SSubresourceLayers dstSubresource;
            asset::VkOffset3D dstOffsets[2];
        };
        bool blitImage(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const SImageBlit* const pRegions, const IGPUSampler::E_TEXTURE_FILTER filter);
        struct SImageResolve
        {
            IGPUImage::SSubresourceLayers srcSubresource;
            asset::VkOffset3D srcOffset;
            IGPUImage::SSubresourceLayers dstSubresource;
            asset::VkOffset3D dstOffset;
            asset::VkExtent3D extent;
        };
        bool resolveImage(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const SImageResolve* const pRegions);

        //! Secondary CommandBuffer execute
        bool executeCommands(const uint32_t count, IGPUCommandBuffer* const* const cmdbufs);

        // Vulkan: const VkCommandBuffer*
        virtual const void* getNativeHandle() const = 0;

        inline const core::unordered_map<const IGPUDescriptorSet*, uint64_t>& getBoundDescriptorSetsRecord() const { return m_boundDescriptorSetsRecord; }

    protected: 
        friend class IGPUQueue;

        IGPUCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const LEVEL lvl, core::smart_refctd_ptr<IGPUCommandPool>&& _cmdpool, system::logger_opt_smart_ptr&& logger);
        virtual ~IGPUCommandBuffer()
        {
            // Only release the resources if the parent pool has not been reset because if it has been then the resources will already be released.
            if (!checkForParentPoolReset())
            {
                releaseResourcesBackToPool();
            }
        }

        template<bool fill=false>
        inline bool validate_updateBuffer(const IGPUBuffer* const dstBuffer, const size_t dstOffset, const size_t dataSize)
        {
            if (!dstBuffer || !this->isCompatibleDevicewise(dstBuffer))
                return false;
            const auto& params = dstBuffer->getCreationParams();
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdUpdateBuffer.html#VUID-vkCmdUpdateBuffer-dstOffset-00034
            if (!params.usage.hasFlags(IGPUBuffer::EUF_TRANSFER_DST_BIT))
                return false;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdFillBuffer.html#VUID-vkCmdFillBuffer-size-00024
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdUpdateBuffer.html#VUID-vkCmdUpdateBuffer-dstOffset-00032
            if (dstOffset>=params.size)
                return false;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdFillBuffer.html#VUID-vkCmdFillBuffer-size-00025
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdUpdateBuffer.html#VUID-vkCmdUpdateBuffer-dstOffset-00036
            if ((dstOffset&0x03ull)!=0ull)
                return false;
            const bool notWholeSize = !fill/* || dataSize!=TODO*/;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdFillBuffer.html#VUID-vkCmdFillBuffer-size-00026
            if (notWholeSize && dataSize==0u)
                return false;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdUpdateBuffer.html#VUID-vkCmdUpdateBuffer-dstOffset-00033
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdFillBuffer.html#VUID-vkCmdFillBuffer-size-00027
            if (notWholeSize && dstOffset+dataSize>params.size)
                return false;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdUpdateBuffer.html#VUID-vkCmdUpdateBuffer-dstOffset-00038
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdFillBuffer.html#VUID-vkCmdFillBuffer-size-00028
            if (notWholeSize && (dataSize&0x03ull)!=0ull)
                return false;
            return true;
        }

        static void bindDescriptorSets_generic(const IGPUPipelineLayout* _newLayout, uint32_t _first, uint32_t _count, const IGPUDescriptorSet* const* _descSets, const IGPUPipelineLayout** const _destPplnLayouts)
        {
            int32_t compatibilityLimits[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT]{};
            for (uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
            {
                const int32_t lim = _destPplnLayouts[i] ? //if no descriptor set bound at this index
                    _destPplnLayouts[i]->isCompatibleUpToSet(IGPUPipelineLayout::DESCRIPTOR_SET_COUNT - 1u, _newLayout) : -1;

                compatibilityLimits[i] = lim;
            }

            /*
            https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#descriptorsets-compatibility
            When binding a descriptor set (see Descriptor Set Binding) to set number N, if the previously bound descriptor sets for sets zero through N-1 were all bound using compatible pipeline layouts, then performing this binding does not disturb any of the lower numbered sets.
            */
            for (int32_t i = 0; i < static_cast<int32_t>(_first); ++i)
                if (compatibilityLimits[i] < i)
                    _destPplnLayouts[i] = nullptr;
            /*
            If, additionally, the previous bound descriptor set for set N was bound using a pipeline layout compatible for set N, then the bindings in sets numbered greater than N are also not disturbed.
            */
            if (compatibilityLimits[_first] < static_cast<int32_t>(_first))
                for (uint32_t i = _first + 1u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
                    _destPplnLayouts[i] = nullptr;
        }

        virtual bool begin_impl(const core::bitflag<USAGE> flags, const SInheritanceInfo* inheritanceInfo) = 0;
        virtual bool reset_impl(const core::bitflag<RESET_FLAGS> flags) { return true; };
        virtual bool end_impl() = 0;

//        virtual bool setDeviceMask_impl(uint32_t deviceMask) { assert(!"Invalid code path"); return false; };

        virtual bool setEvent_impl(IGPUEvent* const _event, const SDependencyInfo& depInfo) = 0;
        virtual bool resetEvent_impl(IGPUEvent* const _event, const core::bitflag<stage_flags_t> stageMask) = 0;
        virtual bool waitEvents_impl(const uint32_t eventCount, IGPUEvent* const* const pEvents, const SDependencyInfo* depInfos) = 0;
        virtual bool pipelineBarrier_impl(const core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags, const SDependencyInfo& depInfo) = 0;

        virtual bool fillBuffer_impl(IGPUBuffer* const dstBuffer, const size_t dstOffset, const size_t size, const uint32_t data) = 0;
        virtual bool updateBuffer_impl(IGPUBuffer* const dstBuffer, const size_t dstOffset, const size_t dataSize, const void* const pData) = 0;
        virtual bool copyBuffer_impl(const IGPUBuffer* const srcBuffer, IGPUBuffer* const dstBuffer, const uint32_t regionCount, const SBufferCopy* const pRegions) = 0;

        virtual bool clearColorImage_impl(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearColorValue* const pColor, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges) = 0;
        virtual bool clearDepthStencilImage_impl(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearDepthStencilValue* const pDepthStencil, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges) = 0;
        virtual bool copyBufferToImage_impl(const IGPUBuffer* const srcBuffer, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions) = 0;
        virtual bool copyImageToBuffer_impl(const IGPUBuffer* const srcImage, const IGPUImage::LAYOUT srcImageLayout, const IGPUBuffer* const dstBuffer, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions) = 0;
        virtual bool copyImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SImageCopy* const pRegions) = 0;

        virtual bool copyAccelerationStructure_impl(const IGPUAccelerationStructure::CopyInfo& copyInfo) = 0;
        virtual bool copyAccelerationStructureToMemory_impl(const IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo) = 0;
        virtual bool copyAccelerationStructureFromMemory_impl(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo) = 0;

        virtual bool buildAccelerationStructures_impl(const core::SRange<const IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const video::IGPUAccelerationStructure::BuildRangeInfo* const* const ppBuildRangeInfos) = 0;
        virtual bool buildAccelerationStructuresIndirect_impl(const core::SRange<const IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const core::SRange<const IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses, const uint32_t* const pIndirectStrides, const uint32_t* const* const ppMaxPrimitiveCounts) = 0;

        virtual bool bindComputePipeline_impl(const IGPUComputePipeline* const pipeline) = 0;
        virtual bool bindGraphicsPipeline_impl(const IGPUGraphicsPipeline* const pipeline) = 0;
        virtual bool bindDescriptorSets_impl(
            const asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const IGPUPipelineLayout* const layout,
            const uint32_t firstSet, const uint32_t descriptorSetCount, const IGPUDescriptorSet* const* const pDescriptorSets,
            const uint32_t dynamicOffsetCount = 0u, const uint32_t* const dynamicOffsets = nullptr
        ) = 0;
        virtual bool pushConstants_impl(const IGPUPipelineLayout* const layout, core::bitflag<IGPUShader::E_SHADER_STAGE> stageFlags, const uint32_t offset, const uint32_t size, const void* const pValues) = 0;
        virtual bool bindVertexBuffers_impl(const uint32_t firstBinding, const uint32_t bindingCount, const IGPUBuffer* const* const pBuffers, const size_t* const pOffsets) = 0;
        virtual bool bindIndexBuffer_impl(const IGPUBuffer* const buffer, const size_t offset, const asset::E_INDEX_TYPE indexType) = 0;

        virtual bool resetQueryPool_impl(IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount) = 0;
        virtual bool beginQuery_impl(IQueryPool* const queryPool, const uint32_t query, const core::bitflag<QUERY_CONTROL_FLAGS> flags = QUERY_CONTROL_FLAGS::NONE) = 0;
        virtual bool endQuery_impl(IQueryPool* const queryPool, const uint32_t query) = 0;
        virtual bool writeTimestamp_impl(const asset::PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* const queryPool, const uint32_t query) = 0;
        virtual bool writeAccelerationStructureProperties_impl(const core::SRange<const IGPUAccelerationStructure>& pAccelerationStructures, const IQueryPool::E_QUERY_TYPE queryType, IQueryPool* const queryPool, const uint32_t firstQuery) = 0;
        virtual bool copyQueryPoolResults_impl(const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount, IGPUBuffer* const dstBuffer, const size_t dstOffset, const size_t stride, const core::bitflag<IQueryPool::E_QUERY_RESULTS_FLAGS> flags) = 0;
#if 0
        virtual bool drawIndirect_impl(const IGPUBuffer* buffer, size_t offset, uint32_t drawCount, uint32_t stride) = 0;
        virtual bool drawIndexedIndirect_impl(const IGPUBuffer* buffer, size_t offset, uint32_t drawCount, uint32_t stride) = 0;
        virtual bool drawIndirectCount_impl(const IGPUBuffer* buffer, size_t offset, const IGPUBuffer* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) = 0;
        virtual bool drawIndexedIndirectCount_impl(const IGPUBuffer* buffer, size_t offset, const IGPUBuffer* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) = 0;

        virtual bool beginRenderPass_impl(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content) = 0;
        virtual bool nextSubpass_impl(const asset::E_SUBPASS_CONTENTS contents) = 0;
        virtual bool endRenderPass_impl() = 0;

        virtual bool clearAttachments(uint32_t attachmentCount, const asset::SClearAttachment* pAttachments, uint32_t rectCount, const asset::SClearRect* pRects) = 0;
        virtual bool dispatchIndirect_impl(const IGPUBuffer* buffer, size_t offset) = 0;
#endif
        virtual bool blitImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const SImageBlit* pRegions, const IGPUSampler::E_TEXTURE_FILTER filter) = 0;
        virtual bool resolveImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const SImageResolve* pRegions) = 0;

        virtual bool executeCommands_impl(const uint32_t count, IGPUCommandBuffer* const* const cmdbufs) = 0;

        virtual void releaseResourcesBackToPool_impl() {}
        virtual void checkForParentPoolReset_impl() const = 0;


        core::smart_refctd_ptr<IGPUCommandPool> m_cmdpool;
        system::logger_opt_smart_ptr m_logger;
        const core::bitflag<stage_flags_t> m_supportedStageMask;
        const LEVEL m_level;

    private:
        // everything here is private on purpose so that derived class can't mess with these basic states
        inline bool checkForParentPoolReset()
        {
            if (m_cmdpool->getResetCounter() <= m_resetCheckedStamp)
                return false;

            m_resetCheckedStamp = m_cmdpool->getResetCounter();
            m_state = STATE::INITIAL;

            m_boundDescriptorSetsRecord.clear();

            m_commandList.head = nullptr;
            m_commandList.tail = nullptr;

            checkForParentPoolReset_impl();

            return true;
        }

        inline void releaseResourcesBackToPool()
        {
            deleteCommandList();
            m_boundDescriptorSetsRecord.clear();
            releaseResourcesBackToPool_impl();
        }

        inline void deleteCommandList()
        {
            m_cmdpool->m_commandListPool.deleteList(m_commandList.head);
            m_commandList.head = nullptr;
            m_commandList.tail = nullptr;
        }

        enum class RENDERPASS_SCOPE : uint8_t
        {
            OUTSIDE = 0x1u,
            INSIDE = 0x2u,
            BOTH = OUTSIDE|INSIDE
        };
        using queue_flags_t = IGPUQueue::FAMILY_FLAGS;
        bool checkStateBeforeRecording(const core::bitflag<queue_flags_t> allowedQueueFlags=queue_flags_t::NONE, const core::bitflag<RENDERPASS_SCOPE> renderpassScope=RENDERPASS_SCOPE::BOTH);

        bool validateDependency(const SDependencyInfo& depInfo) const;


        // This bound descriptor set record doesn't include the descriptor sets whose layout has _any_ one of its bindings
        // created with IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT
        // or IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT.
        core::unordered_map<const IGPUDescriptorSet*,uint64_t> m_boundDescriptorSetsRecord;
    
        IGPUCommandPool::CCommandSegmentListPool::SCommandSegmentList m_commandList = {};

        uint64_t m_resetCheckedStamp = 0;
        STATE m_state = STATE::INITIAL;
        // only useful while recording
        SInheritanceInfo m_cachedInheritanceInfo;
        core::bitflag<USAGE> m_recordingFlags = USAGE::NONE;
        //uint32_t m_deviceMask = ~0u;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IGPUCommandBuffer::USAGE);


}

#endif
