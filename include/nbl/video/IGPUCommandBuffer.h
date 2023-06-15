#ifndef _NBL_VIDEO_I_GPU_COMMAND_BUFFER_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_COMMAND_BUFFER_H_INCLUDED_

#include "nbl/core/declarations.h"

#include "nbl/asset/asset.h"

#include "nbl/video/IGPUEvent.h"
#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/video/IGPUComputePipeline.h"
#include "nbl/video/IGPUGraphicsPipeline.h"
#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/IGPUAccelerationStructure.h"
#include "nbl/video/IQueryPool.h"
#include "nbl/video/IGPUCommandPool.h"


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
            asset::IImage::SSubresourceRange subresourceRange = {};
            // If both layouts match, no transition is performed, so this is our default
            asset::IImage::LAYOUT oldLayout = IGPUImage::LAYOUT::UNDEFINED;
            asset::IImage::LAYOUT newLayout = IGPUImage::LAYOUT::UNDEFINED;
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
        bool setEvent(IGPUEvent* _event, const SDependencyInfo& depInfo);
        bool resetEvent(IGPUEvent* _event, const core::bitflag<stage_flags_t> stageMask);
        bool waitEvents(const uint32_t eventCount, IGPUEvent* const* const pEvents, const SDependencyInfo* depInfos);

        bool pipelineBarrier(const core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags, const SDependencyInfo& depInfo);

        //! buffer transfers
        bool fillBuffer(IGPUBuffer* dstBuffer, size_t dstOffset, size_t size, uint32_t data);
        bool updateBuffer(IGPUBuffer* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData);
        struct SBufferCopy
        {
            size_t srcOffset;
            size_t dstOffset;
            size_t size;
        };
        bool copyBuffer(const IGPUBuffer* srcBuffer, IGPUBuffer* dstBuffer, uint32_t regionCount, const SBufferCopy* pRegions);

#if 0
        //! image transfers
        bool clearColorImage(IGPUImage* image, asset::IImage::LAYOUT imageLayout, const asset::SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges);
        bool clearDepthStencilImage(IGPUImage* image, asset::IImage::LAYOUT imageLayout, const asset::SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges);
        bool copyBufferToImage(const IGPUBuffer* srcBuffer, IGPUImage* dstImage, asset::IImage::LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions);
        bool copyImageToBuffer(const IGPUBuffer* srcImage, asset::IImage::LAYOUT srcImageLayout, IGPUBuffer* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions);
        bool copyImage(const IGPUImage* srcImage, asset::IImage::LAYOUT srcImageLayout, IGPUImage* dstImage, asset::IImage::LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions);

        //! acceleration structure transfers
        bool copyAccelerationStructure(const video::IGPUAccelerationStructure::CopyInfo& copyInfo);
        bool copyAccelerationStructureToMemory(const video::IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo);
        bool copyAccelerationStructureFromMemory(const video::IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo);

        //! acceleration structure builds
        bool buildAccelerationStructures(const core::SRange<video::IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, video::IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos);
        bool buildAccelerationStructuresIndirect(const core::SRange<video::IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const core::SRange<video::IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses, const uint32_t* pIndirectStrides, const uint32_t* const* ppMaxPrimitiveCounts);

        //! state setup
        bool bindComputePipeline(const IGPUComputePipeline* pipeline);
        bool bindGraphicsPipeline(const IGPUGraphicsPipeline* pipeline);
        bool bindDescriptorSets(
            asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const IGPUPipelineLayout* layout,
            const uint32_t firstSet, const uint32_t descriptorSetCount, const IGPUDescriptorSet* const* const pDescriptorSets,
            const uint32_t dynamicOffsetCount=0u, const uint32_t* dynamicOffsets=nullptr
        );
        bool pushConstants(const IGPUPipelineLayout* layout, core::bitflag<asset::IShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues);
        bool bindVertexBuffers(uint32_t firstBinding, uint32_t bindingCount, const IGPUBuffer* const* const pBuffers, const size_t* pOffsets);
        bool bindIndexBuffer(const IGPUBuffer* buffer, size_t offset, asset::E_INDEX_TYPE indexType);

        //! dispatches
        bool dispatchIndirect(const IGPUBuffer* buffer, size_t offset);

        //! queries

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


        bool resetQueryPool(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount);
        bool writeTimestamp(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, video::IQueryPool* queryPool, uint32_t query);

        bool writeAccelerationStructureProperties(const core::SRange<video::IGPUAccelerationStructure>& pAccelerationStructures, video::IQueryPool::E_QUERY_TYPE queryType, video::IQueryPool* queryPool, uint32_t firstQuery);

        bool beginQuery(video::IQueryPool* queryPool, uint32_t query, core::bitflag<QUERY_CONTROL_FLAGS> flags = QUERY_CONTROL_FLAGS::NONE);
        bool endQuery(video::IQueryPool* queryPool, uint32_t query);
        bool copyQueryPoolResults(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, IGPUBuffer* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS> flags);




        struct SImageBlit
        {
            IGPUImage::SSubresourceLayers srcSubresource;
            asset::VkOffset3D srcOffsets[2];
            IGPUImage::SSubresourceLayers dstSubresource;
            asset::VkOffset3D dstOffsets[2];
        };
        bool blitImage(const image_t* srcImage, asset::IImage::LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter) final override;
        struct SImageResolve
        {
            IGPUImage::SSubresourceLayers srcSubresource;
            asset::VkOffset3D srcOffset;
            IGPUImage::SSubresourceLayers dstSubresource;
            asset::VkOffset3D dstOffset;
            asset::VkExtent3D extent;
        };
        bool resolveImage(const image_t* srcImage, asset::IImage::LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageResolve* pRegions);
#endif
        //! Secondary CommandBuffer execute
        bool executeCommands(const uint32_t count, IGPUCommandBuffer* const* const cmdbufs);

        // Vulkan: const VkCommandBuffer*
        virtual const void* getNativeHandle() const = 0;

        inline const core::unordered_map<const IGPUDescriptorSet*, uint64_t>& getBoundDescriptorSetsRecord() const { return m_boundDescriptorSetsRecord; }

    protected: 
        friend class IGPUQueue;

        inline IGPUCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const LEVEL lvl, core::smart_refctd_ptr<IGPUCommandPool>&& _cmdpool, system::logger_opt_smart_ptr&& logger)
            : IBackendObject(std::move(dev)), m_cmdpool(_cmdpool), m_logger(std::move(logger)), m_level(lvl), m_supportedStageMask(getOriginDevice()->getSupportedStagesMask(m_cmdpool->getQueueFamilyIndex()))
        {
        }

        virtual ~IGPUCommandBuffer()
        {
            // Only release the resources if the parent pool has not been reset because if it has been then the resources will already be released.
            if (!checkForParentPoolReset())
            {
                releaseResourcesBackToPool();
            }
        }

        inline bool validate_updateBuffer(IGPUBuffer* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData)
        {
            if (!dstBuffer)
                return false;
            if (!this->isCompatibleDevicewise(dstBuffer))
                return false;
            if ((dstOffset & 0x03ull) != 0ull)
                return false;
            if ((dataSize & 0x03ull) != 0ull)
                return false;
            if (dataSize > 65536ull)
                return false;
            return dstBuffer->getCreationParams().usage.hasFlags(IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF);
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

        virtual bool setEvent_impl(IGPUEvent* _event, const SDependencyInfo& depInfo) = 0;
        virtual bool resetEvent_impl(IGPUEvent* _event, const core::bitflag<stage_flags_t> stageMask) = 0;
        virtual bool waitEvents_impl(const uint32_t eventCount, IGPUEvent* const* const pEvents, const SDependencyInfo* depInfos) = 0;

        virtual bool pipelineBarrier_impl(const core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags, const SDependencyInfo& depInfo) = 0;
#if 0
        virtual void bindIndexBuffer_impl(const IGPUBuffer* buffer, size_t offset, asset::E_INDEX_TYPE indexType) = 0;
        virtual bool drawIndirect_impl(const IGPUBuffer* buffer, size_t offset, uint32_t drawCount, uint32_t stride) = 0;
        virtual bool drawIndexedIndirect_impl(const IGPUBuffer* buffer, size_t offset, uint32_t drawCount, uint32_t stride) = 0;
        virtual bool drawIndirectCount_impl(const IGPUBuffer* buffer, size_t offset, const IGPUBuffer* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) = 0;
        virtual bool drawIndexedIndirectCount_impl(const IGPUBuffer* buffer, size_t offset, const IGPUBuffer* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) = 0;

        virtual bool beginRenderPass_impl(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content) = 0;
        virtual bool nextSubpass_impl(const asset::E_SUBPASS_CONTENTS contents) = 0;
        virtual bool endRenderPass_impl() = 0;

        virtual bool bindDescriptorSets_impl(asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const pipeline_layout_t* layout, uint32_t firstSet, const uint32_t descriptorSetCount,
            const descriptor_set_t* const* const pDescriptorSets, const uint32_t dynamicOffsetCount = 0u, const uint32_t* dynamicOffsets = nullptr) = 0;
        virtual void bindComputePipeline_impl(const compute_pipeline_t* pipeline) = 0;
        virtual bool updateBuffer_impl(IGPUBuffer* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData) = 0;

        virtual bool buildAccelerationStructures_impl(const core::SRange<video::IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, video::IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos) { assert(!"Invalid code path."); return false; }
        virtual bool buildAccelerationStructuresIndirect_impl(const core::SRange<video::IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const core::SRange<video::IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses, const uint32_t* pIndirectStrides, const uint32_t* const* ppMaxPrimitiveCounts) { assert(!"Invalid code path."); return false; }
        virtual bool copyAccelerationStructure_impl(const video::IGPUAccelerationStructure::CopyInfo& copyInfo) { assert(!"Invalid code path."); return false; }
        virtual bool copyAccelerationStructureToMemory_impl(const video::IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo) { assert(!"Invalid code path."); return false; }
        virtual bool copyAccelerationStructureFromMemory_impl(const video::IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo) { assert(!"Invaild code path."); return false; }

        virtual bool resetQueryPool_impl(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount) = 0;
        virtual bool writeTimestamp_impl(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, video::IQueryPool* queryPool, uint32_t query) = 0;
        virtual bool writeAccelerationStructureProperties_impl(const core::SRange<video::IGPUAccelerationStructure>& pAccelerationStructures, video::IQueryPool::E_QUERY_TYPE queryType, video::IQueryPool* queryPool, uint32_t firstQuery) { assert(!"Invalid code path."); return false; }
        virtual bool beginQuery_impl(video::IQueryPool* queryPool, uint32_t query, core::bitflag<QUERY_CONTROL_FLAGS> flags = QUERY_CONTROL_FLAGS::NONE) = 0;
        virtual bool endQuery_impl(video::IQueryPool* queryPool, uint32_t query) = 0;
        virtual bool copyQueryPoolResults_impl(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, IGPUBuffer* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS> flags) = 0;

        virtual bool bindGraphicsPipeline_impl(const graphics_pipeline_t* pipeline) = 0;
        virtual bool pushConstants_impl(const pipeline_layout_t* layout, core::bitflag<asset::IShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) = 0;
        virtual bool clearColorImage_impl(image_t* image, asset::IImage::LAYOUT imageLayout, const asset::SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) = 0;
        virtual bool clearDepthStencilImage_impl(image_t* image, asset::IImage::LAYOUT imageLayout, const asset::SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) = 0;
        virtual bool clearAttachments(uint32_t attachmentCount, const asset::SClearAttachment* pAttachments, uint32_t rectCount, const asset::SClearRect* pRects) = 0;
        virtual bool fillBuffer_impl(IGPUBuffer* dstBuffer, size_t dstOffset, size_t size, uint32_t data) = 0;
        virtual void bindVertexBuffers_impl(uint32_t firstBinding, uint32_t bindingCount, const IGPUBuffer* const* const pBuffers, const size_t* pOffsets) = 0;
        virtual bool dispatchIndirect_impl(const IGPUBuffer* buffer, size_t offset) = 0;
        virtual bool copyBuffer_impl(const IGPUBuffer* srcBuffer, IGPUBuffer* dstBuffer, uint32_t regionCount, const SBufferCopy* pRegions) = 0;
        virtual bool copyImage_impl(const image_t* srcImage, asset::IImage::LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions) = 0;
        virtual bool copyBufferToImage_impl(const IGPUBuffer* srcBuffer, image_t* dstImage, asset::IImage::LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) = 0;
        virtual bool blitImage_impl(const image_t* srcImage, asset::IImage::LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter) = 0;
        virtual bool copyImageToBuffer_impl(const image_t* srcImage, asset::IImage::LAYOUT srcImageLayout, IGPUBuffer* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) = 0;
        virtual bool resolveImage_impl(const image_t* srcImage, asset::IImage::LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageResolve* pRegions) = 0;
#endif
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
