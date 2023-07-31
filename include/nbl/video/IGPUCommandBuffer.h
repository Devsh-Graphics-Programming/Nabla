#ifndef _NBL_VIDEO_I_GPU_COMMAND_BUFFER_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_COMMAND_BUFFER_H_INCLUDED_

#include "nbl/core/declarations.h"

#include "nbl/asset/asset.h"

#include "nbl/video/IGPUShader.h"
#include "nbl/video/IGPUCommandPool.h"
#include "nbl/video/IQueue.h"


#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <type_traits>


namespace nbl::video
{

// depr
class IGPUMeshBuffer;

class NBL_API2 IGPUCommandBuffer : public IBackendObject
{
    public:
        inline IGPUCommandPool::BUFFER_LEVEL getLevel() const { return m_level; }

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
            return m_cmdpool->getCreationFlags().hasFlags(IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
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

        using stage_flags_t = asset::PIPELINE_STAGE_FLAGS;
        //! Barriers and Sync
        template<typename ResourceBarrier>
        struct SBufferMemoryBarrier
        {
            ResourceBarrier barrier = {};
            asset::SBufferRange<const IGPUBuffer> range = {};
        };
        template<typename ResourceBarrier>
        struct SImageMemoryBarrier
        {
            ResourceBarrier barrier = {};
            const IGPUImage* image = nullptr;
            IGPUImage::SSubresourceRange subresourceRange = {};
            // If both layouts match, no transition is performed, so this is our default
            IGPUImage::LAYOUT oldLayout = IGPUImage::LAYOUT::UNDEFINED;
            IGPUImage::LAYOUT newLayout = IGPUImage::LAYOUT::UNDEFINED;
        };
        template<typename ResourceBarrier>
        struct SDependencyInfo
        {
            using buffer_barrier_t = SBufferMemoryBarrier<ResourceBarrier>;
            using image_barrier_t = SImageMemoryBarrier<ResourceBarrier>;

            // no dependency flags because they must be 0 per the spec
            uint32_t memBarrierCount = 0;
            const asset::SMemoryBarrier* memBarriers = nullptr;
            uint32_t bufBarrierCount = 0;
            const buffer_barrier_t* bufBarriers = nullptr;
            uint32_t imgBarrierCount = 0;
            const image_barrier_t* imgBarriers = nullptr;
        };

        using SEventDependencyInfo = SDependencyInfo<asset::SMemoryBarrier>;
        bool setEvent(IEvent* const _event, const SEventDependencyInfo& depInfo);
        bool resetEvent(IEvent* _event, const core::bitflag<stage_flags_t> stageMask);
        bool waitEvents(const uint32_t eventCount, IEvent* const* const pEvents, const SEventDependencyInfo* depInfos);
        
        struct SOwnershipTransferBarrier
        {
            asset::SMemoryBarrier dep = {};
            // If otherQueueFamilyIndex==FamilyIgnored there will be no ownership transfer
            enum class OWNERSHIP_OP : uint32_t 
            {
                RELEASE = 0,
                ACQUIRE = 1
            };
            OWNERSHIP_OP ownershipOp : 1 = OWNERSHIP_OP::ACQUIRE;
            uint32_t otherQueueFamilyIndex : 31 = IQueue::FamilyIgnored;
        };
        using SPipelineBarrierDependencyInfo = SDependencyInfo<SOwnershipTransferBarrier>;
        bool pipelineBarrier(const core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags, const SPipelineBarrierDependencyInfo& depInfo);

        //! buffer transfers
        bool fillBuffer(const asset::SBufferRange<IGPUBuffer>& range, const uint32_t data);
        bool updateBuffer(const asset::SBufferRange<IGPUBuffer>& range, const void* const pData);
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
        bool clearColorImage(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearColorValue* const pColor, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges);
        bool clearDepthStencilImage(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearDepthStencilValue* const pDepthStencil, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges);
        bool copyBufferToImage(const IGPUBuffer* const srcBuffer, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions);
        bool copyImageToBuffer(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, const IGPUBuffer* const dstBuffer, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions);
        bool copyImage(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SImageCopy* const pRegions);
        
        //! acceleration structure builds
        inline bool buildAccelerationStructures(const core::SRange<const IGPUBottomLevelAccelerationStructure::DeviceBuildInfo>& infos, const IGPUBottomLevelAccelerationStructure::DirectBuildRangeRangeInfos buildRangeInfos)
        {
            if (const auto totalGeometryCount=buildAccelerationStructures_common(infos,buildRangeInfos); totalGeometryCount)
                return buildAccelerationStructures_impl(infos,buildRangeInfos,totalGeometryCount);
            return false;
        }
        inline bool buildAccelerationStructures(const core::SRange<const IGPUTopLevelAccelerationStructure::DeviceBuildInfo>& infos, const IGPUTopLevelAccelerationStructure::DirectBuildRangeRangeInfos buildRangeInfos)
        {
            if (buildAccelerationStructures_common(infos,buildRangeInfos))
                return buildAccelerationStructures_impl(infos,buildRangeInfos);
            return false;
        }
        // We don't allow different indirect command addresses due to https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectDeviceAddresses-03646
        template<class AccelerationStructure> requires std::is_base_of_v<IGPUAccelerationStructure,AccelerationStructure>
        inline bool buildAccelerationStructuresIndirect(
            const IGPUBuffer* indirectRangeBuffer, const core::SRange<const typename AccelerationStructure::DeviceBuildInfo>& infos,
            const uint64_t* const pIndirectOffsets, const uint32_t* const pIndirectStrides, typename AccelerationStructure::MaxInputCounts* const maxPrimitiveOrInstanceCounts
        )
        {
            if (!maxPrimitiveOrInstanceCounts || !pIndirectStrides || !pIndirectOffsets)
                return false;

            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectDeviceAddresses-03645
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectDeviceAddresses-03647
            if (!indirectRangeBuffer || indirectRangeBuffer->getDeviceAddress()==0ull || !indirectRangeBuffer->getCreationParams().usage.hasFlags(IGPUBuffer::EUF_INDIRECT_BUFFER_BIT))
                return false;

            for (auto i=0u; i<infos.size(); i++)
            {
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectDeviceAddresses-03648
                if (!core::is_aligned_to(pIndirectOffsets[i],alignof(uint32_t)))
                    return false;
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectStrides-03787
                if (!core::is_aligned_to(pIndirectStrides[i],alignof(uint32_t)))
                    return false;
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectDeviceAddresses-03646
                if (pIndirectOffsets[i]+infos[i].inputCount()*pIndirectStrides[i]>indirectRangeBuffer->getSize())
                    return false;
            }

            if (const auto totalGeometryCount=buildAccelerationStructures_common(infos,maxPrimitiveOrInstanceCounts,indirectRangeBuffer); totalGeometryCount)
            {
                if constexpr(std::is_same_v<AccelerationStructure,IGPUBottomLevelAccelerationStructure>)
                    return buildAccelerationStructuresIndirect_impl(indirectRangeBuffer,infos,pIndirectOffsets,pIndirectStrides,maxPrimitiveOrInstanceCounts,totalGeometryCount);
                else
                    return buildAccelerationStructuresIndirect_impl(indirectRangeBuffer,infos,pIndirectOffsets,pIndirectStrides,maxPrimitiveOrInstanceCounts);
            }
            return false;
        }
        
        //! acceleration structure transfers
        bool copyAccelerationStructure(const IGPUAccelerationStructure::CopyInfo& copyInfo);
        bool copyAccelerationStructureToMemory(const IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo);
        bool copyAccelerationStructureFromMemory(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo);

        //! state setup
        bool bindComputePipeline(const IGPUComputePipeline* const pipeline);
        bool bindGraphicsPipeline(const IGPUGraphicsPipeline* const pipeline);
        bool bindDescriptorSets(
            const asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const IGPUPipelineLayout* const layout,
            const uint32_t firstSet, const uint32_t descriptorSetCount, const IGPUDescriptorSet* const* const pDescriptorSets,
            const uint32_t dynamicOffsetCount=0u, const uint32_t* const dynamicOffsets=nullptr
        );
        bool pushConstants(const IGPUPipelineLayout* const layout, const core::bitflag<IGPUShader::E_SHADER_STAGE> stageFlags, const uint32_t offset, const uint32_t size, const void* const pValues);
        bool bindVertexBuffers(const uint32_t firstBinding, const uint32_t bindingCount, const asset::SBufferBinding<const IGPUBuffer>* const pBindings);
        bool bindIndexBuffer(const asset::SBufferBinding<const IGPUBuffer>& binding, const asset::E_INDEX_TYPE indexType);

        //! dynamic state
        inline bool setScissor(const uint32_t first, const uint32_t count, const VkRect2D* const pScissors)
        {
            if(invalidDynamic(first,count))
                return false;

            for (auto i=0u; i<count; i++)
            {
                const auto& scissor = pScissors[i];
                if (scissor.offset.x<0 || scissor.offset.y<0)
                    return false;
                if (pScissors[i].extent.width>std::numeric_limits<int32_t>::max()-scissor.offset.x)
                    return false;
                if (pScissors[i].extent.height>std::numeric_limits<int32_t>::max()-scissor.offset.y)
                    return false;
            }

            return setScissor_impl(first,count,pScissors);
        }
        inline bool setViewport(const uint32_t first, const uint32_t count, const asset::SViewport* const pViewports)
        {
            if (invalidDynamic(first,count))
                return false;

            return setViewport_impl(first,count,pViewports);
        }

        //! queries
        bool resetQueryPool(IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount);
        bool beginQuery(IQueryPool* const queryPool, const uint32_t query, const core::bitflag<QUERY_CONTROL_FLAGS> flags=QUERY_CONTROL_FLAGS::NONE);
        bool endQuery(IQueryPool* const queryPool, const uint32_t query);
        bool writeTimestamp(const asset::PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* const queryPool, const uint32_t query);
        bool writeAccelerationStructureProperties(const core::SRange<const IGPUAccelerationStructure*>& pAccelerationStructures, const IQueryPool::TYPE queryType, IQueryPool* const queryPool, const uint32_t firstQuery);
        bool copyQueryPoolResults(
            const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount,
            const asset::SBufferBinding<IGPUBuffer>& dstBuffer, const size_t stride, const core::bitflag<IQueryPool::RESULTS_FLAGS> flags
        );

        //! dispatches
        bool dispatch(const uint32_t groupCountX, const uint32_t groupCountY=1, const uint32_t groupCountZ=1);
        bool dispatchIndirect(const asset::SBufferBinding<const IGPUBuffer>& binding);

        //! Begin/End RenderPasses
        struct SRenderpassBeginInfo
        {
            IGPUFramebuffer* framebuffer;
            const SClearColorValue* colorClearValues;
            const SClearDepthStencilValue* depthStencilClearValues;
            VkRect2D renderArea;
        };
        enum SUBPASS_CONTENTS : uint8_t
        {
            INLINE = 0,
            SECONDARY_COMMAND_BUFFERS = 1
        };
        bool beginRenderPass(const SRenderpassBeginInfo& info, const SUBPASS_CONTENTS contents);
        bool nextSubpass(const SUBPASS_CONTENTS contents);
        bool endRenderPass();

        struct SClearAttachments
        {
            struct SRegion
            {
                VkRect2D rect = { {0u,0u},{0u,0u} };
                uint32_t baseArrayLayer = 0u;
                uint32_t layerCount = 0u;

                inline bool used() const {return rect.extent.width!=0u || rect.extent.height!=0u || layerCount!=0u;}
            };

            SRegion depthStencilRegion = {};
            SClearDepthStencilValue depthStencilValue = {};
            SRegion colorRegions[IGPURenderpass::SCreationParams::SSubpassDescription::MaxColorAttachments] = {};
            SClearColorValue colorValues[IGPURenderpass::SCreationParams::SSubpassDescription::MaxColorAttachments] = {};
            // default is don't clear depth stencil
            IGPUImage::E_ASPECT_FLAGS depthStencilAspectMask = IGPUImage::EAF_NONE;
        };
        bool clearAttachments(const SClearAttachments& info);

        //! draws
        bool draw(const uint32_t vertexCount, const uint32_t instanceCount, const uint32_t firstVertex, const uint32_t firstInstance);
        bool drawIndexed(const uint32_t indexCount, const uint32_t instanceCount, const uint32_t firstIndex, const int32_t vertexOffset, const uint32_t firstInstance);
        bool drawIndirect(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, const uint32_t stride);
        bool drawIndexedIndirect(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, const uint32_t stride);
        bool drawIndirectCount(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride);
        bool drawIndexedIndirectCount(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride);
        /* soon: [[deprecated]] */ bool drawMeshBuffer(const IGPUMeshBuffer* const meshBuffer);

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
        friend class IQueue;

        IGPUCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const IGPUCommandPool::BUFFER_LEVEL lvl, core::smart_refctd_ptr<IGPUCommandPool>&& _cmdpool, system::logger_opt_smart_ptr&& logger);
        virtual ~IGPUCommandBuffer()
        {
            // Only release the resources if the parent pool has not been reset because if it has been then the resources will already be released.
            if (!checkForParentPoolReset())
            {
                releaseResourcesBackToPool();
            }
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

        virtual bool setEvent_impl(IEvent* const _event, const SEventDependencyInfo& depInfo) = 0;
        virtual bool resetEvent_impl(IEvent* const _event, const core::bitflag<stage_flags_t> stageMask) = 0;
        virtual bool waitEvents_impl(const uint32_t eventCount, IEvent* const* const pEvents, const SEventDependencyInfo* depInfos) = 0;
        virtual bool pipelineBarrier_impl(const core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags, const SPipelineBarrierDependencyInfo& depInfo) = 0;

        virtual bool fillBuffer_impl(const asset::SBufferRange<IGPUBuffer>& range, const uint32_t data) = 0;
        virtual bool updateBuffer_impl(const asset::SBufferRange<IGPUBuffer>& range, const void* const pData) = 0;
        virtual bool copyBuffer_impl(const IGPUBuffer* const srcBuffer, IGPUBuffer* const dstBuffer, const uint32_t regionCount, const SBufferCopy* const pRegions) = 0;

        virtual bool clearColorImage_impl(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearColorValue* const pColor, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges) = 0;
        virtual bool clearDepthStencilImage_impl(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearDepthStencilValue* const pDepthStencil, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges) = 0;
        virtual bool copyBufferToImage_impl(const IGPUBuffer* const srcBuffer, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions) = 0;
        virtual bool copyImageToBuffer_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, const IGPUBuffer* const dstBuffer, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions) = 0;
        virtual bool copyImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SImageCopy* const pRegions) = 0;
        
        virtual bool buildAccelerationStructures_impl(
            const core::SRange<const IGPUBottomLevelAccelerationStructure::DeviceBuildInfo>& infos,
            const IGPUBottomLevelAccelerationStructure::BuildRangeInfo* const* const ppBuildRangeInfos,
            const uint32_t totalGeometryCount
        ) = 0;
        virtual bool buildAccelerationStructures_impl(const core::SRange<const IGPUTopLevelAccelerationStructure::DeviceBuildInfo>& infos, const IGPUTopLevelAccelerationStructure::BuildRangeInfo* const pBuildRangeInfos) = 0;
        virtual bool buildAccelerationStructuresIndirect_impl(
            const IGPUBuffer* indirectRangeBuffer, const core::SRange<const IGPUBottomLevelAccelerationStructure::DeviceBuildInfo>& infos,
            const uint64_t* const pIndirectOffsets, const uint32_t* const pIndirectStrides,
            const uint32_t* const* const ppMaxPrimitiveCounts, const uint32_t totalGeometryCount
        ) = 0;
        virtual bool buildAccelerationStructuresIndirect_impl(
            const IGPUBuffer* indirectRangeBuffer, const core::SRange<const IGPUTopLevelAccelerationStructure::DeviceBuildInfo>& infos,
            const uint64_t* const pIndirectOffsets, const uint32_t* const pIndirectStrides, const uint32_t* const pMaxInstanceCounts
        ) = 0;

        virtual bool copyAccelerationStructure_impl(const IGPUAccelerationStructure::CopyInfo& copyInfo) = 0;
        virtual bool copyAccelerationStructureToMemory_impl(const IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo) = 0;
        virtual bool copyAccelerationStructureFromMemory_impl(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo) = 0;

        virtual bool bindComputePipeline_impl(const IGPUComputePipeline* const pipeline) = 0;
        virtual bool bindGraphicsPipeline_impl(const IGPUGraphicsPipeline* const pipeline) = 0;
        virtual bool bindDescriptorSets_impl(
            const asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const IGPUPipelineLayout* const layout,
            const uint32_t firstSet, const uint32_t descriptorSetCount, const IGPUDescriptorSet* const* const pDescriptorSets,
            const uint32_t dynamicOffsetCount = 0u, const uint32_t* const dynamicOffsets = nullptr
        ) = 0;
        virtual bool pushConstants_impl(const IGPUPipelineLayout* const layout, const core::bitflag<IGPUShader::E_SHADER_STAGE> stageFlags, const uint32_t offset, const uint32_t size, const void* const pValues) = 0;
        virtual bool bindVertexBuffers_impl(const uint32_t firstBinding, const uint32_t bindingCount, const asset::SBufferBinding<const IGPUBuffer>* const pBindings) = 0;
        virtual bool bindIndexBuffer_impl(const asset::SBufferBinding<const IGPUBuffer>& binding, const asset::E_INDEX_TYPE indexType) = 0;

        virtual bool setScissor_impl(const uint32_t first, const uint32_t count, const VkRect2D* const pScissors) = 0;
        virtual bool setViewport_impl(const uint32_t first, const uint32_t count, const asset::SViewport* const pViewports) = 0;

        virtual bool resetQueryPool_impl(IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount) = 0;
        virtual bool beginQuery_impl(IQueryPool* const queryPool, const uint32_t query, const core::bitflag<QUERY_CONTROL_FLAGS> flags = QUERY_CONTROL_FLAGS::NONE) = 0;
        virtual bool endQuery_impl(IQueryPool* const queryPool, const uint32_t query) = 0;
        virtual bool writeTimestamp_impl(const asset::PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* const queryPool, const uint32_t query) = 0;
        virtual bool writeAccelerationStructureProperties_impl(const core::SRange<const IGPUAccelerationStructure*>& pAccelerationStructures, const IQueryPool::TYPE queryType, IQueryPool* const queryPool, const uint32_t firstQuery) = 0;
        virtual bool copyQueryPoolResults_impl(const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount, const asset::SBufferBinding<IGPUBuffer>& dstBuffer, const size_t stride, const core::bitflag<IQueryPool::RESULTS_FLAGS> flags) = 0;
        
        virtual bool dispatch_impl(const uint32_t groupCountX, const uint32_t groupCountY, const uint32_t groupCountZ) = 0;
        virtual bool dispatchIndirect_impl(const asset::SBufferBinding<const IGPUBuffer>& binding) = 0;

        virtual bool beginRenderPass_impl(const SRenderpassBeginInfo& info, SUBPASS_CONTENTS contents) = 0;
        virtual bool nextSubpass_impl(const SUBPASS_CONTENTS contents) = 0;
        virtual bool endRenderPass_impl() = 0;

        virtual bool clearAttachments_impl(const SClearAttachments& info) = 0;

        virtual bool draw_impl(const uint32_t vertexCount, const uint32_t instanceCount, const uint32_t firstVertex, const uint32_t firstInstance) = 0;
        virtual bool drawIndexed_impl(const uint32_t indexCount, const uint32_t instanceCount, const uint32_t firstIndex, const int32_t vertexOffset, const uint32_t firstInstance) = 0;
        virtual bool drawIndirect_impl(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, const uint32_t stride) = 0;
        virtual bool drawIndexedIndirect_impl(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, const uint32_t stride) = 0;
        virtual bool drawIndirectCount_impl(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride) = 0;
        virtual bool drawIndexedIndirectCount_impl(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride) = 0;

        virtual bool blitImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const SImageBlit* pRegions, const IGPUSampler::E_TEXTURE_FILTER filter) = 0;
        virtual bool resolveImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const SImageResolve* pRegions) = 0;

        virtual bool executeCommands_impl(const uint32_t count, IGPUCommandBuffer* const* const cmdbufs) = 0;

        virtual void releaseResourcesBackToPool_impl() {}
        virtual void checkForParentPoolReset_impl() const = 0;


        core::smart_refctd_ptr<IGPUCommandPool> m_cmdpool;
        system::logger_opt_smart_ptr m_logger;
        const IGPUCommandPool::BUFFER_LEVEL m_level;

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
        using queue_flags_t = IQueue::FAMILY_FLAGS;
        bool checkStateBeforeRecording(const core::bitflag<queue_flags_t> allowedQueueFlags=queue_flags_t::NONE, const core::bitflag<RENDERPASS_SCOPE> renderpassScope=RENDERPASS_SCOPE::BOTH);

        template<typename ResourceBarrier>
        bool invalidDependency(const SDependencyInfo<ResourceBarrier>& depInfo) const;

        inline bool invalidBuffer(const IGPUBuffer* buffer, const IGPUBuffer::E_USAGE_FLAGS usages) const
        {
            if (!buffer || !this->isCompatibleDevicewise(buffer))
                return true;
            if (!buffer->getCreationParams().usage.hasFlags(usages))
            {
                m_logger.log("Incorrect `IGPUBuffer` usage flags for the command!", system::ILogger::ELL_ERROR);
                return true;
            }
            return false;
        }
        inline bool invalidBufferBinding(const asset::SBufferBinding<const IGPUBuffer>& binding, const size_t alignment, const IGPUBuffer::E_USAGE_FLAGS usages) const
        {
            if (!binding.isValid() || invalidBuffer(binding.buffer.get(),usages))
                return true;
            if (binding.offset&(alignment-1))
            {
                m_logger.log("Offset %d not aligned to %d for the command!", system::ILogger::ELL_ERROR, binding.offset, alignment);
                return true;
            }
            return false;
        }
        inline bool invalidBufferRange(const asset::SBufferRange<const IGPUBuffer>& range, const size_t alignment, const IGPUBuffer::E_USAGE_FLAGS usages) const
        {
            if (invalidBufferBinding({range.offset,range.buffer},alignment,usages))
                return true;
            if ((range.size&(alignment-1)) && range.size!=asset::SBufferRange<IGPUBuffer>::WholeBuffer)
                return true;
            return false;
        }

        inline bool invalidImage(const IGPUImage* image, const IGPUImage::E_USAGE_FLAGS usages) const
        {
            if (!image || !this->isCompatibleDevicewise(image))
                return true;
            if (!image->getCreationParameters().usage.hasFlags(usages))
            {
                m_logger.log("Incorrect `IGPUImage` usage flags for the command!", system::ILogger::ELL_ERROR);
                return true;
            }
            return false;
        }
        template<bool clear=false>
        inline bool invalidDestinationImage(const IGPUImage* image, const IGPUImage::LAYOUT layout) const
        {
            switch (layout)
            {
                case IGPUImage::LAYOUT::GENERAL: [[fallthrough]];
                case IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL: [[fallthrough]];
                case IGPUImage::LAYOUT::SHARED_PRESENT:
                    break;
                default:
                    return true;
            }
            if (invalidImage(image,IGPUImage::EUF_TRANSFER_DST_BIT))
                return true;
            if constexpr (!clear)
            {
                if (image->getCreationParameters().samples!=IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT)
                    return true;
            }
            return false;
        }
        inline bool invalidSourceImage(const IGPUImage* image, const IGPUImage::LAYOUT layout) const
        {
            switch (layout)
            {
                case IGPUImage::LAYOUT::GENERAL: [[fallthrough]];
                case IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL: [[fallthrough]];
                case IGPUImage::LAYOUT::SHARED_PRESENT:
                    break;
                default:
                    return true;
            }
            return invalidImage(image,IGPUImage::EUF_TRANSFER_SRC_BIT);
        }
        
        // returns total number of Geometries across all AS build infos
        template<class DeviceBuildInfo, typename BuildRangeInfos>
        uint32_t buildAccelerationStructures_common(const core::SRange<const DeviceBuildInfo>& infos, BuildRangeInfos ranges, const IGPUBuffer* const indirectBuffer=nullptr);

        bool invalidDynamic(const uint32_t first, const uint32_t count);

        template<typename IndirectCommand>
        bool invalidDrawIndirect(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, uint32_t stride);

        template<typename IndirectCommand>
        bool invalidDrawIndirectCount(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride);

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

#ifndef _NBL_VIDEO_I_GPU_COMMAND_BUFFER_CPP_
extern template uint32_t IGPUCommandBuffer::buildAccelerationStructures_common<IGPUBottomLevelAccelerationStructure::DeviceBuildInfo,IGPUBottomLevelAccelerationStructure::DirectBuildRangeRangeInfos>(
    const core::SRange<const IGPUBottomLevelAccelerationStructure::DeviceBuildInfo>&, IGPUBottomLevelAccelerationStructure::DirectBuildRangeRangeInfos, const IGPUBuffer* const
);
extern template uint32_t IGPUCommandBuffer::buildAccelerationStructures_common<IGPUBottomLevelAccelerationStructure::DeviceBuildInfo,IGPUBottomLevelAccelerationStructure::MaxInputCounts>(
    const core::SRange<const IGPUBottomLevelAccelerationStructure::DeviceBuildInfo>&, IGPUBottomLevelAccelerationStructure::MaxInputCounts, const IGPUBuffer* const
);
extern template uint32_t IGPUCommandBuffer::buildAccelerationStructures_common<IGPUTopLevelAccelerationStructure::DeviceBuildInfo,IGPUTopLevelAccelerationStructure::DirectBuildRangeRangeInfos>(
    const core::SRange<const IGPUTopLevelAccelerationStructure::DeviceBuildInfo>&, IGPUTopLevelAccelerationStructure::DirectBuildRangeRangeInfos, const IGPUBuffer* const
);
extern template uint32_t IGPUCommandBuffer::buildAccelerationStructures_common<IGPUTopLevelAccelerationStructure::DeviceBuildInfo,IGPUTopLevelAccelerationStructure::MaxInputCounts>(
    const core::SRange<const IGPUTopLevelAccelerationStructure::DeviceBuildInfo>&, IGPUTopLevelAccelerationStructure::MaxInputCounts, const IGPUBuffer* const
);

extern template bool IGPUCommandBuffer::invalidDrawIndirect<asset::DrawArraysIndirectCommand_t>(const asset::SBufferBinding<const IGPUBuffer>&, const uint32_t, uint32_t);
extern template bool IGPUCommandBuffer::invalidDrawIndirect<asset::DrawElementsIndirectCommand_t>(const asset::SBufferBinding<const IGPUBuffer>&, const uint32_t, uint32_t);
extern template bool IGPUCommandBuffer::invalidDrawIndirectCount<asset::DrawArraysIndirectCommand_t>(const asset::SBufferBinding<const IGPUBuffer>&, const asset::SBufferBinding<const IGPUBuffer>&, const uint32_t, const uint32_t);
extern template bool IGPUCommandBuffer::invalidDrawIndirectCount<asset::DrawElementsIndirectCommand_t>(const asset::SBufferBinding<const IGPUBuffer>&, const asset::SBufferBinding<const IGPUBuffer>&, const uint32_t, const uint32_t);

extern template bool IGPUCommandBuffer::invalidDependency(const SDependencyInfo<asset::SMemoryBarrier>&) const;
extern template bool IGPUCommandBuffer::invalidDependency(const SDependencyInfo<IGPUCommandBuffer::SOwnershipTransferBarrier>&) const;
#endif

}

#endif
