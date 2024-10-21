#ifndef _NBL_VIDEO_C_VULKAN_LOGICAL_DEVICE_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_LOGICAL_DEVICE_H_INCLUDED_


#include "nbl/core/containers/CMemoryPool.h"

#include <algorithm>

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/CVulkanCommon.h"
#include "nbl/video/CVulkanDeviceFunctionTable.h"
#include "nbl/video/CVulkanSwapchain.h"
#include "nbl/video/CVulkanQueue.h"
#include "nbl/video/CVulkanRenderpass.h"
#include "nbl/video/CVulkanImageView.h"
#include "nbl/video/CVulkanFramebuffer.h"
#include "nbl/video/CVulkanSemaphore.h"
#include "nbl/video/CVulkanShader.h"
#include "nbl/video/CVulkanCommandPool.h"
#include "nbl/video/CVulkanDescriptorSetLayout.h"
#include "nbl/video/CVulkanSampler.h"
#include "nbl/video/CVulkanPipelineLayout.h"
#include "nbl/video/CVulkanPipelineCache.h"
#include "nbl/video/CVulkanComputePipeline.h"
#include "nbl/video/CVulkanDescriptorPool.h"
#include "nbl/video/CVulkanDescriptorSet.h"
#include "nbl/video/CVulkanMemoryAllocation.h"
#include "nbl/video/CVulkanBuffer.h"
#include "nbl/video/CVulkanBufferView.h"
#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanDeferredOperation.h"
#include "nbl/video/CVulkanAccelerationStructure.h"
#include "nbl/video/CVulkanGraphicsPipeline.h"


namespace nbl::video
{

class CVulkanCommandBuffer;

class CVulkanLogicalDevice final : public ILogicalDevice
{
    public:
        // in the future we'll make proper Vulkan allocators and RAII free functions to pass into Vulkan API calls
        using memory_pool_mt_t = core::CMemoryPool<core::PoolAddressAllocator<uint32_t>,core::default_aligned_allocator,true,uint32_t>;
        
        CVulkanLogicalDevice(core::smart_refctd_ptr<const IAPIConnection>&& api, renderdoc_api_t* const rdoc, const IPhysicalDevice* const physicalDevice, const VkDevice vkdev, const SCreationParams& params);

        // sync stuff
        core::smart_refctd_ptr<ISemaphore> createSemaphore(const uint64_t initialValue) override;
        ISemaphore::WAIT_RESULT waitForSemaphores(const std::span<const ISemaphore::SWaitInfo> infos, const bool waitAll, const uint64_t timeout) override;
            
        core::smart_refctd_ptr<IEvent> createEvent(const IEvent::CREATE_FLAGS flags) override;
              
        core::smart_refctd_ptr<IDeferredOperation> createDeferredOperation() override;

        // memory  stuff
        SAllocation allocate(const SAllocateInfo& info) override;

        // descriptor creation
        core::smart_refctd_ptr<IGPUSampler> createSampler(const IGPUSampler::SParams& _params) override;
        
        inline core::smart_refctd_ptr<IGPUPipelineCache> createPipelineCache(const std::span<const uint8_t> initialData, const bool notThreadsafe=false) override
        {
            VkPipelineCacheCreateInfo createInfo = { VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,nullptr };
            createInfo.flags = notThreadsafe ? VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT : 0;
            createInfo.initialDataSize = initialData.size();
            createInfo.pInitialData = initialData.data();
            VkPipelineCache vk_pipelineCache;
            if (m_devf.vk.vkCreatePipelineCache(m_vkdev, &createInfo, nullptr, &vk_pipelineCache) == VK_SUCCESS)
                return core::make_smart_refctd_ptr<CVulkanPipelineCache>(core::smart_refctd_ptr<CVulkanLogicalDevice>(this), vk_pipelineCache);
            return nullptr;
        }

        inline memory_pool_mt_t& getMemoryPoolForDeferredOperations()
        {
            return m_deferred_op_mempool;
        }

        const CVulkanDeviceFunctionTable* getFunctionTable() const { return &m_devf; }

        inline const void* getNativeHandle() const {return &m_vkdev;}
        VkDevice getInternalObject() const {return m_vkdev;}

    private:
        inline ~CVulkanLogicalDevice()
        {
            m_devf.vk.vkDestroyDescriptorSetLayout(m_vkdev,m_dummyDSLayout,nullptr);
            m_devf.vk.vkDestroyDevice(m_vkdev,nullptr);
        }

        // sync stuff
        inline IQueue::RESULT waitIdle_impl() const override
        {
            return CVulkanQueue::getResultFrom(m_devf.vk.vkDeviceWaitIdle(m_vkdev));
        }
        
        // memory  stuff
        bool flushMappedMemoryRanges_impl(const std::span<const MappedMemoryRange> ranges) override;
        bool invalidateMappedMemoryRanges_impl(const std::span<const MappedMemoryRange> ranges) override;

        // memory binding
        bool bindBufferMemory_impl(const uint32_t count, const SBindBufferMemoryInfo* pInfos) override;
        bool bindImageMemory_impl(const uint32_t count, const SBindImageMemoryInfo* pInfos) override;

        // descriptor creation
        core::smart_refctd_ptr<IGPUBuffer> createBuffer_impl(IGPUBuffer::SCreationParams&& creationParams) override;
        core::smart_refctd_ptr<IGPUBufferView> createBufferView_impl(const asset::SBufferRange<const IGPUBuffer>& underlying, const asset::E_FORMAT _fmt) override;
        core::smart_refctd_ptr<IGPUImage> createImage_impl(IGPUImage::SCreationParams&& params) override;
        core::smart_refctd_ptr<IGPUImageView> createImageView_impl(IGPUImageView::SCreationParams&& params) override;
        VkAccelerationStructureKHR createAccelerationStructure(const IGPUAccelerationStructure::SCreationParams& params, const VkAccelerationStructureTypeKHR type, const VkAccelerationStructureMotionInfoNV* motionInfo=nullptr);
        inline core::smart_refctd_ptr<IGPUBottomLevelAccelerationStructure> createBottomLevelAccelerationStructure_impl(IGPUAccelerationStructure::SCreationParams&& params) override
        {
            const auto vk_as = createAccelerationStructure(params,VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
            if (vk_as!=VK_NULL_HANDLE)
                return core::make_smart_refctd_ptr<CVulkanBottomLevelAccelerationStructure>(core::smart_refctd_ptr<const CVulkanLogicalDevice>(this),std::move(params),vk_as);
            return nullptr;
        }
        inline core::smart_refctd_ptr<IGPUTopLevelAccelerationStructure> createTopLevelAccelerationStructure_impl(IGPUTopLevelAccelerationStructure::SCreationParams&& params) override
        {
            VkAccelerationStructureMotionInfoNV motionInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MOTION_INFO_NV,nullptr,0 };
            motionInfo.maxInstances = params.maxInstanceCount;

            const bool hasMotionBit = params.flags.hasFlags(IGPUAccelerationStructure::SCreationParams::FLAGS::MOTION_BIT);
            const auto vk_as = createAccelerationStructure(params,VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,hasMotionBit ? (&motionInfo):nullptr);
            if (vk_as!=VK_NULL_HANDLE)
            {
                if (!hasMotionBit)
                    params.maxInstanceCount = getPhysicalDevice()->getLimits().maxAccelerationStructureInstanceCount;
                return core::make_smart_refctd_ptr<CVulkanTopLevelAccelerationStructure>(core::smart_refctd_ptr<const CVulkanLogicalDevice>(this),std::move(params),vk_as);
            }
            return nullptr;
        }

        // acceleration structure modifiers
        inline AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur,
            const std::span<const IGPUBottomLevelAccelerationStructure::AABBs<const IGPUBuffer>> geometries, const uint32_t* const pMaxPrimitiveCounts
        ) const override
        {
            return getAccelerationStructureBuildSizes_impl_impl_impl(flags,motionBlur,geometries,pMaxPrimitiveCounts);
        }
        inline AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur,
            const std::span<const IGPUBottomLevelAccelerationStructure::AABBs<const asset::ICPUBuffer>> geometries, const uint32_t* const pMaxPrimitiveCounts
        ) const override
        {
            return getAccelerationStructureBuildSizes_impl_impl_impl(flags,motionBlur,geometries,pMaxPrimitiveCounts);
        }
        inline AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur,
            const std::span<const IGPUBottomLevelAccelerationStructure::Triangles<const IGPUBuffer>> geometries, const uint32_t* const pMaxPrimitiveCounts
        ) const override
        {
            return getAccelerationStructureBuildSizes_impl_impl_impl(flags,motionBlur,geometries,pMaxPrimitiveCounts);
        }
        inline AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur,
            const std::span<const IGPUBottomLevelAccelerationStructure::Triangles<const asset::ICPUBuffer>> geometries, const uint32_t* const pMaxPrimitiveCounts
        ) const override
        {
            return getAccelerationStructureBuildSizes_impl_impl_impl(flags,motionBlur,geometries,pMaxPrimitiveCounts);
        }
        template<class Geometry>
        inline AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl_impl_impl(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur,
            const std::span<const Geometry> geometries, const uint32_t* const pMaxPrimitiveCounts
        ) const
        {
            constexpr bool IsAABB = std::is_same_v<Geometry,IGPUBottomLevelAccelerationStructure::AABBs<const typename Geometry::buffer_t>>;

            core::vector<VkAccelerationStructureGeometryKHR> vk_geometries(geometries.size());
            core::vector<VkAccelerationStructureGeometryMotionTrianglesDataNV> vk_triangleMotions(IsAABB ? 0u:geometries.size());
            auto outTriangleMotions = vk_triangleMotions.data();
            for (auto i=0u; i<geometries.size(); i++)
            {
                if constexpr (IsAABB)
                    getVkASGeometryFrom<typename Geometry::buffer_t,true>(geometries[i],vk_geometries[i]);
                else
                    getVkASGeometryFrom<typename Geometry::buffer_t,true>(geometries[i],vk_geometries[i],outTriangleMotions);
            }

            return getAccelerationStructureBuildSizes_impl_impl(
                std::is_same_v<typename Geometry::buffer_t,asset::ICPUBuffer>,false,
                getVkASBuildFlagsFrom<IGPUBottomLevelAccelerationStructure>(flags,motionBlur),
                vk_geometries,pMaxPrimitiveCounts
            );
        }

        AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl(
            const bool hostBuild, const core::bitflag<IGPUTopLevelAccelerationStructure::BUILD_FLAGS> flags,
            const bool motionBlur, const uint32_t maxInstanceCount
        ) const override;
        AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl_impl(
            const bool hostBuild, const bool isTLAS, const VkBuildAccelerationStructureFlagsKHR flags,
            const std::span<const VkAccelerationStructureGeometryKHR> geometries, const uint32_t* const pMaxPrimitiveOrInstanceCounts
        ) const;

        static inline DEFERRABLE_RESULT getDeferrableResultFrom(const VkResult res)
        {
            switch (res)
            {
                case VK_OPERATION_DEFERRED_KHR:
                    return DEFERRABLE_RESULT::DEFERRED;
                case VK_OPERATION_NOT_DEFERRED_KHR:
                    return DEFERRABLE_RESULT::NOT_DEFERRED;
                case VK_SUCCESS:
                    assert(false); // should never happen if I read the spec correctly
                    break;
                default:
                    break;
            }
            return DEFERRABLE_RESULT::SOME_ERROR;
        }
        inline DEFERRABLE_RESULT buildAccelerationStructures_impl(
            IDeferredOperation* const deferredOperation, const std::span<const IGPUBottomLevelAccelerationStructure::HostBuildInfo> infos,
            const IGPUBottomLevelAccelerationStructure::BuildRangeInfo* const* const ppBuildRangeInfos, const uint32_t totalGeometryCount
        ) override
        {
            const auto infoCount = infos.size();
            core::vector<const VkAccelerationStructureBuildRangeInfoKHR*> vk_pBuildRangeInfos(infoCount);
            core::vector<VkAccelerationStructureBuildRangeInfoKHR> vk_buildRangeInfos(totalGeometryCount);
            core::vector<VkAccelerationStructureGeometryMotionTrianglesDataNV> vk_vertexMotions(m_enabledFeatures.rayTracingMotionBlur ? totalGeometryCount:0u);
            
            auto out_vk_infos = vk_buildRangeInfos.data();
            for (auto i=0u; i<infoCount; i++)
            {
                vk_pBuildRangeInfos[i] = out_vk_infos;
                getVkASBuildRangeInfos(infos[i].inputCount(),ppBuildRangeInfos[i],out_vk_infos);
            }
            return buildAccelerationStructures_impl_impl<IGPUBottomLevelAccelerationStructure>(deferredOperation,infos,vk_pBuildRangeInfos.data(),vk_vertexMotions.data());
        }
        inline DEFERRABLE_RESULT buildAccelerationStructures_impl(
            IDeferredOperation* const deferredOperation, const std::span<const IGPUTopLevelAccelerationStructure::HostBuildInfo> infos,
            const IGPUTopLevelAccelerationStructure::BuildRangeInfo* const pBuildRangeInfos, const uint32_t totalGeometryCount
        ) override
        {
            const auto infoCount = infos.size();
            core::vector<const VkAccelerationStructureBuildRangeInfoKHR*> vk_pBuildRangeInfos(infoCount);
            core::vector<VkAccelerationStructureBuildRangeInfoKHR> vk_buildRangeInfos(infoCount);

            for (auto i=0u; i<infoCount; i++)
            {
                vk_buildRangeInfos[i] = getVkASBuildRangeInfo(pBuildRangeInfos[i]);
                vk_pBuildRangeInfos[i] = vk_buildRangeInfos.data()+i;
            }
            return buildAccelerationStructures_impl_impl<IGPUTopLevelAccelerationStructure>(deferredOperation,infos,vk_pBuildRangeInfos.data());
        }
        template<class AccelerationStructure> requires std::is_base_of_v<IGPUAccelerationStructure,AccelerationStructure>
        inline DEFERRABLE_RESULT buildAccelerationStructures_impl_impl(
            IDeferredOperation* const deferredOperation, const std::span<const typename AccelerationStructure::HostBuildInfo> infos,
            const VkAccelerationStructureBuildRangeInfoKHR* const* const vk_ppBuildRangeInfos, VkAccelerationStructureGeometryMotionTrianglesDataNV* out_vk_vertexMotions=nullptr
        )
        {
            const auto infoCount = infos.size();
            core::vector<VkAccelerationStructureBuildGeometryInfoKHR> vk_buildGeomsInfos(infoCount);
            // I can actually rely on this pointer arithmetic because I allocated and populated the arrays myself
            const uint32_t totalGeometryCount = infos[infoCount-1].inputCount()+(vk_ppBuildRangeInfos[infoCount-1]- vk_ppBuildRangeInfos[0]);
            core::vector<VkAccelerationStructureGeometryKHR> vk_geometries(totalGeometryCount);
            
            auto out_vk_geoms = vk_geometries.data();
            for (auto i=0u; i<infoCount; i++)
                getVkASBuildGeometryInfo<typename AccelerationStructure::HostBuildInfo>(infos[i],out_vk_geoms,out_vk_vertexMotions);
            return getDeferrableResultFrom(m_devf.vk.vkBuildAccelerationStructuresKHR(m_vkdev,static_cast<CVulkanDeferredOperation*>(deferredOperation)->getInternalObject(),infoCount,vk_buildGeomsInfos.data(),vk_ppBuildRangeInfos));
        }
        bool writeAccelerationStructuresProperties_impl(const std::span<const IGPUAccelerationStructure* const> accelerationStructures, const IQueryPool::TYPE type, size_t* data, const size_t stride) override;
        DEFERRABLE_RESULT copyAccelerationStructure_impl(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::CopyInfo& copyInfo) override;
        DEFERRABLE_RESULT copyAccelerationStructureToMemory_impl(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::HostCopyToMemoryInfo& copyInfo) override;
        DEFERRABLE_RESULT copyAccelerationStructureFromMemory_impl(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::HostCopyFromMemoryInfo& copyInfo) override;

        // shaders
        core::smart_refctd_ptr<IGPUShader> createShader_impl(const asset::ICPUShader* spirvShader) override;

        // layouts
        core::smart_refctd_ptr<IGPUDescriptorSetLayout> createDescriptorSetLayout_impl(const std::span<const IGPUDescriptorSetLayout::SBinding> bindings, const uint32_t maxSamplersCount) override;
        core::smart_refctd_ptr<IGPUPipelineLayout> createPipelineLayout_impl(
            const std::span<const asset::SPushConstantRange> pcRanges,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3
        ) override;

        // descriptor sets
        core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool_impl(const IDescriptorPool::SCreateInfo& createInfo) override;
        void updateDescriptorSets_impl(const SUpdateDescriptorSetsParams& params) override;
        void nullifyDescriptors_impl(const SDropDescriptorSetsParams& params) override;

        // renderpasses and framebuffers
        core::smart_refctd_ptr<IGPURenderpass> createRenderpass_impl(const IGPURenderpass::SCreationParams& params, IGPURenderpass::SCreationParamValidationResult&& validation) override;
        core::smart_refctd_ptr<IGPUFramebuffer> createFramebuffer_impl(IGPUFramebuffer::SCreationParams&& params) override;

        // pipelines
        void createComputePipelines_impl(
            IGPUPipelineCache* const pipelineCache,
            const std::span<const IGPUComputePipeline::SCreationParams> createInfos,
            core::smart_refctd_ptr<IGPUComputePipeline>* const output,
            const IGPUComputePipeline::SCreationParams::SSpecializationValidationResult& validation
        ) override;
        void createGraphicsPipelines_impl(
            IGPUPipelineCache* const pipelineCache,
            const std::span<const IGPUGraphicsPipeline::SCreationParams> params,
            core::smart_refctd_ptr<IGPUGraphicsPipeline>* const output,
            const IGPUGraphicsPipeline::SCreationParams::SSpecializationValidationResult& validation
        ) override;

        // queries
        core::smart_refctd_ptr<IQueryPool> createQueryPool_impl(const IQueryPool::SCreationParams& params) override;
        bool getQueryPoolResults_impl(const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount, void* const pData, const size_t stride, const core::bitflag<IQueryPool::RESULTS_FLAGS> flags) override;

        // command buffers
        core::smart_refctd_ptr<IGPUCommandPool> createCommandPool_impl(const uint32_t familyIx, const core::bitflag<IGPUCommandPool::CREATE_FLAGS> flags) override;


        const VkDevice m_vkdev;
        CVulkanDeviceFunctionTable m_devf;
    
        constexpr static inline uint32_t NODES_PER_BLOCK_DEFERRED_OP = 4096u;
        constexpr static inline uint32_t MAX_BLOCK_COUNT_DEFERRED_OP = 256u;
        memory_pool_mt_t m_deferred_op_mempool;

        VkDescriptorSetLayout m_dummyDSLayout;
};

}

#endif