#include "nbl/video/CVulkanLogicalDevice.h"

#include "nbl/video/CThreadSafeQueueAdapter.h"
#include "nbl/video/surface/CSurfaceVulkan.h"

#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanQueryPool.h"
#include "nbl/video/CVulkanCommandBuffer.h"


using namespace nbl;
using namespace nbl::video;



CVulkanLogicalDevice::CVulkanLogicalDevice(core::smart_refctd_ptr<const IAPIConnection>&& api, renderdoc_api_t* const rdoc, const IPhysicalDevice* const physicalDevice, const VkDevice vkdev, const VkInstance vkinst, const SCreationParams& params)
    : ILogicalDevice(std::move(api),physicalDevice,params), m_vkdev(vkdev), m_devf(vkdev), m_deferred_op_mempool(NODES_PER_BLOCK_DEFERRED_OP*sizeof(CVulkanDeferredOperation), 1u, MAX_BLOCK_COUNT_DEFERRED_OP, static_cast<uint32_t>(sizeof(CVulkanDeferredOperation)))
{
    // create actual queue objects
    for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
    {
        const auto& qci = params.queueParams[i];
        const uint32_t famIx = qci.familyIndex;
        const uint32_t offset = m_queueFamilyInfos->operator[](famIx).first;
        const auto flags = qci.flags;
                    
        for (uint32_t j = 0u; j < qci.count; ++j)
        {
            const float priority = qci.priorities[j];
                        
            VkQueue q;
            VkDeviceQueueInfo2 vk_info = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,nullptr };
            vk_info.queueFamilyIndex = famIx;
            vk_info.queueIndex = j;
            vk_info.flags = 0; // we don't do protected queues yet
            m_devf.vk.vkGetDeviceQueue(m_vkdev, famIx, j, &q);
                        
            const uint32_t ix = offset + j;
            (*m_queues)[ix] = new CThreadSafeQueueAdapter(this,std::make_unique<CVulkanQueue>(this,rdoc,vkinst,q,famIx,flags,priority));
        }
    }
        
    std::ostringstream pool;
    bool runningInRenderdoc = (rdoc != nullptr);
    addCommonShaderDefines(pool,runningInRenderdoc);
    finalizeShaderDefinePool(std::move(pool));

    m_dummyDSLayout = createDescriptorSetLayout({nullptr,nullptr});
}


core::smart_refctd_ptr<ISemaphore> CVulkanLogicalDevice::createSemaphore(const uint64_t initialValue)
{
    VkSemaphoreTypeCreateInfoKHR type = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR };
    type.pNext = nullptr; // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkExportSemaphoreCreateInfo, VkExportSemaphoreWin32HandleInfoKHR
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR;
    type.initialValue = initialValue;

    VkSemaphoreCreateInfo createInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,&type };
    createInfo.flags = static_cast<VkSemaphoreCreateFlags>(0); // flags must be 0

    VkSemaphore semaphore;
    if (m_devf.vk.vkCreateSemaphore(m_vkdev,&createInfo,nullptr,&semaphore)==VK_SUCCESS)
        return core::make_smart_refctd_ptr<CVulkanSemaphore>(core::smart_refctd_ptr<const CVulkanLogicalDevice>(this),semaphore);
    else
        return nullptr;
}
auto CVulkanLogicalDevice::waitForSemaphores(const uint32_t count, const SSemaphoreWaitInfo* const infos, const bool waitAll, const uint64_t timeout) -> WAIT_RESULT
{
    core::vector<VkSemaphore> semaphores(count);
    core::vector<uint64_t> values(count);
    for (auto i=0u; i<count; i++)
    {
        auto sema = IBackendObject::device_compatibility_cast<CVulkanSemaphore*>(infos[i].semaphore,this);
        if (!sema)
            WAIT_RESULT::_ERROR;
        semaphores[i] = sema->getInternalObject();
        values[i] = infos[i].value;
    }

    VkSemaphoreWaitInfoKHR waitInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR,nullptr };
    waitInfo.flags = waitAll ? 0:VK_SEMAPHORE_WAIT_ANY_BIT_KHR;
    waitInfo.semaphoreCount = count;
    waitInfo.pSemaphores = semaphores.data();
    waitInfo.pValues = values.data();
    switch (m_devf.vk.vkWaitSemaphoresKHR(m_vkdev,&waitInfo,timeout))
    {
        case VK_SUCCESS:
            return WAIT_RESULT::SUCCESS;
        case VK_TIMEOUT:
            return WAIT_RESULT::TIMEOUT;
        case VK_ERROR_DEVICE_LOST:
            return WAIT_RESULT::DEVICE_LOST;
        default:
            break;
    }
    return WAIT_RESULT::_ERROR;
}

core::smart_refctd_ptr<IEvent> CVulkanLogicalDevice::createEvent(const IEvent::CREATE_FLAGS flags)
{
    VkEventCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_EVENT_CREATE_INFO };
    vk_createInfo.pNext = nullptr;
    vk_createInfo.flags = static_cast<VkEventCreateFlags>(flags);

    VkEvent vk_event;
    if (m_devf.vk.vkCreateEvent(m_vkdev,&vk_createInfo,nullptr,&vk_event)==VK_SUCCESS)
        return core::make_smart_refctd_ptr<CVulkanEvent>(core::smart_refctd_ptr<const CVulkanLogicalDevice>(this), flags, vk_event);
    else
        return nullptr;
}
              
core::smart_refctd_ptr<IDeferredOperation> CVulkanLogicalDevice::createDeferredOperation()
{
    VkDeferredOperationKHR vk_deferredOp = VK_NULL_HANDLE;
    const VkResult vk_res = m_devf.vk.vkCreateDeferredOperationKHR(m_vkdev, nullptr, &vk_deferredOp);
    if(vk_res!=VK_SUCCESS || vk_deferredOp==VK_NULL_HANDLE)
        return nullptr;

    void* memory = m_deferred_op_mempool.allocate(sizeof(CVulkanDeferredOperation),alignof(CVulkanDeferredOperation));
    if (!memory)
        return nullptr;

    new (memory) CVulkanDeferredOperation(core::smart_refctd_ptr<const CVulkanLogicalDevice>(this),vk_deferredOp);
    return core::smart_refctd_ptr<CVulkanDeferredOperation>(reinterpret_cast<CVulkanDeferredOperation*>(memory),core::dont_grab);
}


IDeviceMemoryAllocator::SAllocation CVulkanLogicalDevice::allocate(const SAllocateInfo& info)
{
    IDeviceMemoryAllocator::SAllocation ret = {};
    if (info.memoryTypeIndex>=m_physicalDevice->getMemoryProperties().memoryTypeCount)
        return ret;

    const core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags(info.flags);
    VkMemoryAllocateFlagsInfo vk_allocateFlagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr };
    {
        if (allocateFlags.hasFlags(IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT))
            vk_allocateFlagsInfo.flags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        vk_allocateFlagsInfo.deviceMask = 0u; // unused: for now
    }
    VkMemoryDedicatedAllocateInfo vk_dedicatedInfo = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, nullptr};
    if(info.dedication)
    {
        // VK_KHR_dedicated_allocation is in core 1.1, no querying for support needed
        static_assert(MinimumVulkanApiVersion >= VK_MAKE_API_VERSION(0,1,1,0));
        vk_allocateFlagsInfo.pNext = &vk_dedicatedInfo;
        switch (info.dedication->getObjectType())
        {
            case IDeviceMemoryBacked::EOT_BUFFER:
                vk_dedicatedInfo.buffer = static_cast<CVulkanBuffer*>(info.dedication)->getInternalObject();
                break;
            case IDeviceMemoryBacked::EOT_IMAGE:
                vk_dedicatedInfo.image = static_cast<CVulkanImage*>(info.dedication)->getInternalObject();
                break;
            default:
                assert(false);
                return ret;
                break;
        }
    }
    VkMemoryAllocateInfo vk_allocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &vk_allocateFlagsInfo};
    vk_allocateInfo.allocationSize = info.size;
    vk_allocateInfo.memoryTypeIndex = info.memoryTypeIndex;

    VkDeviceMemory vk_deviceMemory;
    auto vk_res = m_devf.vk.vkAllocateMemory(m_vkdev, &vk_allocateInfo, nullptr, &vk_deviceMemory);
    if (vk_res!=VK_SUCCESS)
        return ret;

    // automatically allocation goes out of scope and frees itself if no success later on
    const auto memoryPropertyFlags = m_physicalDevice->getMemoryProperties().memoryTypes[info.memoryTypeIndex].propertyFlags;
    ret.memory = core::make_smart_refctd_ptr<CVulkanMemoryAllocation>(this,info.size,info.dedication,vk_deviceMemory,allocateFlags,memoryPropertyFlags);
    ret.offset = 0ull; // LogicalDevice doesn't suballocate, so offset is always 0, if you want to suballocate, write/use an allocator
    if(info.dedication)
    {
        bool dedicationSuccess = false;
        switch (info.dedication->getObjectType())
        {
            case IDeviceMemoryBacked::EOT_BUFFER:
            {
                SBindBufferMemoryInfo bindBufferInfo = {};
                bindBufferInfo.buffer = static_cast<IGPUBuffer*>(info.dedication);
                bindBufferInfo.binding.memory = ret.memory.get();
                bindBufferInfo.binding.offset = ret.offset;
                dedicationSuccess = bindBufferMemory(1u,&bindBufferInfo);
            }
                break;
            case IDeviceMemoryBacked::EOT_IMAGE:
            {
                SBindImageMemoryInfo bindImageInfo = {};
                bindImageInfo.image = static_cast<IGPUImage*>(info.dedication);
                bindImageInfo.binding.memory = ret.memory.get();
                bindImageInfo.binding.offset = ret.offset;
                dedicationSuccess = bindImageMemory(1u,&bindImageInfo);
            }
                break;
        }
        if(!dedicationSuccess)
            ret = {};
    }
    return ret;
}

static inline void getVkMappedMemoryRanges(VkMappedMemoryRange* outRanges, const core::SRange<const ILogicalDevice::MappedMemoryRange>& ranges)
{
    for (auto& range : ranges)
    {
        VkMappedMemoryRange& vk_memoryRange = *(outRanges++);
        vk_memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        vk_memoryRange.pNext = nullptr; // pNext must be NULL
        vk_memoryRange.memory = static_cast<const CVulkanMemoryAllocation*>(range.memory)->getInternalObject();
        vk_memoryRange.offset = range.offset;
        vk_memoryRange.size = range.length;
    }
}
bool CVulkanLogicalDevice::flushMappedMemoryRanges_impl(const core::SRange<const MappedMemoryRange>& ranges)
{
    constexpr uint32_t MAX_MEMORY_RANGE_COUNT = 408u;
    if (ranges.size()>MAX_MEMORY_RANGE_COUNT)
        return false;

    VkMappedMemoryRange vk_memoryRanges[MAX_MEMORY_RANGE_COUNT];
    getVkMappedMemoryRanges(vk_memoryRanges,ranges);
    return m_devf.vk.vkFlushMappedMemoryRanges(m_vkdev,ranges.size(),vk_memoryRanges)==VK_SUCCESS;
}
bool CVulkanLogicalDevice::invalidateMappedMemoryRanges_impl(const core::SRange<const MappedMemoryRange>& ranges)
{
    constexpr uint32_t MAX_MEMORY_RANGE_COUNT = 408u;
    if (ranges.size()>MAX_MEMORY_RANGE_COUNT)
        return false;

    VkMappedMemoryRange vk_memoryRanges[MAX_MEMORY_RANGE_COUNT];
    getVkMappedMemoryRanges(vk_memoryRanges,ranges);
    m_devf.vk.vkInvalidateMappedMemoryRanges(m_vkdev,ranges.size(),vk_memoryRanges)==VK_SUCCESS;
}


bool CVulkanLogicalDevice::bindBufferMemory_impl(const uint32_t count, const SBindBufferMemoryInfo* pInfos)
{
    core::vector<VkBindBufferMemoryInfo> vk_infos(count,{VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,nullptr});
    for (uint32_t i=0u; i<count; ++i)
    {
        const auto& info = pInfos[i];
        vk_infos[i].buffer = static_cast<CVulkanBuffer*>(info.buffer)->getInternalObject();
        vk_infos[i].memory = static_cast<CVulkanMemoryAllocation*>(info.binding.memory)->getInternalObject();
        vk_infos[i].memoryOffset = info.binding.offset;
    }

    if (m_devf.vk.vkBindBufferMemory2(m_vkdev,vk_infos.size(),vk_infos.data())!=VK_SUCCESS)
    {
        m_logger.log("Call to `vkBindBufferMemory2` on Device %p failed!",system::ILogger::ELL_ERROR,this);
        return false;
    }
    
    for (uint32_t i=0u; i<count; ++i)
    {
        auto* vulkanBuffer = static_cast<CVulkanBuffer*>(pInfos[i].buffer);
        vulkanBuffer->setMemoryBinding(pInfos[i].binding);
        if (vulkanBuffer->getCreationParams().usage.hasFlags(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT))
        {
            VkBufferDeviceAddressInfoKHR info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR,nullptr};
            info.buffer = vulkanBuffer->getInternalObject();
            vulkanBuffer->setDeviceAddress(m_devf.vk.vkGetBufferDeviceAddressKHR(m_vkdev,&info));
        }
    }
    return true;
}
bool CVulkanLogicalDevice::bindImageMemory_impl(const uint32_t count, const SBindImageMemoryInfo* pInfos)
{
    core::vector<VkBindImageMemoryInfo> vk_infos(count,{VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,nullptr});
    for (uint32_t i=0u; i<count; ++i)
    {
        const auto& info = pInfos[i];
        vk_infos[i].image = static_cast<CVulkanImage*>(info.image)->getInternalObject();
        vk_infos[i].memory = static_cast<CVulkanMemoryAllocation*>(info.binding.memory)->getInternalObject();
        vk_infos[i].memoryOffset = info.binding.offset;
    }
    if (m_devf.vk.vkBindImageMemory2(m_vkdev,vk_infos.size(),vk_infos.data())!=VK_SUCCESS)
    {
        m_logger.log("Call to `vkBindImageMemory2` on Device %p failed!",system::ILogger::ELL_ERROR,this);
        return false;
    }
    
    for (uint32_t i=0u; i<count; ++i)
        static_cast<CVulkanImage*>(pInfos[i].image)->setMemoryBinding(pInfos[i].binding);
    return true;
}


core::smart_refctd_ptr<IGPUBuffer> CVulkanLogicalDevice::createBuffer_impl(IGPUBuffer::SCreationParams&& creationParams)
{
    VkBufferCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    // VkBufferDeviceAddressCreateInfoEXT, VkExternalMemoryBufferCreateInfo, VkVideoProfileKHR, or VkVideoProfilesKHR
    vk_createInfo.pNext = nullptr;
    vk_createInfo.flags = static_cast<VkBufferCreateFlags>(0u); // Nabla doesn't support any of these flags
    vk_createInfo.size = static_cast<VkDeviceSize>(creationParams.size);
    vk_createInfo.usage = getVkBufferUsageFlagsFromBufferUsageFlags(creationParams.usage);
    vk_createInfo.sharingMode = creationParams.isConcurrentSharing() ? VK_SHARING_MODE_CONCURRENT:VK_SHARING_MODE_EXCLUSIVE;
    vk_createInfo.queueFamilyIndexCount = creationParams.queueFamilyIndexCount;
    vk_createInfo.pQueueFamilyIndices = creationParams.queueFamilyIndices;

    VkBuffer vk_buffer;
    if (m_devf.vk.vkCreateBuffer(m_vkdev,&vk_createInfo,nullptr,&vk_buffer)!=VK_SUCCESS)
        return nullptr;
    return core::make_smart_refctd_ptr<CVulkanBuffer>(core::smart_refctd_ptr<CVulkanLogicalDevice>(this),std::move(creationParams),vk_buffer);
}

core::smart_refctd_ptr<IGPUBufferView> CVulkanLogicalDevice::createBufferView_impl(const asset::SBufferRange<const IGPUBuffer>& underlying, const asset::E_FORMAT _fmt)
{
    VkBufferViewCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO };
    vk_createInfo.pNext = nullptr; // pNext must be NULL
    vk_createInfo.flags = static_cast<VkBufferViewCreateFlags>(0); // flags must be 0
    vk_createInfo.buffer = static_cast<const CVulkanBuffer*>(underlying.buffer.get())->getInternalObject();
    vk_createInfo.format = getVkFormatFromFormat(_fmt);
    vk_createInfo.offset = underlying.offset;
    vk_createInfo.range = underlying.size;

    VkBufferView vk_bufferView;
    if (m_devf.vk.vkCreateBufferView(m_vkdev,&vk_createInfo,nullptr,&vk_bufferView)==VK_SUCCESS)
        return core::make_smart_refctd_ptr<CVulkanBufferView>(core::smart_refctd_ptr<const CVulkanLogicalDevice>(this),std::move(underlying),_fmt,vk_bufferView);
    return nullptr;
}

core::smart_refctd_ptr<IGPUImage> CVulkanLogicalDevice::createImage_impl(IGPUImage::SCreationParams&& params)
{
    VkImageStencilUsageCreateInfo vk_stencilUsage = { VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO, nullptr };
    vk_stencilUsage.stencilUsage = getVkImageUsageFlagsFromImageUsageFlags(params.actualStencilUsage().value,true);

    std::array<VkFormat,asset::E_FORMAT::EF_COUNT> vk_formatList;
    VkImageFormatListCreateInfo vk_formatListStruct = { VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, &vk_stencilUsage };
    vk_formatListStruct.viewFormatCount = 0u;
    // if only there existed a nice iterator that would let me iterate over set bits 64 faster
    if (params.viewFormats.any())
    for (auto fmt=0; fmt<vk_formatList.size(); fmt++)
    if (params.viewFormats.test(fmt))
        vk_formatList[vk_formatListStruct.viewFormatCount++] = getVkFormatFromFormat(static_cast<asset::E_FORMAT>(fmt));
    vk_formatListStruct.pViewFormats = vk_formatList.data();

    VkImageCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &vk_formatListStruct };
    vk_createInfo.flags = static_cast<VkImageCreateFlags>(params.flags.value);
    vk_createInfo.imageType = static_cast<VkImageType>(params.type);
    vk_createInfo.format = getVkFormatFromFormat(params.format);
    vk_createInfo.extent = { params.extent.width, params.extent.height, params.extent.depth };
    vk_createInfo.mipLevels = params.mipLevels;
    vk_createInfo.arrayLayers = params.arrayLayers;
    vk_createInfo.samples = static_cast<VkSampleCountFlagBits>(params.samples);
    vk_createInfo.tiling = static_cast<VkImageTiling>(params.tiling);
    vk_createInfo.usage = getVkImageUsageFlagsFromImageUsageFlags(params.usage.value,asset::isDepthOrStencilFormat(params.format));
    vk_createInfo.sharingMode = params.isConcurrentSharing() ? VK_SHARING_MODE_CONCURRENT:VK_SHARING_MODE_EXCLUSIVE;
    vk_createInfo.queueFamilyIndexCount = params.queueFamilyIndexCount;
    vk_createInfo.pQueueFamilyIndices = params.queueFamilyIndices;
    vk_createInfo.initialLayout = params.preinitialized ? VK_IMAGE_LAYOUT_PREINITIALIZED:VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage vk_image;
    if (m_devf.vk.vkCreateImage(m_vkdev,&vk_createInfo,nullptr,&vk_image)!=VK_SUCCESS)
        return nullptr;
    return core::make_smart_refctd_ptr<CVulkanImage>(core::smart_refctd_ptr<CVulkanLogicalDevice>(this),std::move(params),vk_image);
}

core::smart_refctd_ptr<IGPUImageView> CVulkanLogicalDevice::createImageView_impl(IGPUImageView::SCreationParams&& params)
{
    // pNext can be VkImageViewASTCDecodeModeEXT, VkSamplerYcbcrConversionInfo, VkVideoProfileKHR, or VkVideoProfilesKHR
    VkImageViewUsageCreateInfo vk_imageViewUsageInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,nullptr };
    vk_imageViewUsageInfo.usage = getVkImageUsageFlagsFromImageUsageFlags(params.actualUsages(),asset::isDepthOrStencilFormat(params.format));

    VkImageViewCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &vk_imageViewUsageInfo };
    vk_createInfo.flags = static_cast<VkImageViewCreateFlags>(params.flags);
    vk_createInfo.image = static_cast<const CVulkanImage*>(params.image.get())->getInternalObject();
    vk_createInfo.viewType = static_cast<VkImageViewType>(params.viewType);
    vk_createInfo.format = getVkFormatFromFormat(params.format);
    vk_createInfo.components.r = static_cast<VkComponentSwizzle>(params.components.r);
    vk_createInfo.components.g = static_cast<VkComponentSwizzle>(params.components.g);
    vk_createInfo.components.b = static_cast<VkComponentSwizzle>(params.components.b);
    vk_createInfo.components.a = static_cast<VkComponentSwizzle>(params.components.a);
    vk_createInfo.subresourceRange.aspectMask = static_cast<VkImageAspectFlags>(params.subresourceRange.aspectMask.value);
    vk_createInfo.subresourceRange.baseMipLevel = params.subresourceRange.baseMipLevel;
    vk_createInfo.subresourceRange.levelCount = params.subresourceRange.levelCount;
    vk_createInfo.subresourceRange.baseArrayLayer = params.subresourceRange.baseArrayLayer;
    vk_createInfo.subresourceRange.layerCount = params.subresourceRange.layerCount;

    VkImageView vk_imageView;
    if (m_devf.vk.vkCreateImageView(m_vkdev,&vk_createInfo,nullptr,&vk_imageView)==VK_SUCCESS)
        return core::make_smart_refctd_ptr<CVulkanImageView>(core::smart_refctd_ptr<CVulkanLogicalDevice>(this),std::move(params),vk_imageView);
}

core::smart_refctd_ptr<IGPUSampler> CVulkanLogicalDevice::createSampler(const IGPUSampler::SParams& _params)
{
    VkSamplerCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    vk_createInfo.pNext = nullptr; // VkSamplerCustomBorderColorCreateInfoEXT, VkSamplerReductionModeCreateInfo, or VkSamplerYcbcrConversionInfo
    vk_createInfo.flags = static_cast<VkSamplerCreateFlags>(0); // No flags supported yet
    assert(_params.MaxFilter <= asset::ISampler::ETF_LINEAR);
    vk_createInfo.magFilter = static_cast<VkFilter>(_params.MaxFilter);
    assert(_params.MinFilter <= asset::ISampler::ETF_LINEAR);
    vk_createInfo.minFilter = static_cast<VkFilter>(_params.MinFilter);
    vk_createInfo.mipmapMode = static_cast<VkSamplerMipmapMode>(_params.MipmapMode);
    vk_createInfo.addressModeU = getVkAddressModeFromTexClamp(static_cast<asset::ISampler::E_TEXTURE_CLAMP>(_params.TextureWrapU));
    vk_createInfo.addressModeV = getVkAddressModeFromTexClamp(static_cast<asset::ISampler::E_TEXTURE_CLAMP>(_params.TextureWrapV));
    vk_createInfo.addressModeW = getVkAddressModeFromTexClamp(static_cast<asset::ISampler::E_TEXTURE_CLAMP>(_params.TextureWrapW));
    vk_createInfo.mipLodBias = _params.LodBias;
    assert(_params.AnisotropicFilter <= m_physicalDevice->getLimits().maxSamplerAnisotropyLog2);
    vk_createInfo.maxAnisotropy = std::exp2(_params.AnisotropicFilter);
    vk_createInfo.anisotropyEnable = _params.AnisotropicFilter; // ROADMAP 2022
    vk_createInfo.compareEnable = _params.CompareEnable;
    vk_createInfo.compareOp = static_cast<VkCompareOp>(_params.CompareFunc);
    vk_createInfo.minLod = _params.MinLod;
    vk_createInfo.maxLod = _params.MaxLod;
    assert(_params.BorderColor < asset::ISampler::ETBC_COUNT);
    vk_createInfo.borderColor = static_cast<VkBorderColor>(_params.BorderColor);
    vk_createInfo.unnormalizedCoordinates = VK_FALSE;

    VkSampler vk_sampler;
    if (m_devf.vk.vkCreateSampler(m_vkdev,&vk_createInfo,nullptr,&vk_sampler)==VK_SUCCESS)
        return core::make_smart_refctd_ptr<CVulkanSampler>(core::smart_refctd_ptr<ILogicalDevice>(this),_params,vk_sampler);
    return nullptr;
}

VkAccelerationStructureKHR CVulkanLogicalDevice::createAccelerationStructure(const IGPUAccelerationStructure::SCreationParams& params, const VkAccelerationStructureTypeKHR type, const VkAccelerationStructureMotionInfoNV* motionInfo)
{
    VkAccelerationStructureCreateInfoKHR vasci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,motionInfo};
    vasci.createFlags = CVulkanAccelerationStructure::getVkASCreateFlagsFrom(params.flags.value);
    vasci.type = type;
    vasci.buffer = static_cast<const CVulkanBuffer*>(params.bufferRange.buffer.get())->getInternalObject();
    vasci.offset = params.bufferRange.offset;
    vasci.size = params.bufferRange.size;

    VkAccelerationStructureKHR vk_as;
    if (m_devf.vk.vkCreateAccelerationStructureKHR(m_vkdev,&vasci,nullptr,&vk_as)==VK_SUCCESS)
        return vk_as;
    return VK_NULL_HANDLE;
}


core::smart_refctd_ptr<IGPUShader> CVulkanLogicalDevice::createShader(core::smart_refctd_ptr<asset::ICPUShader>&& cpushader, const asset::ISPIRVOptimizer* optimizer)
{
    const char* entryPoint = "main"; // every compiler seems to be handicapped this way?
    const asset::IShader::E_SHADER_STAGE shaderStage = cpushader->getStage();

    const asset::ICPUBuffer* source = cpushader->getContent();

    core::smart_refctd_ptr<const asset::ICPUShader> spirvShader;

    if (cpushader->getContentType()==asset::ICPUShader::E_CONTENT_TYPE::ECT_SPIRV)
        spirvShader = cpushader;
    else
    {
        auto compiler = m_compilerSet->getShaderCompiler(cpushader->getContentType());

        asset::IShaderCompiler::SCompilerOptions commonCompileOptions = {};

        commonCompileOptions.preprocessorOptions.logger = (m_physicalDevice->getDebugCallback()) ? m_physicalDevice->getDebugCallback()->getLogger() : nullptr;
        commonCompileOptions.preprocessorOptions.includeFinder = compiler->getDefaultIncludeFinder(); // to resolve includes before compilation
        commonCompileOptions.preprocessorOptions.sourceIdentifier = cpushader->getFilepathHint().c_str();
        commonCompileOptions.preprocessorOptions.extraDefines = getExtraShaderDefines();

        commonCompileOptions.stage = shaderStage;
        commonCompileOptions.genDebugInfo = true;
        commonCompileOptions.spirvOptimizer = optimizer;
        commonCompileOptions.targetSpirvVersion = m_physicalDevice->getLimits().spirvVersion;

        if (cpushader->getContentType() == asset::ICPUShader::E_CONTENT_TYPE::ECT_HLSL)
        {
            // TODO: add specific HLSLCompiler::SOption params
            spirvShader = m_compilerSet->compileToSPIRV(cpushader.get(), commonCompileOptions);
        }
        else if (cpushader->getContentType() == asset::ICPUShader::E_CONTENT_TYPE::ECT_GLSL)
        {
            spirvShader = m_compilerSet->compileToSPIRV(cpushader.get(), commonCompileOptions);
        }
        else
            spirvShader = m_compilerSet->compileToSPIRV(cpushader.get(), commonCompileOptions);
    }

    if (!spirvShader || !spirvShader->getContent())
        return nullptr;

    auto spirv = spirvShader->getContent();

    VkShaderModuleCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    vk_createInfo.pNext = nullptr;
    vk_createInfo.flags = static_cast<VkShaderModuleCreateFlags>(0u); // reserved for future use by Vulkan
    vk_createInfo.codeSize = spirv->getSize();
    vk_createInfo.pCode = static_cast<const uint32_t*>(spirv->getPointer());
        
    VkShaderModule vk_shaderModule;
    if (m_devf.vk.vkCreateShaderModule(m_vkdev,&vk_createInfo,nullptr,&vk_shaderModule)==VK_SUCCESS)
        return core::make_smart_refctd_ptr<video::CVulkanShader>(core::smart_refctd_ptr<const CVulkanLogicalDevice>(this),spirvShader->getStage(),std::string(cpushader->getFilepathHint()),vk_shaderModule);
    return nullptr;
}


core::smart_refctd_ptr<IGPUDescriptorSetLayout> CVulkanLogicalDevice::createDescriptorSetLayout_impl(const core::SRange<const IGPUDescriptorSetLayout::SBinding>& bindings, const uint32_t maxSamplersCount)
{
    std::vector<VkSampler> vk_samplers;
    std::vector<VkDescriptorSetLayoutBinding> vk_dsLayoutBindings;
    vk_samplers.reserve(maxSamplersCount); // Reserve to avoid resizing and pointer change while iterating 
    vk_dsLayoutBindings.reserve(bindings.size());

    for (const auto& binding : bindings)
    {
        auto& vkDescSetLayoutBinding = vk_dsLayoutBindings.emplace_back();
        vkDescSetLayoutBinding.binding = binding.binding;
        vkDescSetLayoutBinding.descriptorType = getVkDescriptorTypeFromDescriptorType(binding.type);
        vkDescSetLayoutBinding.descriptorCount = binding.count;
        vkDescSetLayoutBinding.stageFlags = getVkShaderStageFlagsFromShaderStage(binding.stageFlags);
        vkDescSetLayoutBinding.pImmutableSamplers = nullptr;

        if (binding.type==asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER && binding.samplers && binding.count)
        {
            // If descriptorType is VK_DESCRIPTOR_TYPE_SAMPLER or VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, and descriptorCount is not 0 and pImmutableSamplers is not NULL:
            // pImmutableSamplers must be a valid pointer to an array of descriptorCount valid VkSampler handles.
            const uint32_t samplerOffset = vk_samplers.size();
            for (uint32_t i=0u; i<binding.count; ++i)
                vk_samplers.push_back(static_cast<const CVulkanSampler*>(binding.samplers[i].get())->getInternalObject());
            vkDescSetLayoutBinding.pImmutableSamplers = vk_samplers.data()+samplerOffset;
        }
    }

    VkDescriptorSetLayoutCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    vk_createInfo.pNext = nullptr; // pNext of interest:  VkDescriptorSetLayoutBindingFlagsCreateInfo
    vk_createInfo.flags = 0; // Todo(achal): I would need to create a IDescriptorSetLayout::SCreationParams for this
    vk_createInfo.bindingCount = vk_dsLayoutBindings.size();
    vk_createInfo.pBindings = vk_dsLayoutBindings.data();

    VkDescriptorSetLayout vk_dsLayout;
    if (m_devf.vk.vkCreateDescriptorSetLayout(m_vkdev,&vk_createInfo,nullptr,&vk_dsLayout)==VK_SUCCESS)
        return core::make_smart_refctd_ptr<CVulkanDescriptorSetLayout>(core::smart_refctd_ptr<const CVulkanLogicalDevice>(this),bindings,vk_dsLayout);
    return nullptr;
}

core::smart_refctd_ptr<IGPUPipelineLayout> CVulkanLogicalDevice::createPipelineLayout_impl(
    const core::SRange<const asset::SPushConstantRange>& pcRanges,
    core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout0,
    core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout1,
    core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout2,
    core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout3
)
{
    const core::smart_refctd_ptr<IGPUDescriptorSetLayout> tmp[] = { layout0, layout1, layout2, layout3 };

    VkDescriptorSetLayout vk_dsLayouts[asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT];
    uint32_t nonNullSetLayoutCount = ~0u;
    for (uint32_t i = 0u; i < asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
    {
        if (tmp[i])
            nonNullSetLayoutCount = i;
        vk_dsLayouts[i] = static_cast<const CVulkanDescriptorSetLayout*>((tmp[i] ? tmp[i]:m_dummyDSLayout).get())->getInternalObject();
    }
    nonNullSetLayoutCount++;

    VkPushConstantRange vk_pushConstantRanges[IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE];
    auto oit = vk_pushConstantRanges;
    for (const auto pcRange : pcRanges)
    {
        oit->stageFlags = getVkShaderStageFlagsFromShaderStage(pcRange.stageFlags);
        oit->offset = pcRange.offset;
        oit->size = pcRange.size;
        oit++;
    }

    VkPipelineLayoutCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,nullptr };
    vk_createInfo.flags = static_cast<VkPipelineLayoutCreateFlags>(0); // flags must be 0
    vk_createInfo.setLayoutCount = nonNullSetLayoutCount;
    vk_createInfo.pSetLayouts = vk_dsLayouts;
    vk_createInfo.pushConstantRangeCount = pcRanges.size();
    vk_createInfo.pPushConstantRanges = vk_pushConstantRanges;
                
    VkPipelineLayout vk_pipelineLayout;
    if (m_devf.vk.vkCreatePipelineLayout(m_vkdev,&vk_createInfo,nullptr,&vk_pipelineLayout)==VK_SUCCESS)
        return core::make_smart_refctd_ptr<CVulkanPipelineLayout>(core::smart_refctd_ptr<const CVulkanLogicalDevice>(this),pcRanges,std::move(layout0),std::move(layout1),std::move(layout2),std::move(layout3),vk_pipelineLayout);
    return nullptr;
}

            
core::smart_refctd_ptr<IDescriptorPool> CVulkanLogicalDevice::createDescriptorPool_impl(const IDescriptorPool::SCreateInfo& createInfo)
{
    uint32_t poolSizeCount = 0;
    VkDescriptorPoolSize poolSizes[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)];

    for (uint32_t t=0; t<static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
    {
        if (createInfo.maxDescriptorCount[t]==0)
            continue;

        auto& poolSize = poolSizes[poolSizeCount++];
        poolSize.type = getVkDescriptorTypeFromDescriptorType(static_cast<asset::IDescriptor::E_TYPE>(t));
        poolSize.descriptorCount = createInfo.maxDescriptorCount[t];
    }

    VkDescriptorPoolCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    vk_createInfo.pNext = nullptr; // no pNext of interest so far
    vk_createInfo.flags = static_cast<VkDescriptorPoolCreateFlags>(createInfo.flags.value);
    vk_createInfo.maxSets = createInfo.maxSets;
    vk_createInfo.poolSizeCount = poolSizeCount;
    vk_createInfo.pPoolSizes = poolSizes;

    VkDescriptorPool vk_descriptorPool;
    if (m_devf.vk.vkCreateDescriptorPool(m_vkdev,&vk_createInfo,nullptr,&vk_descriptorPool)==VK_SUCCESS)
        return core::make_smart_refctd_ptr<CVulkanDescriptorPool>(core::smart_refctd_ptr<CVulkanLogicalDevice>(this),std::move(createInfo),vk_descriptorPool);
    return nullptr;
}












bool CVulkanLogicalDevice::createCommandBuffers_impl(IGPUCommandPool* cmdPool, IGPUCommandBuffer::E_LEVEL level,
    uint32_t count, core::smart_refctd_ptr<IGPUCommandBuffer>* outCmdBufs)
{
    constexpr uint32_t MAX_COMMAND_BUFFER_COUNT = 1000u;

    if (cmdPool->getAPIType() != EAT_VULKAN)
        return false;

    auto vulkanCommandPool = IBackendObject::device_compatibility_cast<CVulkanCommandPool*>(cmdPool, this)->getInternalObject();

    assert(count <= MAX_COMMAND_BUFFER_COUNT);
    VkCommandBuffer vk_commandBuffers[MAX_COMMAND_BUFFER_COUNT];

    VkCommandBufferAllocateInfo vk_allocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    vk_allocateInfo.pNext = nullptr; // this must be NULL
    vk_allocateInfo.commandPool = vulkanCommandPool;
    vk_allocateInfo.level = static_cast<VkCommandBufferLevel>(level);
    vk_allocateInfo.commandBufferCount = count;

    if (m_devf.vk.vkAllocateCommandBuffers(m_vkdev, &vk_allocateInfo, vk_commandBuffers) == VK_SUCCESS)
    {
        for (uint32_t i = 0u; i < count; ++i)
        {
            const auto* debugCb = m_physicalDevice->getDebugCallback();

            outCmdBufs[i] = core::make_smart_refctd_ptr<CVulkanCommandBuffer>(
                core::smart_refctd_ptr<ILogicalDevice>(this), level, vk_commandBuffers[i],
                core::smart_refctd_ptr<IGPUCommandPool>(cmdPool),
                debugCb ? core::smart_refctd_ptr<system::ILogger>(debugCb->getLogger()) : nullptr);
        }

        return true;
    }
    else
    {
        return false;
    }
}


core::smart_refctd_ptr<IGPUGraphicsPipeline> CVulkanLogicalDevice::createGraphicsPipeline_impl(
    IGPUPipelineCache* pipelineCache,
    IGPUGraphicsPipeline::SCreationParams&& params)
{
    core::smart_refctd_ptr<IGPUGraphicsPipeline> result;
    if (createGraphicsPipelines_impl(pipelineCache, { &params, &params + 1 }, &result))
        return result;
    else
        return nullptr;
}

bool CVulkanLogicalDevice::createGraphicsPipelines_impl(
    IGPUPipelineCache* pipelineCache,
    core::SRange<const IGPUGraphicsPipeline::SCreationParams> params,
    core::smart_refctd_ptr<IGPUGraphicsPipeline>* output)
{
    IGPUGraphicsPipeline::SCreationParams* creationParams = const_cast<IGPUGraphicsPipeline::SCreationParams*>(params.begin());

    VkPipelineCache vk_pipelineCache = VK_NULL_HANDLE;
    if (pipelineCache && pipelineCache->getAPIType() == EAT_VULKAN)
        vk_pipelineCache = IBackendObject::device_compatibility_cast<const CVulkanPipelineCache*>(pipelineCache, this)->getInternalObject();

    // Shader stages
    uint32_t shaderStageCount_total = 0u;
    core::vector<VkPipelineShaderStageCreateInfo> vk_shaderStages(params.size() * IGPURenderpassIndependentPipeline::GRAPHICS_SHADER_STAGE_COUNT);
    uint32_t specInfoCount_total = 0u;
    core::vector<VkSpecializationInfo> vk_specInfos(vk_shaderStages.size());
    constexpr uint32_t MAX_MAP_ENTRIES_PER_SHADER = 100u;
    uint32_t mapEntryCount_total = 0u;
    core::vector<VkSpecializationMapEntry> vk_mapEntries(vk_specInfos.size()*MAX_MAP_ENTRIES_PER_SHADER);

    // Vertex input
    uint32_t vertexBindingDescriptionCount_total = 0u;
    core::vector<VkVertexInputBindingDescription> vk_vertexBindingDescriptions(params.size() * asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT);
    uint32_t vertexAttribDescriptionCount_total = 0u;
    core::vector<VkVertexInputAttributeDescription> vk_vertexAttribDescriptions(params.size() * asset::SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT);
    core::vector<VkPipelineVertexInputStateCreateInfo> vk_vertexInputStates(params.size());

    // Input Assembly
    core::vector<VkPipelineInputAssemblyStateCreateInfo> vk_inputAssemblyStates(params.size());

    core::vector<VkPipelineViewportStateCreateInfo> vk_viewportStates(params.size());

    core::vector<VkPipelineRasterizationStateCreateInfo> vk_rasterizationStates(params.size());

    core::vector<VkPipelineMultisampleStateCreateInfo> vk_multisampleStates(params.size());

    core::vector<VkStencilOpState> vk_stencilFrontStates(params.size());
    core::vector<VkStencilOpState> vk_stencilBackStates(params.size());
    core::vector<VkPipelineDepthStencilStateCreateInfo> vk_depthStencilStates(params.size());

    uint32_t colorBlendAttachmentCount_total = 0u;
    core::vector<VkPipelineColorBlendAttachmentState> vk_colorBlendAttachmentStates(params.size() * asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT);
    core::vector<VkPipelineColorBlendStateCreateInfo> vk_colorBlendStates(params.size());

    constexpr uint32_t DYNAMIC_STATE_COUNT = 2u;
    VkDynamicState vk_dynamicStates[DYNAMIC_STATE_COUNT] = { VK_DYNAMIC_STATE_VIEWPORT , VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo vk_dynamicStateCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    vk_dynamicStateCreateInfo.pNext = nullptr;
    vk_dynamicStateCreateInfo.flags = 0u;
    vk_dynamicStateCreateInfo.dynamicStateCount = DYNAMIC_STATE_COUNT;
    vk_dynamicStateCreateInfo.pDynamicStates = vk_dynamicStates;

    core::vector<VkGraphicsPipelineCreateInfo> vk_createInfos(params.size());
    for (size_t i = 0ull; i < params.size(); ++i)
    {
        const auto& rpIndie = creationParams[i].renderpassIndependent;

        vk_createInfos[i].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        vk_createInfos[i].pNext = nullptr;
        vk_createInfos[i].flags = static_cast<VkPipelineCreateFlags>(creationParams[i].createFlags.value);

        uint32_t shaderStageCount = 0u;
        for (uint32_t ss = 0u; ss < IGPURenderpassIndependentPipeline::GRAPHICS_SHADER_STAGE_COUNT; ++ss)
        {
            const IGPUSpecializedShader* shader = rpIndie->getShaderAtIndex(ss);
            if (!shader || shader->getAPIType() != EAT_VULKAN)
                continue;

            const auto* vulkanSpecShader = IBackendObject::device_compatibility_cast<const CVulkanSpecializedShader*>(shader, this);

            auto& vk_shaderStage = vk_shaderStages[shaderStageCount_total + shaderStageCount];

            vk_shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            vk_shaderStage.pNext = nullptr;
            vk_shaderStage.flags = 0u;
            vk_shaderStage.stage = static_cast<VkShaderStageFlagBits>(shader->getStage());
            vk_shaderStage.module = vulkanSpecShader->getInternalObject();
            vk_shaderStage.pName = "main";

            const auto& shaderSpecInfo = vulkanSpecShader->getSpecInfo();

            if (shaderSpecInfo.m_backingBuffer && shaderSpecInfo.m_entries)
            {
                for (uint32_t me = 0u; me < shaderSpecInfo.m_entries->size(); ++me)
                {
                    const auto entry = shaderSpecInfo.m_entries->begin() + me;

                    vk_mapEntries[mapEntryCount_total + me].constantID = entry->specConstID;
                    vk_mapEntries[mapEntryCount_total + me].offset = entry->offset;
                    vk_mapEntries[mapEntryCount_total + me].size = entry->size;
                }

                vk_specInfos[specInfoCount_total].mapEntryCount = static_cast<uint32_t>(shaderSpecInfo.m_entries->size());
                vk_specInfos[specInfoCount_total].pMapEntries = vk_mapEntries.data() + mapEntryCount_total;
                mapEntryCount_total += vk_specInfos[specInfoCount_total].mapEntryCount;
                vk_specInfos[specInfoCount_total].dataSize = shaderSpecInfo.m_backingBuffer->getSize();
                vk_specInfos[specInfoCount_total].pData = shaderSpecInfo.m_backingBuffer->getPointer();

                vk_shaderStage.pSpecializationInfo = vk_specInfos.data() + specInfoCount_total++;
            }
            else
            {
                vk_shaderStage.pSpecializationInfo = nullptr;
            }

            ++shaderStageCount;
        }
        vk_createInfos[i].stageCount = shaderStageCount;
        vk_createInfos[i].pStages = vk_shaderStages.data() + shaderStageCount_total;
        shaderStageCount_total += shaderStageCount;

        // Vertex Input
        {
            vk_vertexInputStates[i].sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vk_vertexInputStates[i].pNext = nullptr;
            vk_vertexInputStates[i].flags = 0u;

            const auto& vertexInputParams = rpIndie->getVertexInputParams();

            // Fill up vertex binding descriptions
            uint32_t offset = vertexBindingDescriptionCount_total;
            uint32_t vertexBindingDescriptionCount = 0u;

            for (uint32_t b = 0u; b < asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++b)
            {
                if (vertexInputParams.enabledBindingFlags & (1 << b))
                {
                    auto& bndDesc = vk_vertexBindingDescriptions[offset + vertexBindingDescriptionCount++];

                    bndDesc.binding = b;
                    bndDesc.stride = vertexInputParams.bindings[b].stride;
                    bndDesc.inputRate = static_cast<VkVertexInputRate>(vertexInputParams.bindings[b].inputRate);
                }
            }
            vk_vertexInputStates[i].vertexBindingDescriptionCount = vertexBindingDescriptionCount;
            vk_vertexInputStates[i].pVertexBindingDescriptions = vk_vertexBindingDescriptions.data() + offset;
            vertexBindingDescriptionCount_total += vertexBindingDescriptionCount;

            // Fill up vertex attribute descriptions
            offset = vertexAttribDescriptionCount_total;
            uint32_t vertexAttribDescriptionCount = 0u;

            for (uint32_t l = 0u; l < asset::SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; ++l)
            {
                if (vertexInputParams.enabledAttribFlags & (1 << l))
                {
                    auto& attribDesc = vk_vertexAttribDescriptions[offset + vertexAttribDescriptionCount++];

                    attribDesc.location = l;
                    attribDesc.binding = vertexInputParams.attributes[l].binding;
                    attribDesc.format = getVkFormatFromFormat(static_cast<asset::E_FORMAT>(vertexInputParams.attributes[l].format));
                    attribDesc.offset = vertexInputParams.attributes[l].relativeOffset;
                }
            }
            vk_vertexInputStates[i].vertexAttributeDescriptionCount = vertexAttribDescriptionCount;
            vk_vertexInputStates[i].pVertexAttributeDescriptions = vk_vertexAttribDescriptions.data() + offset;
            vertexAttribDescriptionCount_total += vertexAttribDescriptionCount;
        }
        vk_createInfos[i].pVertexInputState = &vk_vertexInputStates[i];

        // Input Assembly
        {
            const auto& primAssParams = rpIndie->getPrimitiveAssemblyParams();

            vk_inputAssemblyStates[i].sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            vk_inputAssemblyStates[i].pNext = nullptr;
            vk_inputAssemblyStates[i].flags = 0u; // reserved for future use by Vulkan
            vk_inputAssemblyStates[i].topology = static_cast<VkPrimitiveTopology>(primAssParams.primitiveType);
            vk_inputAssemblyStates[i].primitiveRestartEnable = primAssParams.primitiveRestartEnable;
        }
        vk_createInfos[i].pInputAssemblyState = &vk_inputAssemblyStates[i];

        // Tesselation
        vk_createInfos[i].pTessellationState = nullptr;

        // Viewport State
        {
            const uint32_t viewportCount = rpIndie->getRasterizationParams().viewportCount;

            vk_viewportStates[i].sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vk_viewportStates[i].pNext = nullptr;
            vk_viewportStates[i].flags = 0u;
            vk_viewportStates[i].viewportCount = viewportCount;
            vk_viewportStates[i].pViewports = nullptr; // ignored
            vk_viewportStates[i].scissorCount = viewportCount; // must be identical to viewport count unless VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT_EXT or VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT_EXT are used
            vk_viewportStates[i].pScissors = nullptr; // ignored
        }
        vk_createInfos[i].pViewportState = &vk_viewportStates[i];

        // Rasterization
        {
            const auto& rasterizationParams = rpIndie->getRasterizationParams();

            vk_rasterizationStates[i].sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            vk_rasterizationStates[i].pNext = nullptr;
            vk_rasterizationStates[i].flags = 0u;
            vk_rasterizationStates[i].depthClampEnable = rasterizationParams.depthClampEnable;
            vk_rasterizationStates[i].rasterizerDiscardEnable = rasterizationParams.rasterizerDiscard;
            vk_rasterizationStates[i].polygonMode = static_cast<VkPolygonMode>(rasterizationParams.polygonMode);
            vk_rasterizationStates[i].cullMode = static_cast<VkCullModeFlags>(rasterizationParams.faceCullingMode);
            vk_rasterizationStates[i].frontFace = rasterizationParams.frontFaceIsCCW ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
            vk_rasterizationStates[i].depthBiasEnable = rasterizationParams.depthBiasEnable;
            vk_rasterizationStates[i].depthBiasConstantFactor = rasterizationParams.depthBiasConstantFactor;
            vk_rasterizationStates[i].depthBiasClamp = 0.f;
            vk_rasterizationStates[i].depthBiasSlopeFactor = rasterizationParams.depthBiasSlopeFactor;
            vk_rasterizationStates[i].lineWidth = 1.f;
        }
        vk_createInfos[i].pRasterizationState = &vk_rasterizationStates[i];

        // Multisampling
        {
            const auto& rasterizationParams = rpIndie->getRasterizationParams();

            vk_multisampleStates[i].sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            vk_multisampleStates[i].pNext = nullptr;
            vk_multisampleStates[i].flags = 0u;
            vk_multisampleStates[i].rasterizationSamples = static_cast<VkSampleCountFlagBits>(creationParams[i].rasterizationSamples);
            vk_multisampleStates[i].sampleShadingEnable = rasterizationParams.sampleShadingEnable;
            vk_multisampleStates[i].minSampleShading = rasterizationParams.minSampleShading;
            vk_multisampleStates[i].pSampleMask = rasterizationParams.sampleMask;
            vk_multisampleStates[i].alphaToCoverageEnable = rasterizationParams.alphaToCoverageEnable;
            vk_multisampleStates[i].alphaToOneEnable = rasterizationParams.alphaToOneEnable;
        }
        vk_createInfos[i].pMultisampleState = &vk_multisampleStates[i];

        // Depth-stencil
        {
            const auto& rasterParams = rpIndie->getRasterizationParams();

            // Front stencil state
            vk_stencilFrontStates[i].failOp = static_cast<VkStencilOp>(rasterParams.frontStencilOps.failOp);
            vk_stencilFrontStates[i].passOp = static_cast<VkStencilOp>(rasterParams.frontStencilOps.passOp);
            vk_stencilFrontStates[i].depthFailOp = static_cast<VkStencilOp>(rasterParams.frontStencilOps.depthFailOp);
            vk_stencilFrontStates[i].compareOp = static_cast<VkCompareOp>(rasterParams.frontStencilOps.compareOp);
            vk_stencilFrontStates[i].compareMask = 0xFFFFFFFF;
            vk_stencilFrontStates[i].writeMask = rasterParams.frontStencilOps.writeMask;
            vk_stencilFrontStates[i].reference = rasterParams.frontStencilOps.reference;

            // Back stencil state
            vk_stencilBackStates[i].failOp = static_cast<VkStencilOp>(rasterParams.backStencilOps.failOp);
            vk_stencilBackStates[i].passOp = static_cast<VkStencilOp>(rasterParams.backStencilOps.passOp);
            vk_stencilBackStates[i].depthFailOp = static_cast<VkStencilOp>(rasterParams.backStencilOps.depthFailOp);
            vk_stencilBackStates[i].compareOp = static_cast<VkCompareOp>(rasterParams.backStencilOps.compareOp);
            vk_stencilBackStates[i].compareMask = 0xFFFFFFFF;
            vk_stencilBackStates[i].writeMask = rasterParams.backStencilOps.writeMask;
            vk_stencilBackStates[i].reference = rasterParams.backStencilOps.reference;
            
            vk_depthStencilStates[i].sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            vk_depthStencilStates[i].pNext = nullptr;
            vk_depthStencilStates[i].flags = static_cast<VkPipelineDepthStencilStateCreateFlags>(0u);
            vk_depthStencilStates[i].depthTestEnable = rasterParams.depthTestEnable;
            vk_depthStencilStates[i].depthWriteEnable = rasterParams.depthWriteEnable;
            vk_depthStencilStates[i].depthCompareOp = static_cast<VkCompareOp>(rasterParams.depthCompareOp);
            vk_depthStencilStates[i].depthBoundsTestEnable = rasterParams.depthBoundsTestEnable;
            vk_depthStencilStates[i].stencilTestEnable = rasterParams.stencilTestEnable;
            vk_depthStencilStates[i].front = vk_stencilFrontStates[i];
            vk_depthStencilStates[i].back = vk_stencilBackStates[i];
            vk_depthStencilStates[i].minDepthBounds = 0.f;
            vk_depthStencilStates[i].maxDepthBounds = 1.f;
        }
        vk_createInfos[i].pDepthStencilState = &vk_depthStencilStates[i];

        // Color blend
        {
            const auto& blendParams = rpIndie->getBlendParams();
            
            uint32_t offset = colorBlendAttachmentCount_total;

            assert(creationParams[i].subpassIx < creationParams[i].renderpass->getCreationParameters().subpassCount);
            auto subpassDescription = creationParams[i].renderpass->getCreationParameters().subpasses[creationParams[i].subpassIx];
            uint32_t colorBlendAttachmentCount = subpassDescription.colorAttachmentCount;

            for (uint32_t as = 0u; as < colorBlendAttachmentCount; ++as)
            {
                const auto& inBlendParams = blendParams.blendParams[as];
                auto& outBlendState = vk_colorBlendAttachmentStates[offset + as];

                outBlendState.blendEnable = inBlendParams.blendEnable;
                outBlendState.srcColorBlendFactor = getVkBlendFactorFromBlendFactor(static_cast<asset::E_BLEND_FACTOR>(inBlendParams.srcColorFactor));
                outBlendState.dstColorBlendFactor = getVkBlendFactorFromBlendFactor(static_cast<asset::E_BLEND_FACTOR>(inBlendParams.dstColorFactor));
                assert(inBlendParams.colorBlendOp <= asset::EBO_MAX);
                outBlendState.colorBlendOp = getVkBlendOpFromBlendOp(static_cast<asset::E_BLEND_OP>(inBlendParams.colorBlendOp));
                outBlendState.srcAlphaBlendFactor = getVkBlendFactorFromBlendFactor(static_cast<asset::E_BLEND_FACTOR>(inBlendParams.srcAlphaFactor));
                outBlendState.dstAlphaBlendFactor = getVkBlendFactorFromBlendFactor(static_cast<asset::E_BLEND_FACTOR>(inBlendParams.dstAlphaFactor));
                assert(inBlendParams.alphaBlendOp <= asset::EBO_MAX);
                outBlendState.alphaBlendOp = getVkBlendOpFromBlendOp(static_cast<asset::E_BLEND_OP>(inBlendParams.alphaBlendOp));
                outBlendState.colorWriteMask = getVkColorComponentFlagsFromColorWriteMask(inBlendParams.colorWriteMask);
            }
            colorBlendAttachmentCount_total += colorBlendAttachmentCount;

            vk_colorBlendStates[i].sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            vk_colorBlendStates[i].pNext = nullptr;
            vk_colorBlendStates[i].flags = 0u;
            vk_colorBlendStates[i].logicOpEnable = blendParams.logicOpEnable;
            vk_colorBlendStates[i].logicOp = getVkLogicOpFromLogicOp(static_cast<asset::E_LOGIC_OP>(blendParams.logicOp));
            vk_colorBlendStates[i].attachmentCount = colorBlendAttachmentCount;
            vk_colorBlendStates[i].pAttachments = vk_colorBlendAttachmentStates.data() + offset;
            vk_colorBlendStates[i].blendConstants[0] = 0.0f;
            vk_colorBlendStates[i].blendConstants[1] = 0.0f;
            vk_colorBlendStates[i].blendConstants[2] = 0.0f;
            vk_colorBlendStates[i].blendConstants[3] = 0.0f;
        }
        vk_createInfos[i].pColorBlendState = &vk_colorBlendStates[i];

        // Dynamic state
        vk_createInfos[i].pDynamicState = &vk_dynamicStateCreateInfo;

        vk_createInfos[i].layout = IBackendObject::device_compatibility_cast<const CVulkanPipelineLayout*>(rpIndie->getLayout(), this)->getInternalObject();
        vk_createInfos[i].renderPass = IBackendObject::device_compatibility_cast<const CVulkanRenderpass*>(creationParams[i].renderpass.get(), this)->getInternalObject();
        vk_createInfos[i].subpass = creationParams[i].subpassIx;
        vk_createInfos[i].basePipelineHandle = VK_NULL_HANDLE;
        vk_createInfos[i].basePipelineIndex = 0u;
    }

    core::vector<VkPipeline> vk_pipelines(params.size());
    if (m_devf.vk.vkCreateGraphicsPipelines(m_vkdev, vk_pipelineCache,
        static_cast<uint32_t>(params.size()), vk_createInfos.data(), nullptr, vk_pipelines.data()) == VK_SUCCESS)
    {
        for (size_t i = 0ull; i < params.size(); ++i)
        {
            output[i] = core::make_smart_refctd_ptr<CVulkanGraphicsPipeline>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this),
                std::move(creationParams[i]),
                vk_pipelines[i]);
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool CVulkanLogicalDevice::buildAccelerationStructures(
    core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation,
    const core::SRange<IGPUAccelerationStructure::HostBuildGeometryInfo>& pInfos,
    IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos)
{
    auto features = getEnabledFeatures();
    if(!features.accelerationStructure)
    {
        assert(false && "device acceleration structures is not enabled.");
        return false;
    }


    bool ret = false;
    if(!pInfos.empty() && deferredOperation.get() != nullptr)
    {
        VkDeferredOperationKHR vk_deferredOp = IBackendObject::device_compatibility_cast<CVulkanDeferredOperation *>(deferredOperation.get(), this)->getInternalObject();
        static constexpr size_t MaxGeometryPerBuildInfoCount = 64;
        static constexpr size_t MaxBuildInfoCount = 128;
        size_t infoCount = pInfos.size();
        assert(infoCount <= MaxBuildInfoCount);
                
        // TODO: Use better container when ready for these stack allocated memories.
        VkAccelerationStructureBuildGeometryInfoKHR vk_buildGeomsInfos[MaxBuildInfoCount] = {};

        uint32_t geometryArrayOffset = 0u;
        VkAccelerationStructureGeometryKHR vk_geometries[MaxGeometryPerBuildInfoCount * MaxBuildInfoCount] = {};

        IGPUAccelerationStructure::HostBuildGeometryInfo* infos = pInfos.begin();
        for(uint32_t i = 0; i < infoCount; ++i)
        {
            uint32_t geomCount = infos[i].geometries.size();

            assert(geomCount > 0);
            assert(geomCount <= MaxGeometryPerBuildInfoCount);

            vk_buildGeomsInfos[i] = CVulkanAccelerationStructure::getVkASBuildGeomInfoFromBuildGeomInfo(m_vkdev, &m_devf, infos[i], &vk_geometries[geometryArrayOffset]);
            geometryArrayOffset += geomCount; 
        }
                
        static_assert(sizeof(IGPUAccelerationStructure::BuildRangeInfo) == sizeof(VkAccelerationStructureBuildRangeInfoKHR));
        auto buildRangeInfos = reinterpret_cast<const VkAccelerationStructureBuildRangeInfoKHR* const*>(ppBuildRangeInfos);
        VkResult vk_res = m_devf.vk.vkBuildAccelerationStructuresKHR(m_vkdev, vk_deferredOp, infoCount, vk_buildGeomsInfos, buildRangeInfos);
        if(VK_SUCCESS == vk_res)
        {
            ret = true;
        }
    }
    return ret;
}

bool CVulkanLogicalDevice::copyAccelerationStructure(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::CopyInfo& copyInfo)
{
    auto features = getEnabledFeatures();
    if(!features.accelerationStructureHostCommands || !features.accelerationStructure)
    {
        assert(false && "device accelerationStructuresHostCommands is not enabled.");
        return false;
    }

    bool ret = false;
    if(deferredOperation.get() != nullptr)
    {
        VkDeferredOperationKHR vk_deferredOp = IBackendObject::device_compatibility_cast<CVulkanDeferredOperation *>(deferredOperation.get(), this)->getInternalObject();
        if(copyInfo.dst == nullptr || copyInfo.src == nullptr) 
        {
            assert(false && "invalid src or dst");
            return false;
        }

        VkCopyAccelerationStructureInfoKHR info = CVulkanAccelerationStructure::getVkASCopyInfo(m_vkdev, &m_devf, copyInfo);
        VkResult res = m_devf.vk.vkCopyAccelerationStructureKHR(m_vkdev, vk_deferredOp, &info);
        if(VK_SUCCESS == res)
        {
            ret = true;
        }
    }
    return ret;
}
    
bool CVulkanLogicalDevice::copyAccelerationStructureToMemory(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::HostCopyToMemoryInfo& copyInfo)
{
    auto features = getEnabledFeatures();
    if(!features.accelerationStructureHostCommands || !features.accelerationStructure)
    {
        assert(false && "device accelerationStructuresHostCommands is not enabled.");
        return false;
    }

    bool ret = false;
    if(deferredOperation.get() != nullptr)
    {
        VkDeferredOperationKHR vk_deferredOp = IBackendObject::device_compatibility_cast<CVulkanDeferredOperation *>(deferredOperation.get(), this)->getInternalObject();

        if(copyInfo.dst.isValid() == false || copyInfo.src == nullptr) 
        {
            assert(false && "invalid src or dst");
            return false;
        }

        VkCopyAccelerationStructureToMemoryInfoKHR info = CVulkanAccelerationStructure::getVkASCopyToMemoryInfo(m_vkdev, &m_devf, copyInfo);
        VkResult res = m_devf.vk.vkCopyAccelerationStructureToMemoryKHR(m_vkdev, vk_deferredOp, &info);
        if(VK_SUCCESS == res)
        {
            ret = true;
        }
    }
    return ret;
}

bool CVulkanLogicalDevice::copyAccelerationStructureFromMemory(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::HostCopyFromMemoryInfo& copyInfo)
{
    auto features = getEnabledFeatures();
    if(!features.accelerationStructureHostCommands || !features.accelerationStructure)
    {
        assert(false && "device accelerationStructuresHostCommands is not enabled.");
        return false;
    }

    bool ret = false;
    if(deferredOperation.get() != nullptr)
    {
        VkDeferredOperationKHR vk_deferredOp = IBackendObject::device_compatibility_cast<CVulkanDeferredOperation *>(deferredOperation.get(), this)->getInternalObject();
        if(copyInfo.dst == nullptr || copyInfo.src.isValid() == false) 
        {
            assert(false && "invalid src or dst");
            return false;
        }

        VkCopyMemoryToAccelerationStructureInfoKHR info = CVulkanAccelerationStructure::getVkASCopyFromMemoryInfo(m_vkdev, &m_devf, copyInfo);
        VkResult res = m_devf.vk.vkCopyMemoryToAccelerationStructureKHR(m_vkdev, vk_deferredOp, &info);
        if(VK_SUCCESS == res)
        {
            ret = true;
        }
    }
    return ret;
}

IGPUAccelerationStructure::BuildSizes CVulkanLogicalDevice::getAccelerationStructureBuildSizes(const IGPUAccelerationStructure::HostBuildGeometryInfo& pBuildInfo, const uint32_t* pMaxPrimitiveCounts)
{
    // TODO(Validation): Rayquery or RayTracing Pipeline must be enabled
    return getAccelerationStructureBuildSizes_impl(VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR, pBuildInfo, pMaxPrimitiveCounts);
}

IGPUAccelerationStructure::BuildSizes CVulkanLogicalDevice::getAccelerationStructureBuildSizes(const IGPUAccelerationStructure::DeviceBuildGeometryInfo& pBuildInfo, const uint32_t* pMaxPrimitiveCounts)
{
    // TODO(Validation): Rayquery or RayTracing Pipeline must be enabled
    return getAccelerationStructureBuildSizes_impl(VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, pBuildInfo, pMaxPrimitiveCounts);
}

core::smart_refctd_ptr<IQueryPool> CVulkanLogicalDevice::createQueryPool(IQueryPool::SCreationParams&& params)
{
    VkQueryPool vk_queryPool = VK_NULL_HANDLE;
    VkQueryPoolCreateInfo vk_qpci = CVulkanQueryPool::getVkCreateInfoFromCreationParams(std::move(params));
    auto vk_res = m_devf.vk.vkCreateQueryPool(m_vkdev, &vk_qpci, nullptr, &vk_queryPool);
    if(VK_SUCCESS != vk_res)
        return nullptr;
    return core::make_smart_refctd_ptr<CVulkanQueryPool>(core::smart_refctd_ptr<CVulkanLogicalDevice>(this), std::move(params), vk_queryPool);
}

bool CVulkanLogicalDevice::getQueryPoolResults(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void * pData, uint64_t stride, core::bitflag<IQueryPool::E_QUERY_RESULTS_FLAGS> flags)
{
    bool ret = false;
    if(queryPool != nullptr)
    {
        auto vk_queryPool = IBackendObject::device_compatibility_cast<CVulkanQueryPool*>(queryPool, this)->getInternalObject();
        auto vk_queryResultsflags = CVulkanQueryPool::getVkQueryResultsFlagsFromQueryResultsFlags(flags.value);
        auto vk_res = m_devf.vk.vkGetQueryPoolResults(m_vkdev, vk_queryPool, firstQuery, queryCount, dataSize, pData, static_cast<VkDeviceSize>(stride), vk_queryResultsflags);
        if(VK_SUCCESS == vk_res)
            ret = true;
    }
    return ret;
}