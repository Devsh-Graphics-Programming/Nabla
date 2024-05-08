#include "nbl/video/CVulkanLogicalDevice.h"

#include "nbl/video/CThreadSafeQueueAdapter.h"
#include "nbl/video/surface/CSurfaceVulkan.h"

#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanQueryPool.h"
#include "nbl/video/CVulkanCommandBuffer.h"


using namespace nbl;
using namespace nbl::video;



CVulkanLogicalDevice::CVulkanLogicalDevice(core::smart_refctd_ptr<const IAPIConnection>&& api, renderdoc_api_t* const rdoc, const IPhysicalDevice* const physicalDevice, const VkDevice vkdev, const SCreationParams& params)
: ILogicalDevice(std::move(api),physicalDevice,params,rdoc),
m_vkdev(vkdev), m_devf(vkdev), m_deferred_op_mempool(NODES_PER_BLOCK_DEFERRED_OP*sizeof(CVulkanDeferredOperation),1u,MAX_BLOCK_COUNT_DEFERRED_OP,static_cast<uint32_t>(sizeof(CVulkanDeferredOperation)))
{
    // create actual queue objects
    for (uint32_t i=0u; i<ILogicalDevice::MaxQueueFamilies; ++i)
    {
        const auto& qci = params.queueParams[i];
        const uint32_t offset = m_queueFamilyInfos[i].firstQueueIndex;
        const auto flags = qci.flags;
                    
        for (uint32_t j=0u; j<qci.count; ++j)
        {
            const float priority = qci.priorities[j];
                        
            VkQueue q;
            VkDeviceQueueInfo2 vk_info = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,nullptr};
            vk_info.queueFamilyIndex = i;
            vk_info.queueIndex = j;
            vk_info.flags = 0; // we don't do protected queues yet
            m_devf.vk.vkGetDeviceQueue(m_vkdev,i,j,&q);
                        
            const uint32_t ix = offset+j;
            auto queue = new CVulkanQueue(this,rdoc,static_cast<const CVulkanConnection*>(m_api.get())->getInternalObject(),q,i,flags,priority);
            (*m_queues)[ix] = new CThreadSafeQueueAdapter(this,queue);
        }
    }

    {
        const VkDescriptorSetLayoutCreateInfo vk_createInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 0,
            .pBindings = nullptr
        };
        const bool success = m_devf.vk.vkCreateDescriptorSetLayout(m_vkdev,&vk_createInfo,nullptr,&m_dummyDSLayout)==VK_SUCCESS;
        assert(success);
    }
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
ISemaphore::WAIT_RESULT CVulkanLogicalDevice::waitForSemaphores(const std::span<const ISemaphore::SWaitInfo> infos, const bool waitAll, const uint64_t timeout)
{
    using retval_t = ISemaphore::WAIT_RESULT;

    core::vector<VkSemaphore> semaphores(infos.size());
    core::vector<uint64_t> values(infos.size());
    auto outSemaphores = semaphores.data();
    auto outValues = values.data();
    for (const auto& info : infos)
    {
        auto sema = IBackendObject::device_compatibility_cast<const CVulkanSemaphore*>(info.semaphore,this);
        if (!sema)
            retval_t::_ERROR;
        *(outSemaphores++) = sema->getInternalObject();
        *(outValues++) = info.value;
    }

    VkSemaphoreWaitInfoKHR waitInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR,nullptr };
    waitInfo.flags = waitAll ? 0:VK_SEMAPHORE_WAIT_ANY_BIT_KHR;
    waitInfo.semaphoreCount = semaphores.size();
    waitInfo.pSemaphores = semaphores.data();
    waitInfo.pValues = values.data();
    switch (m_devf.vk.vkWaitSemaphores(m_vkdev,&waitInfo,timeout))
    {
        case VK_SUCCESS:
            return retval_t::SUCCESS;
        case VK_TIMEOUT:
            return retval_t::TIMEOUT;
        case VK_ERROR_DEVICE_LOST:
            return retval_t::DEVICE_LOST;
        default:
            break;
    }
    return retval_t::_ERROR;
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

    new (memory) CVulkanDeferredOperation(this,vk_deferredOp);
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
    ret.memory = core::make_smart_refctd_ptr<CVulkanMemoryAllocation>(this,info.size,allocateFlags,memoryPropertyFlags,info.dedication,vk_deviceMemory);
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

static inline void getVkMappedMemoryRanges(VkMappedMemoryRange* outRanges, const std::span<const ILogicalDevice::MappedMemoryRange>& ranges)
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
bool CVulkanLogicalDevice::flushMappedMemoryRanges_impl(const std::span<const MappedMemoryRange> ranges)
{
    constexpr uint32_t MAX_MEMORY_RANGE_COUNT = 408u;
    if (ranges.size()>MAX_MEMORY_RANGE_COUNT)
        return false;

    VkMappedMemoryRange vk_memoryRanges[MAX_MEMORY_RANGE_COUNT];
    getVkMappedMemoryRanges(vk_memoryRanges,ranges);
    return m_devf.vk.vkFlushMappedMemoryRanges(m_vkdev,ranges.size(),vk_memoryRanges)==VK_SUCCESS;
}
bool CVulkanLogicalDevice::invalidateMappedMemoryRanges_impl(const std::span<const MappedMemoryRange> ranges)
{
    constexpr uint32_t MAX_MEMORY_RANGE_COUNT = 408u;
    if (ranges.size()>MAX_MEMORY_RANGE_COUNT)
        return false;

    VkMappedMemoryRange vk_memoryRanges[MAX_MEMORY_RANGE_COUNT];
    getVkMappedMemoryRanges(vk_memoryRanges,ranges);
    return m_devf.vk.vkInvalidateMappedMemoryRanges(m_vkdev,ranges.size(),vk_memoryRanges)==VK_SUCCESS;
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
            vulkanBuffer->setDeviceAddress(m_devf.vk.vkGetBufferDeviceAddress(m_vkdev,&info));
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
    if (creationParams.queueFamilyIndexCount<2)
    {
        creationParams.queueFamilyIndexCount = 0;
        creationParams.queueFamilyIndices = nullptr;
    }
    vk_createInfo.sharingMode = creationParams.isConcurrentSharing() ? VK_SHARING_MODE_CONCURRENT:VK_SHARING_MODE_EXCLUSIVE;
    vk_createInfo.queueFamilyIndexCount = creationParams.queueFamilyIndexCount;
    vk_createInfo.pQueueFamilyIndices = creationParams.queueFamilyIndices;

    VkBuffer vk_buffer;
    if (m_devf.vk.vkCreateBuffer(m_vkdev,&vk_createInfo,nullptr,&vk_buffer)!=VK_SUCCESS)
        return nullptr;
    return core::make_smart_refctd_ptr<CVulkanBuffer>(this,std::move(creationParams),vk_buffer);
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
        return core::make_smart_refctd_ptr<CVulkanBufferView>(this,std::move(underlying),_fmt,vk_bufferView);
    return nullptr;
}

core::smart_refctd_ptr<IGPUImage> CVulkanLogicalDevice::createImage_impl(IGPUImage::SCreationParams&& params)
{
    const bool hasStencil = asset::isDepthOrStencilFormat(params.format) && !asset::isDepthOnlyFormat(params.format);
    VkImageStencilUsageCreateInfo vk_stencilUsage = { VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO, nullptr };
    vk_stencilUsage.stencilUsage = getVkImageUsageFlagsFromImageUsageFlags(params.actualStencilUsage().value,true);

    std::array<VkFormat,asset::E_FORMAT::EF_COUNT> vk_formatList;
    VkImageFormatListCreateInfo vk_formatListStruct = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,hasStencil ? &vk_stencilUsage:nullptr};
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
    if (params.queueFamilyIndexCount < 2)
    {
        params.queueFamilyIndexCount = 0;
        params.queueFamilyIndices = nullptr;
    }
    vk_createInfo.sharingMode = params.isConcurrentSharing() ? VK_SHARING_MODE_CONCURRENT:VK_SHARING_MODE_EXCLUSIVE;
    vk_createInfo.queueFamilyIndexCount = params.queueFamilyIndexCount;
    vk_createInfo.pQueueFamilyIndices = params.queueFamilyIndices;
    vk_createInfo.initialLayout = params.preinitialized ? VK_IMAGE_LAYOUT_PREINITIALIZED:VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage vk_image;
    if (m_devf.vk.vkCreateImage(m_vkdev,&vk_createInfo,nullptr,&vk_image)!=VK_SUCCESS)
        return nullptr;
    return core::make_smart_refctd_ptr<CVulkanImage>(this,std::move(params),vk_image);
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
    return nullptr;
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
    vk_createInfo.anisotropyEnable = (_params.AnisotropicFilter != 0u); // ROADMAP 2022
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
    vasci.createFlags = static_cast<VkAccelerationStructureCreateFlagsKHR>(params.flags.value);
    vasci.type = type;
    vasci.buffer = static_cast<const CVulkanBuffer*>(params.bufferRange.buffer.get())->getInternalObject();
    vasci.offset = params.bufferRange.offset;
    vasci.size = params.bufferRange.size;

    VkAccelerationStructureKHR vk_as;
    if (m_devf.vk.vkCreateAccelerationStructureKHR(m_vkdev,&vasci,nullptr,&vk_as)==VK_SUCCESS)
        return vk_as;
    return VK_NULL_HANDLE;
}


auto CVulkanLogicalDevice::getAccelerationStructureBuildSizes_impl(
    const bool hostBuild, const core::bitflag<IGPUTopLevelAccelerationStructure::BUILD_FLAGS> flags,
    const bool motionBlur, const uint32_t maxInstanceCount
) const -> AccelerationStructureBuildSizes
{
    VkAccelerationStructureGeometryKHR geometry = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,nullptr,VK_GEOMETRY_TYPE_INSTANCES_KHR};
    geometry.geometry.instances = {};
    // no "geometry flags" are valid for all instances!
    geometry.flags = static_cast<VkGeometryFlagBitsKHR>(0);
    
    return getAccelerationStructureBuildSizes_impl_impl(hostBuild,true,getVkASBuildFlagsFrom<IGPUTopLevelAccelerationStructure>(flags,motionBlur),{&geometry,1},&maxInstanceCount);
}
auto CVulkanLogicalDevice::getAccelerationStructureBuildSizes_impl_impl(
    const bool hostBuild, const bool isTLAS, const VkBuildAccelerationStructureFlagsKHR flags,
    const std::span<const VkAccelerationStructureGeometryKHR> geometries, const uint32_t* const pMaxPrimitiveOrInstanceCounts
) const -> AccelerationStructureBuildSizes
{
    VkAccelerationStructureBuildGeometryInfoKHR vk_buildGeomsInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,nullptr};
    vk_buildGeomsInfo.type = isTLAS ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR:VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    vk_buildGeomsInfo.flags = flags;
    vk_buildGeomsInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_MAX_ENUM_KHR; // ignored by this command
    vk_buildGeomsInfo.srcAccelerationStructure = VK_NULL_HANDLE; // ignored by this command
    vk_buildGeomsInfo.dstAccelerationStructure = VK_NULL_HANDLE; // ignored by this command
    vk_buildGeomsInfo.geometryCount = geometries.size();
    vk_buildGeomsInfo.pGeometries = geometries.data();
    vk_buildGeomsInfo.ppGeometries = nullptr;
    vk_buildGeomsInfo.scratchData.deviceAddress = 0x0ull; // ignored by this command

    VkAccelerationStructureBuildSizesInfoKHR vk_ret = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,nullptr};
    m_devf.vk.vkGetAccelerationStructureBuildSizesKHR(m_vkdev,hostBuild ? VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR:VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&vk_buildGeomsInfo,pMaxPrimitiveOrInstanceCounts,&vk_ret);
    return AccelerationStructureBuildSizes{
        .accelerationStructureSize = vk_ret.accelerationStructureSize,
        .updateScratchSize = vk_ret.updateScratchSize,
        .buildScratchSize = vk_ret.buildScratchSize
    };
}

bool CVulkanLogicalDevice::writeAccelerationStructuresProperties_impl(const std::span<const IGPUAccelerationStructure* const> accelerationStructures, const IQueryPool::TYPE type, size_t* data, const size_t stride)
{
    core::vector<VkAccelerationStructureKHR> vk_accelerationStructures;
    vk_accelerationStructures.reserve(accelerationStructures.size());
    for (const auto& as : accelerationStructures)
        vk_accelerationStructures.push_back(*reinterpret_cast<const VkAccelerationStructureKHR*>(as->getNativeHandle()));
    return m_devf.vk.vkWriteAccelerationStructuresPropertiesKHR(m_vkdev,vk_accelerationStructures.size(),vk_accelerationStructures.data(),static_cast<VkQueryType>(type),stride*accelerationStructures.size(),data,stride);
}

auto CVulkanLogicalDevice::copyAccelerationStructure_impl(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::CopyInfo& copyInfo) -> DEFERRABLE_RESULT
{
    const auto info = getVkCopyAccelerationStructureInfoFrom(copyInfo);
    return getDeferrableResultFrom(m_devf.vk.vkCopyAccelerationStructureKHR(m_vkdev,static_cast<CVulkanDeferredOperation*>(deferredOperation)->getInternalObject(),&info));
}

auto CVulkanLogicalDevice::copyAccelerationStructureToMemory_impl(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::HostCopyToMemoryInfo& copyInfo) -> DEFERRABLE_RESULT
{
    const auto info = getVkCopyAccelerationStructureToMemoryInfoFrom(copyInfo);
    return getDeferrableResultFrom(m_devf.vk.vkCopyAccelerationStructureToMemoryKHR(m_vkdev,static_cast<CVulkanDeferredOperation*>(deferredOperation)->getInternalObject(),&info));
}

auto CVulkanLogicalDevice::copyAccelerationStructureFromMemory_impl(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::HostCopyFromMemoryInfo& copyInfo) -> DEFERRABLE_RESULT
{
    const auto info = getVkCopyMemoryToAccelerationStructureInfoFrom(copyInfo);
    return getDeferrableResultFrom(m_devf.vk.vkCopyMemoryToAccelerationStructureKHR(m_vkdev,static_cast<CVulkanDeferredOperation*>(deferredOperation)->getInternalObject(),&info));
}


core::smart_refctd_ptr<IGPUShader> CVulkanLogicalDevice::createShader_impl(const asset::ICPUShader* spirvShader)
{
    auto spirv = spirvShader->getContent();

    VkShaderModuleCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    vk_createInfo.pNext = nullptr;
    vk_createInfo.flags = static_cast<VkShaderModuleCreateFlags>(0u); // reserved for future use by Vulkan
    vk_createInfo.codeSize = spirv->getSize();
    vk_createInfo.pCode = static_cast<const uint32_t*>(spirv->getPointer());

    VkShaderModule vk_shaderModule;
    if (m_devf.vk.vkCreateShaderModule(m_vkdev,&vk_createInfo,nullptr,&vk_shaderModule)==VK_SUCCESS)
        return core::make_smart_refctd_ptr<video::CVulkanShader>(this,spirvShader->getStage(),std::string(spirvShader->getFilepathHint()),vk_shaderModule);
    return nullptr;
}


core::smart_refctd_ptr<IGPUDescriptorSetLayout> CVulkanLogicalDevice::createDescriptorSetLayout_impl(const std::span<const IGPUDescriptorSetLayout::SBinding> bindings, const uint32_t maxSamplersCount)
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
        return core::make_smart_refctd_ptr<CVulkanDescriptorSetLayout>(this,bindings,vk_dsLayout);
    return nullptr;
}

core::smart_refctd_ptr<IGPUPipelineLayout> CVulkanLogicalDevice::createPipelineLayout_impl(
    const std::span<const asset::SPushConstantRange> pcRanges,
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
        vk_dsLayouts[i] = tmp[i] ? static_cast<const CVulkanDescriptorSetLayout*>(tmp[i].get())->getInternalObject():m_dummyDSLayout;
    }
    nonNullSetLayoutCount++;

    VkPushConstantRange vk_pushConstantRanges[SPhysicalDeviceLimits::MaxMaxPushConstantsSize];
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
        return core::make_smart_refctd_ptr<CVulkanPipelineLayout>(this,pcRanges,std::move(layout0),std::move(layout1),std::move(layout2),std::move(layout3),vk_pipelineLayout);
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
        return core::make_smart_refctd_ptr<CVulkanDescriptorPool>(this,std::move(createInfo),vk_descriptorPool);
    return nullptr;
}

// a lot of empirical research went into defining this constant
constexpr uint32_t MaxDescriptorSetAsWrites = 69u;

void CVulkanLogicalDevice::updateDescriptorSets_impl(const SUpdateDescriptorSetsParams& params)
{
    // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of
    // VkWriteDescriptorSetAccelerationStructureKHR, VkWriteDescriptorSetAccelerationStructureNV, or VkWriteDescriptorSetInlineUniformBlockEXT
    core::vector<VkWriteDescriptorSet> vk_writeDescriptorSets(params.writes.size(),{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr});
    core::vector<VkWriteDescriptorSetAccelerationStructureKHR> vk_writeDescriptorSetAS(params.accelerationStructureWriteCount,{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,nullptr});

    core::vector<VkDescriptorBufferInfo> vk_bufferInfos(params.bufferCount);
    core::vector<VkDescriptorImageInfo> vk_imageInfos(params.imageCount);
    core::vector<VkBufferView> vk_bufferViews(params.bufferViewCount);
    core::vector<VkAccelerationStructureKHR> vk_accelerationStructures(params.accelerationStructureCount);
    {
        auto outWrite = vk_writeDescriptorSets.data();
        auto outWriteAS = vk_writeDescriptorSetAS.data();

        auto outBufferInfo = vk_bufferInfos.data();
        auto outImageInfo = vk_imageInfos.data();
        auto outBufferViewInfo = vk_bufferViews.data();
        auto outASInfo = vk_accelerationStructures.data();
        for (auto i=0; i<params.writes.size(); i++)
        {
            const auto& write = params.writes[i];
            const auto type = params.pWriteTypes[i];
            const auto& infos = write.info;

            outWrite->dstSet = static_cast<const CVulkanDescriptorSet*>(write.dstSet)->getInternalObject();
            outWrite->dstBinding = write.binding;
            outWrite->dstArrayElement = write.arrayElement;
            outWrite->descriptorType = getVkDescriptorTypeFromDescriptorType(type);
            outWrite->descriptorCount = write.count;
            switch (asset::IDescriptor::GetTypeCategory(type))
            {
                case asset::IDescriptor::EC_BUFFER:
                {
                    outWrite->pBufferInfo = outBufferInfo;
                    for (auto j=0u; j<write.count; j++,outBufferInfo++)
                    {
                        const auto& bufferInfo = infos[j].info.buffer;
                        outBufferInfo->buffer = static_cast<const CVulkanBuffer*>(infos[j].desc.get())->getInternalObject();
                        outBufferInfo->offset = bufferInfo.offset;
                        outBufferInfo->range = bufferInfo.size;
                    }
                } break;
                case asset::IDescriptor::EC_IMAGE:
                {
                    outWrite->pImageInfo = outImageInfo;
                    for (auto j=0u; j<write.count; j++,outImageInfo++)
                    {
                        const auto& imageInfo = infos[j].info.image;
                        outImageInfo->sampler = imageInfo.sampler ? static_cast<const CVulkanSampler*>(imageInfo.sampler.get())->getInternalObject():VK_NULL_HANDLE;
                        outImageInfo->imageView = static_cast<const CVulkanImageView*>(infos[j].desc.get())->getInternalObject();
                        outImageInfo->imageLayout = getVkImageLayoutFromImageLayout(imageInfo.imageLayout);
                    }
                } break;
                case asset::IDescriptor::EC_BUFFER_VIEW:
                {
                    outWrite->pTexelBufferView = outBufferViewInfo;
                    for (auto j=0u; j<write.count; j++,outBufferViewInfo++)
                        *outBufferViewInfo = static_cast<const CVulkanBufferView*>(infos[j].desc.get())->getInternalObject();
                } break;
                case asset::IDescriptor::EC_ACCELERATION_STRUCTURE:
                {
                    outWriteAS->accelerationStructureCount = write.count;
                    outWriteAS->pAccelerationStructures = outASInfo;
                    for (auto j=0u; j<write.count; j++,outASInfo++)
                        *outASInfo = *reinterpret_cast<const VkAccelerationStructureKHR*>(static_cast<const IGPUAccelerationStructure*>(infos[j].desc.get())->getNativeHandle());
                    outWrite->pNext = outWriteAS++;
                } break;
                default:
                    assert(!"Invalid code path.");
            }
            outWrite++;
        }
    }

    core::vector<VkCopyDescriptorSet> vk_copyDescriptorSets(params.copies.size(),{VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,nullptr});
    auto outCopy = vk_copyDescriptorSets.data();
    for (const auto& copy : params.copies)
    {
        outCopy->srcSet = static_cast<const CVulkanDescriptorSet*>(copy.srcSet)->getInternalObject();
        outCopy->srcBinding = copy.srcBinding;
        outCopy->srcArrayElement = copy.srcArrayElement;
        outCopy->dstSet = static_cast<const CVulkanDescriptorSet*>(copy.dstSet)->getInternalObject();
        outCopy->dstBinding = copy.dstBinding;
        outCopy->dstArrayElement = copy.dstArrayElement;
        outCopy->descriptorCount = copy.count;
        outCopy++;
    }

    m_devf.vk.vkUpdateDescriptorSets(m_vkdev,vk_writeDescriptorSets.size(),vk_writeDescriptorSets.data(),vk_copyDescriptorSets.size(),vk_copyDescriptorSets.data());
}

void CVulkanLogicalDevice::nullifyDescriptors_impl(const SDropDescriptorSetsParams& params)
{
    const auto& drops = params.drops;
    if (getEnabledFeatures().nullDescriptor)
    {
        return;
    }

	core::vector<VkWriteDescriptorSet> vk_writeDescriptorSets(drops.size(),{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr});
	core::vector<VkWriteDescriptorSetAccelerationStructureKHR> vk_writeDescriptorSetAS(params.accelerationStructureWriteCount,{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,nullptr});

	size_t maxSize = core::max(
        core::max(params.bufferCount * sizeof(VkDescriptorBufferInfo), params.imageCount * sizeof(VkDescriptorImageInfo)), 
        core::max(params.bufferViewCount * sizeof(VkBufferView), params.accelerationStructureCount * sizeof(VkAccelerationStructureKHR))
    );

	core::vector<uint8_t> nullDescriptors(maxSize, 0u);

	{
		auto outWrite = vk_writeDescriptorSets.data();
		auto outWriteAS = vk_writeDescriptorSetAS.data();

		for (auto i=0; i<drops.size(); i++)
		{
			const auto& write = drops[i];
			auto descriptorType = write.dstSet->getBindingType(write.binding);

			outWrite->dstSet = static_cast<const CVulkanDescriptorSet*>(write.dstSet)->getInternalObject();
			outWrite->dstBinding = write.binding;
			outWrite->dstArrayElement = write.arrayElement;
			outWrite->descriptorType = getVkDescriptorTypeFromDescriptorType(descriptorType);
			outWrite->descriptorCount = write.count;
			switch (asset::IDescriptor::GetTypeCategory(descriptorType))
			{
			case asset::IDescriptor::EC_BUFFER:
				outWrite->pBufferInfo = reinterpret_cast<VkDescriptorBufferInfo*>(nullDescriptors.data());
				break;
			case asset::IDescriptor::EC_IMAGE:
				outWrite->pImageInfo = reinterpret_cast<VkDescriptorImageInfo*>(nullDescriptors.data());
				break;
			case asset::IDescriptor::EC_BUFFER_VIEW:
				outWrite->pTexelBufferView = reinterpret_cast<VkBufferView*>(nullDescriptors.data());
				break;
			case asset::IDescriptor::EC_ACCELERATION_STRUCTURE:
				outWriteAS->accelerationStructureCount = write.count;
				outWriteAS->pAccelerationStructures = reinterpret_cast<VkAccelerationStructureKHR*>(nullDescriptors.data());
				outWrite->pNext = outWriteAS++;
				break;
            default:
                assert(!"Invalid code path.");
			}
			outWrite++;
		}
	}
	m_devf.vk.vkUpdateDescriptorSets(m_vkdev,vk_writeDescriptorSets.size(),vk_writeDescriptorSets.data(),0,nullptr);
}


core::smart_refctd_ptr<IGPURenderpass> CVulkanLogicalDevice::createRenderpass_impl(const IGPURenderpass::SCreationParams& params, IGPURenderpass::SCreationParamValidationResult&& validation)
{
    using params_t = IGPURenderpass::SCreationParams;

    core::vector<VkAttachmentDescription2> attachments(validation.depthStencilAttachmentCount+validation.colorAttachmentCount,{VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,nullptr});
    core::vector<VkAttachmentDescriptionStencilLayout> stencilAttachmentLayouts(validation.depthStencilAttachmentCount,{VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_STENCIL_LAYOUT,nullptr});
    {
        auto outAttachment = attachments.begin();
        auto failAttachment = [&]<typename T> requires is_any_of_v<T,params_t::SDepthStencilAttachmentDescription,params_t::SColorAttachmentDescription>(const T& desc) -> bool
        {
            outAttachment->flags = desc.mayAlias ? VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT:0;
            outAttachment->format = getVkFormatFromFormat(desc.format);
            outAttachment->samples = static_cast<VkSampleCountFlagBits>(desc.samples);

            outAttachment++;
            return false;
        };
        auto outStencilLayout = stencilAttachmentLayouts.data();
        for (uint32_t i=0u; i<validation.depthStencilAttachmentCount; i++,outStencilLayout++)
        {
            const auto& desc = params.depthStencilAttachments[i];
            outAttachment->loadOp = getVkAttachmentLoadOpFrom(desc.loadOp.depth);
            outAttachment->storeOp = getVkAttachmentStoreOpFrom(desc.storeOp.depth);
            outAttachment->stencilLoadOp = getVkAttachmentLoadOpFrom(desc.loadOp.actualStencilOp());
            outAttachment->stencilStoreOp = getVkAttachmentStoreOpFrom(desc.storeOp.actualStencilOp());
            outAttachment->initialLayout = getVkImageLayoutFromImageLayout(desc.initialLayout.depth);
            outAttachment->finalLayout = getVkImageLayoutFromImageLayout(desc.finalLayout.depth);
            // For depth-only formats, the VkAttachmentDescriptionStencilLayout structure is ignored.
            outAttachment->pNext = outStencilLayout;
            outStencilLayout->stencilInitialLayout = getVkImageLayoutFromImageLayout(desc.initialLayout.actualStencilLayout());
            outStencilLayout->stencilFinalLayout = getVkImageLayoutFromImageLayout(desc.finalLayout.actualStencilLayout());

            if (failAttachment(params.depthStencilAttachments[i]))
                return nullptr;
        }
        for (uint32_t i=0u; i<validation.colorAttachmentCount; i++)
        {
            const auto& desc = params.colorAttachments[i];
            outAttachment->loadOp = getVkAttachmentLoadOpFrom(desc.loadOp);
            outAttachment->storeOp = getVkAttachmentStoreOpFrom(desc.storeOp);
            outAttachment->initialLayout = getVkImageLayoutFromImageLayout(desc.initialLayout);
            outAttachment->finalLayout = getVkImageLayoutFromImageLayout(desc.finalLayout);

            if (failAttachment(desc))
                return nullptr;
        }
    }
    
    core::vector<VkSubpassDescription2> subpasses(validation.subpassCount,{VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,nullptr});
    using subpass_desc_t = IGPURenderpass::SCreationParams::SSubpassDescription;
    // worst case sizing: 2 attachments, render and resolve for each of the color and depth attachments
    constexpr size_t MaxWriteableAttachments = (subpass_desc_t::MaxColorAttachments+1u)*2u;
    core::vector<VkAttachmentReference2> attachmentRef(MaxWriteableAttachments*validation.subpassCount+validation.totalInputAttachmentCount,{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,nullptr});
    // one depth-stencil attachment, means 1 resolve depth stencil at most!
    core::vector<VkSubpassDescriptionDepthStencilResolve> depthStencilResolve(validation.subpassCount,{VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE,nullptr});
    // one depth-stencil attachmentand possibly one depth-stencil resolve attachment, so max 2 stencil layouts
    core::vector<VkAttachmentReferenceStencilLayout> stencilLayout(validation.subpassCount*2,{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_STENCIL_LAYOUT,nullptr});
    core::vector<uint32_t> preserveAttachment(validation.totalPreserveAttachmentCount);
    {
        auto outSubpass = subpasses.begin();
        auto outAttachmentRef = attachmentRef.data();
        auto outStencilLayout = stencilLayout.data();
        auto pushAttachmentRef = [validation,&outAttachmentRef,&outStencilLayout]<typename layout_t>(const subpass_desc_t::SAttachmentRef<layout_t>& ref)->bool
        {
            if (ref.used())
            {
                if constexpr (std::is_same_v<layout_t,IGPUImage::SDepthStencilLayout>)
                {
                    outAttachmentRef->attachment = ref.attachmentIndex;
                    outAttachmentRef->layout = getVkImageLayoutFromImageLayout(ref.layout.depth);
                    outStencilLayout->stencilLayout = getVkImageLayoutFromImageLayout(ref.layout.actualStencilLayout());
                    outAttachmentRef->pNext = outStencilLayout++;
                }
                else
                {
                    // need to offset in the whole array
                    outAttachmentRef->attachment = validation.depthStencilAttachmentCount+ref.attachmentIndex;
                    outAttachmentRef->layout = getVkImageLayoutFromImageLayout(ref.layout);
                }
                // aspect mask gets ignored for anything thats not an input attachment
            }
            else
                outAttachmentRef->attachment = VK_ATTACHMENT_UNUSED;
            outAttachmentRef++;
            return ref.used();
        };
        auto outDepthStencilResolve = depthStencilResolve.data();
        auto outPreserveAttachment = preserveAttachment.data();
        for (uint32_t i=0u; i<validation.subpassCount; i++,outSubpass++)
        {
            const subpass_desc_t& subpass = params.subpasses[i];
            outSubpass->flags = static_cast<VkSubpassDescriptionFlags>(subpass.flags.value);
            outSubpass->pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            outSubpass->viewMask = subpass.viewMask;
            outSubpass->inputAttachmentCount = 0;
            outSubpass->pInputAttachments = outAttachmentRef;
            core::visit_token_terminated_array(subpass.inputAttachments,subpass_desc_t::InputAttachmentsEnd,[&](const subpass_desc_t::SInputAttachmentRef& inputAttachmentRef)->bool
            {
                outAttachmentRef->aspectMask = static_cast<VkImageAspectFlags>(inputAttachmentRef.aspectMask.value);
                if (inputAttachmentRef.isColor())
                    pushAttachmentRef(inputAttachmentRef.asColor);
                else
                    pushAttachmentRef(inputAttachmentRef.asDepthStencil);
                outSubpass->inputAttachmentCount++;
                return true;
            });
            outSubpass->colorAttachmentCount = 0;
            outSubpass->pColorAttachments = outAttachmentRef;
            for (auto j=0u; j<subpass_desc_t::MaxColorAttachments; j++)
            {
                const auto& att = subpass.colorAttachments[j];
                if (pushAttachmentRef(att.render))
                    outSubpass->colorAttachmentCount = i+1;
            }
            outSubpass->pResolveAttachments = outAttachmentRef;
            for (auto j=0u; j<outSubpass->colorAttachmentCount; j++)
                pushAttachmentRef(subpass.colorAttachments[i].resolve);
            if (subpass.depthStencilAttachment.render.used())
            {
                const auto& render = subpass.depthStencilAttachment.render;
                outSubpass->pDepthStencilAttachment = outAttachmentRef;
                pushAttachmentRef(render);
                // have to add reoslve anyway because of multisample to single sample render
                outSubpass->pNext = outDepthStencilResolve;
                outDepthStencilResolve->depthResolveMode = static_cast<VkResolveModeFlagBits>(subpass.depthStencilAttachment.resolveMode.depth);
                outDepthStencilResolve->stencilResolveMode = static_cast<VkResolveModeFlagBits>(subpass.depthStencilAttachment.resolveMode.stencil);
                const auto& resolve = subpass.depthStencilAttachment.resolve;
                if (resolve.used())
                {
                    outDepthStencilResolve->pDepthStencilResolveAttachment = outAttachmentRef;
                    pushAttachmentRef(resolve);
                }
                outDepthStencilResolve++;
            }
            else
                outSubpass->pDepthStencilAttachment = nullptr;
            outSubpass->pPreserveAttachments = outPreserveAttachment;
            core::visit_token_terminated_array(subpass.preserveAttachments, subpass_desc_t::PreserveAttachmentsEnd, [&](const subpass_desc_t::SPreserveAttachmentRef& preserveAttachmentRef)->bool
            {
                *outPreserveAttachment = preserveAttachmentRef.index;
                if (preserveAttachmentRef.color)
                    *outPreserveAttachment += validation.depthStencilAttachmentCount;
                outPreserveAttachment++;
                return true;
            });
            outSubpass->preserveAttachmentCount = outPreserveAttachment-outSubpass->pPreserveAttachments;
        }
    }

    core::vector<VkSubpassDependency2> dependencies(validation.dependencyCount,{VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,nullptr});
    core::vector<VkMemoryBarrier2> barriers(validation.dependencyCount,{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,nullptr});
    {
        auto outDependency = dependencies.data();
        auto outBarrier = barriers.data();
        auto getSubpassIndex = [](const uint32_t ix)->uint32_t{return ix!=IGPURenderpass::SCreationParams::SSubpassDependency::External ? ix:VK_SUBPASS_EXTERNAL;};
        for (uint32_t i=0u; i<validation.dependencyCount; i++)
        {
            const auto& dep = params.dependencies[i];
            outBarrier->srcStageMask = getVkPipelineStageFlagsFromPipelineStageFlags(dep.memoryBarrier.srcStageMask);
            outBarrier->dstStageMask = getVkPipelineStageFlagsFromPipelineStageFlags(dep.memoryBarrier.dstStageMask);
            outBarrier->srcAccessMask = getVkAccessFlagsFromAccessFlags(dep.memoryBarrier.srcAccessMask);
            outBarrier->dstAccessMask = getVkAccessFlagsFromAccessFlags(dep.memoryBarrier.dstAccessMask);
            outDependency->pNext = outBarrier++;
            outDependency->srcSubpass = getSubpassIndex(dep.srcSubpass);
            outDependency->dstSubpass = getSubpassIndex(dep.dstSubpass);
            outDependency->dependencyFlags = static_cast<VkDependencyFlags>(dep.flags.value);
            outDependency->viewOffset = dep.viewOffset;
            outDependency++;
        }
    }

    constexpr auto MaxMultiviewViewCount = IGPURenderpass::SCreationParams::MaxMultiviewViewCount;
    uint32_t viewMasks[MaxMultiviewViewCount] = { 0u };
    // group up
    for (auto i=0u; i<MaxMultiviewViewCount; i++)
    if (params.viewCorrelationGroup[i]<MaxMultiviewViewCount) // not default
        viewMasks[i] |= 0x1u<<i;
    // compact (removing zero valued entries)
    const auto viewMaskCount = std::remove_if(viewMasks,viewMasks+MaxMultiviewViewCount,[](const uint32_t mask)->bool{return mask==0;})-viewMasks;

    // Nothing useful in pNext, didn't implement VRS yet
    VkRenderPassCreateInfo2 createInfo = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2,nullptr};
    createInfo.flags = 0; // reserved for future use according to Vulkan 1.3.264 spec
    createInfo.attachmentCount = attachments.size();
    createInfo.pAttachments = attachments.data();
    createInfo.subpassCount = subpasses.size();
    createInfo.pSubpasses = subpasses.data();
    createInfo.dependencyCount = dependencies.size();
    createInfo.pDependencies = dependencies.data();
    createInfo.correlatedViewMaskCount = viewMaskCount;
    createInfo.pCorrelatedViewMasks = viewMasks;

    VkRenderPass vk_renderpass;
    if (m_devf.vk.vkCreateRenderPass2(m_vkdev,&createInfo,nullptr,&vk_renderpass)==VK_SUCCESS)
        return core::make_smart_refctd_ptr<CVulkanRenderpass>(this,params,validation,vk_renderpass);
    return nullptr;
}

core::smart_refctd_ptr<IGPUFramebuffer> CVulkanLogicalDevice::createFramebuffer_impl(IGPUFramebuffer::SCreationParams&& params)
{
    auto* const renderpass = static_cast<CVulkanRenderpass*>(params.renderpass.get());
    core::vector<VkImageView> attachments;
    {
        attachments.reserve(renderpass->getDepthStencilAttachmentCount()+renderpass->getColorAttachmentCount());
        auto pushAttachment = [&attachments](IGPUImageView* const view) -> void
        {
            attachments.push_back(static_cast<CVulkanImageView*>(view)->getInternalObject());
        };

        for (auto i=0u; i<renderpass->getDepthStencilAttachmentCount(); i++)
            pushAttachment(params.depthStencilAttachments[i]);
        for (auto i=0u; i<renderpass->getColorAttachmentCount(); i++)
            pushAttachment(params.colorAttachments[i]);
    }

    VkFramebufferCreateInfo createInfo = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,nullptr};
    createInfo.flags = 0; // WE SHALL NOT EXPOSE IMAGELESS FRAMEBUFFER EXTENSION
    createInfo.renderPass = renderpass->getInternalObject();
    createInfo.attachmentCount = attachments.size();
    createInfo.pAttachments = attachments.data();
    createInfo.width = params.width;
    createInfo.height = params.height;
    createInfo.layers = params.layers;
    
    VkFramebuffer vk_framebuffer;
    if (m_devf.vk.vkCreateFramebuffer(m_vkdev,&createInfo,nullptr,&vk_framebuffer)==VK_SUCCESS)
        return core::make_smart_refctd_ptr<CVulkanFramebuffer>(core::smart_refctd_ptr<CVulkanLogicalDevice>(this),std::move(params),vk_framebuffer);
    return nullptr;
}


VkPipelineShaderStageCreateInfo getVkShaderStageCreateInfoFrom(
    const IGPUShader::SSpecInfo& specInfo,
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo* &outRequiredSubgroupSize,
    VkSpecializationInfo* &outSpecInfo, VkSpecializationMapEntry* &outSpecMapEntry, uint8_t* &outSpecData
)
{
    VkPipelineShaderStageCreateInfo retval = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr};
    {
        // VkDebugUtilsObjectNameInfoEXT is weird because it comes form `graphicsPipelineLibrary` feature which allows to create pipelines
        // without creating VkShaderModule (IGPUShaders) beforehand, so thats how you name individual shaders in that case.
        // In that case you'd set `module=VK_NULL_HANDLE` and put a `vkShaderModuleCreateInfo` in the `pNext`.
        // There's another thing called `shaderModuleIdentifier` which lets you "pull-in"; built-ins from a driver or 
        // provide tight control over the cache handles making sure you hit it (what the `FAIL_ON_PIPELINE_COMPILE_REQUIRED` flag is for),
        // in that case you have a `VkPipelineShaderStageModuleIdentifier` in `pNext` with non-zero length identifier.
        // TL;DR Basically you can skip needing the SPIR-V contents to hash the IGPUShader, we may implement this later on.
        void** ppNext = const_cast<void**>(&retval.pNext);
        // TODO: VkShaderModuleValidationCacheCreateInfoEXT from VK_EXT_validation_cache
        // TODO: VkPipelineRobustnessCreateInfoEXT from VK_EXT_pipeline_robustness (allows per-pipeline control of robustness)

        // Implicit: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pNext-02754
        using subgroup_size_t = std::remove_reference_t<decltype(specInfo)>::SUBGROUP_SIZE;
        if (specInfo.requiredSubgroupSize>=subgroup_size_t::REQUIRE_4)
        {
            *ppNext = outRequiredSubgroupSize;
            ppNext = &outRequiredSubgroupSize->pNext;
            outRequiredSubgroupSize->requiredSubgroupSize = 0x1u<<static_cast<uint8_t>(specInfo.requiredSubgroupSize);
            outRequiredSubgroupSize++;
        }
        else if (specInfo.requiredSubgroupSize==subgroup_size_t::VARYING)
            retval.flags = VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT;
        else
            retval.flags = 0;

        const auto stage = specInfo.shader->getStage();
        if (specInfo.requireFullSubgroups)
        {
            assert(stage==IGPUShader::ESS_COMPUTE/*TODO: Or Mesh Or Task*/);
            retval.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        }
        // Implicit: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-00706
        retval.stage = static_cast<VkShaderStageFlagBits>(stage);
        retval.module = static_cast<const CVulkanShader*>(specInfo.shader)->getInternalObject();
        retval.pName = specInfo.entryPoint.c_str();
        outSpecInfo->pMapEntries = outSpecMapEntry;
        outSpecInfo->dataSize = 0;
        const uint8_t* const specDataBegin = outSpecData;
        outSpecInfo->pData = specDataBegin;
        if (specInfo.entries)
        {
            outSpecInfo->mapEntryCount = specInfo.entries->size();
            for (const auto& entry : *specInfo.entries)
            {
                outSpecMapEntry->constantID = entry.first;
                outSpecMapEntry->offset = std::distance<const uint8_t*>(specDataBegin,outSpecData);
                outSpecMapEntry->size = entry.second.size;
                memcpy(outSpecData,entry.second.data,outSpecMapEntry->size);
                outSpecData += outSpecMapEntry->size;
                outSpecMapEntry++;
            }
        }
        else
            outSpecInfo->mapEntryCount = 0;
        outSpecInfo->dataSize = std::distance<const uint8_t*>(specDataBegin,outSpecData);
        retval.pSpecializationInfo = outSpecInfo++;
    }
    return retval;
}

template<typename VkPipelineCreateInfo_t, typename SCreationParams>
void initPipelineCreateInfo(VkPipelineCreateInfo_t* vk_info, const SCreationParams& info)
{
    // the new flags type (64bit) is only available with maintenance5
    vk_info->flags = static_cast<VkPipelineCreateFlags>(info.flags.value);
    vk_info->layout = static_cast<const CVulkanPipelineLayout*>(info.layout)->getInternalObject();
    if (info.isDerivative())
    {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html#VUID-VkComputePipelineCreateInfo-flags-07984
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html#VUID-VkComputePipelineCreateInfo-flags-07986
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCreateGraphicsPipelines.html#VUID-vkCreateGraphicsPipelines-flags-00720
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCreateGraphicsPipelines.html#VUID-vkCreateGraphicsPipelines-flags-00721
        vk_info->basePipelineHandle = info.basePipeline ? *reinterpret_cast<const VkPipeline*>(info.basePipeline->getNativeHandle()):VK_NULL_HANDLE;
        vk_info->basePipelineIndex = info.basePipeline ? (-1):info.basePipelineIndex;
        vk_info->flags |= VK_PIPELINE_CREATE_DERIVATIVE_BIT;
    }
}

void CVulkanLogicalDevice::createComputePipelines_impl(
    IGPUPipelineCache* const pipelineCache,
    const std::span<const IGPUComputePipeline::SCreationParams> createInfos,
    core::smart_refctd_ptr<IGPUComputePipeline>* const output,
    const IGPUComputePipeline::SCreationParams::SSpecializationValidationResult& validation
)
{
    const VkPipelineCache vk_pipelineCache = pipelineCache ? static_cast<const CVulkanPipelineCache*>(pipelineCache)->getInternalObject():VK_NULL_HANDLE;
    
    // pNext can only be VkComputePipelineIndirectBufferInfoNV, creation feedback, robustness and VkPipelineCreateFlags2CreateInfoKHR
    core::vector<VkComputePipelineCreateInfo> vk_createInfos(createInfos.size(),{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,nullptr});
    core::vector<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo> vk_requiredSubgroupSize(createInfos.size(),{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,nullptr
    });
    core::vector<VkSpecializationInfo> vk_specializationInfos(createInfos.size(),{0,nullptr,0,nullptr});
    core::vector<VkSpecializationMapEntry> vk_specializationMapEntry(validation.count);
    core::vector<uint8_t> specializationData(validation.dataSize);

    auto outCreateInfo = vk_createInfos.data();
    auto outRequiredSubgroupSize = vk_requiredSubgroupSize.data();
    auto outSpecInfo = vk_specializationInfos.data();
    auto outSpecMapEntry = vk_specializationMapEntry.data();
    auto outSpecData = specializationData.data();
    for (const auto& info : createInfos)
    {
        initPipelineCreateInfo(outCreateInfo,info);
        outCreateInfo->stage = getVkShaderStageCreateInfoFrom(info.shader,outRequiredSubgroupSize,outSpecInfo,outSpecMapEntry,outSpecData);
        outCreateInfo++;
    }
    auto vk_pipelines = reinterpret_cast<VkPipeline*>(output);
    if (m_devf.vk.vkCreateComputePipelines(m_vkdev,vk_pipelineCache,vk_createInfos.size(),vk_createInfos.data(),nullptr,vk_pipelines)==VK_SUCCESS)
    {
        for (size_t i=0ull; i<createInfos.size(); ++i)
        {
            const auto& info = createInfos[i];
            const VkPipeline vk_pipeline = vk_pipelines[i];
            // break the lifetime cause of the aliasing
            std::uninitialized_default_construct_n(output+i,1);
            output[i] = core::make_smart_refctd_ptr<CVulkanComputePipeline>(
                core::smart_refctd_ptr<const IGPUPipelineLayout>(info.layout),
                core::smart_refctd_ptr<const CVulkanShader>(static_cast<const CVulkanShader*>(info.shader.shader)),
                info.flags,vk_pipeline
            );
        }
    }
    else
        std::fill_n(output,vk_createInfos.size(),nullptr);
}

void CVulkanLogicalDevice::createGraphicsPipelines_impl(
    IGPUPipelineCache* const pipelineCache,
    const std::span<const IGPUGraphicsPipeline::SCreationParams> createInfos,
    core::smart_refctd_ptr<IGPUGraphicsPipeline>* const output,
    const IGPUGraphicsPipeline::SCreationParams::SSpecializationValidationResult& validation
)
{
    auto getVkStencilOpStateFrom = [](const asset::SStencilOpParams& params)->VkStencilOpState
    {
        return {
            .failOp = static_cast<VkStencilOp>(params.failOp),
            .passOp = static_cast<VkStencilOp>(params.passOp),
            .depthFailOp = static_cast<VkStencilOp>(params.depthFailOp),
            .compareOp = static_cast<VkCompareOp>(params.compareOp)
        };
    };

    const auto& features = getEnabledFeatures();

    core::vector<VkDynamicState> vk_dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_LINE_WIDTH,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE
    };
    if (features.depthBounds)
        vk_dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS);
    // TODO: VK_DYNAMIC_STATE_DISCARD_RECTANGLE_EXT, VK_DYNAMIC_STATE_DISCARD_RECTANGLE_ENABLE_EXT, VK_DYNAMIC_STATE_DISCARD_RECTANGLE_MODE_EXT
    
    const VkPipelineDynamicStateCreateInfo vk_dynamicStateCreateInfo = { 
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0u,
        .dynamicStateCount = static_cast<uint32_t>(vk_dynamicStates.size()),
        .pDynamicStates = vk_dynamicStates.data()
    };

    const VkPipelineCache vk_pipelineCache = pipelineCache ? static_cast<const CVulkanPipelineCache*>(pipelineCache)->getInternalObject():VK_NULL_HANDLE;
    // Interesting things to put in pNext:
    // - AttachmentSampleCountInfoAMD
    // - Graphics Pipeline Library styff
    // - CreateFlags 2
    // - Pipeline Creation Feedback
    // - Discard Rectangle State
    // - Fragment Shading Rate State Creation Info
    // - Piepline Robustness 
    core::vector<VkGraphicsPipelineCreateInfo> vk_createInfos(createInfos.size(),{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,nullptr});

    const auto maxShaderStages = createInfos.size()*IGPUGraphicsPipeline::GRAPHICS_SHADER_STAGE_COUNT;
    core::vector<VkPipelineShaderStageCreateInfo> vk_shaderStage(maxShaderStages,{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr});
    core::vector<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo> vk_requiredSubgroupSize(maxShaderStages,{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,nullptr
    });
    core::vector<VkSpecializationInfo> vk_specializationInfos(maxShaderStages,{0,nullptr,0,nullptr});
    core::vector<VkSpecializationMapEntry> vk_specializationMapEntry(validation.count);
    core::vector<uint8_t> specializationData(validation.dataSize);
    core::vector<VkPipelineVertexInputStateCreateInfo> vk_vertexInput(createInfos.size(),{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,nullptr,0});
    core::vector<VkVertexInputBindingDescription> vk_inputBinding(createInfos.size()*asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT);
    core::vector<VkVertexInputAttributeDescription> vk_inputAttribute(createInfos.size()*asset::SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT);
    core::vector<VkPipelineInputAssemblyStateCreateInfo> vk_inputAssembly(createInfos.size(),{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,nullptr,0});
    core::vector<VkPipelineTessellationStateCreateInfo> vk_tessellation(createInfos.size(),{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,nullptr,0});
    core::vector<VkPipelineViewportStateCreateInfo> vk_viewportStates(createInfos.size(),{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr, // the extensions that interest us have a dynamic state variant anyway
        .flags = 0, // must be 0
        .viewportCount = 0,
        .pViewports = nullptr,
        .scissorCount = 0,
        .pScissors = nullptr,
    });
    core::vector<VkPipelineRasterizationStateCreateInfo> vk_rasterizationStates(createInfos.size(),{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,nullptr,0});
    core::vector<VkPipelineMultisampleStateCreateInfo> vk_multisampleStates(createInfos.size(),{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,nullptr,0});
    core::vector<VkPipelineDepthStencilStateCreateInfo> vk_depthStencilStates(createInfos.size(),{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,nullptr,0});
    core::vector<VkPipelineColorBlendStateCreateInfo> vk_colorBlendStates(createInfos.size(),{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,nullptr,0});
    core::vector<VkPipelineColorBlendAttachmentState> vk_colorBlendAttachmentStates(createInfos.size()*IGPURenderpass::SCreationParams::SSubpassDescription::MaxColorAttachments);

    auto outCreateInfo = vk_createInfos.data();
    auto outShaderStage = vk_shaderStage.data();
    auto outRequiredSubgroupSize = vk_requiredSubgroupSize.data();
    auto outSpecInfo = vk_specializationInfos.data();
    auto outSpecMapEntry = vk_specializationMapEntry.data();
    auto outSpecData = specializationData.data();
    auto outVertexInput = vk_vertexInput.data();
    auto outInputBinding = vk_inputBinding.data();
    auto outInputAttribute = vk_inputAttribute.data();
    auto outInputAssembly = vk_inputAssembly.data();
    auto outTessellation = vk_tessellation.data();
    auto outViewport = vk_viewportStates.data();
    auto outRaster = vk_rasterizationStates.data();
    auto outMultisample = vk_multisampleStates.data();
    auto outDepthStencil = vk_depthStencilStates.data();
    auto outColorBlend = vk_colorBlendStates.data();
    auto outColorBlendAttachmentState = vk_colorBlendAttachmentStates.data();
    for (const auto& info : createInfos)
    {
        initPipelineCreateInfo(outCreateInfo,info);
        outCreateInfo->pStages = outShaderStage;
        for (const auto& spec : info.shaders)
        if (spec.shader)
            *(outShaderStage++) = getVkShaderStageCreateInfoFrom(spec,outRequiredSubgroupSize,outSpecInfo,outSpecMapEntry,outSpecData);
        outCreateInfo->stageCount = std::distance<decltype(outCreateInfo->pStages)>(outCreateInfo->pStages,outShaderStage);
        // when dealing with mesh shaders, the vertex input and assembly state will be null
        {
            {
                const auto& vertexInputParams = info.cached.vertexInput;
                outVertexInput->pVertexBindingDescriptions = outInputBinding;
                for (auto b=0u; b<asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; b++)
                if (vertexInputParams.enabledBindingFlags&(1<<b))
                {
                    outInputBinding->binding = b;
                    outInputBinding->stride = vertexInputParams.bindings[b].stride;
                    outInputBinding->inputRate = static_cast<VkVertexInputRate>(vertexInputParams.bindings[b].inputRate);
                    outInputBinding++;
                }
                outVertexInput->vertexBindingDescriptionCount = std::distance<const VkVertexInputBindingDescription*>(outVertexInput->pVertexBindingDescriptions,outInputBinding);
                outVertexInput->pVertexAttributeDescriptions = outInputAttribute;
                for (auto l=0u; l<asset::SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; l++)
                if (vertexInputParams.enabledAttribFlags&(1<<l))
                {
                    outInputAttribute->location = l;
                    outInputAttribute->binding = vertexInputParams.attributes[l].binding;
                    outInputAttribute->format = getVkFormatFromFormat(static_cast<asset::E_FORMAT>(vertexInputParams.attributes[l].format));
                    outInputAttribute->offset = vertexInputParams.attributes[l].relativeOffset;
                    outInputAttribute++;
                }
                outVertexInput->vertexAttributeDescriptionCount = std::distance<const VkVertexInputAttributeDescription*>(outVertexInput->pVertexAttributeDescriptions,outInputAttribute);
            }
            outCreateInfo->pVertexInputState = outVertexInput++;
            {
                const auto& primAssParams = info.cached.primitiveAssembly;
                outInputAssembly->topology = static_cast<VkPrimitiveTopology>(primAssParams.primitiveType);
                outInputAssembly->primitiveRestartEnable = primAssParams.primitiveRestartEnable;
            }
            outCreateInfo->pInputAssemblyState = outInputAssembly++;
        }
        for (const auto& spec : info.shaders)
        if (spec.shader)
        {
            const auto stage = spec.shader->getStage();
            if (stage==IGPUShader::ESS_TESSELLATION_CONTROL || stage==IGPUShader::ESS_TESSELLATION_EVALUATION)
            {
                outTessellation->patchControlPoints = info.cached.primitiveAssembly.tessPatchVertCount;
                outCreateInfo->pTessellationState = outTessellation++;
                break;
            }
        }
        const auto& raster = info.cached.rasterization;
        {
            outViewport->viewportCount = raster.viewportCount;
            // must be identical to viewport count unless VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT_EXT or VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT_EXT are used
            outViewport->scissorCount = raster.viewportCount;
            outCreateInfo->pViewportState = outViewport++;
        }
        {
            outRaster->depthClampEnable = raster.depthClampEnable;
            outRaster->rasterizerDiscardEnable = raster.rasterizerDiscard;
            outRaster->polygonMode = static_cast<VkPolygonMode>(raster.polygonMode);
            outRaster->cullMode = static_cast<VkCullModeFlags>(raster.faceCullingMode);
            outRaster->frontFace = raster.frontFaceIsCCW ? VK_FRONT_FACE_COUNTER_CLOCKWISE:VK_FRONT_FACE_CLOCKWISE;
            outRaster->depthBiasEnable = raster.depthBiasEnable;
            outCreateInfo->pRasterizationState = outRaster++;
        }
        {
            outMultisample->rasterizationSamples = static_cast<VkSampleCountFlagBits>(0x1<<raster.samplesLog2);
            if (raster.minSampleShadingUnorm>0)
            {
                outMultisample->sampleShadingEnable = true;
                outMultisample->minSampleShading = float(raster.minSampleShadingUnorm)/255.f;
            }
            else
            {
                outMultisample->sampleShadingEnable = false;
                outMultisample->minSampleShading = 0.f;
            }
            outMultisample->pSampleMask = raster.sampleMask;
            outMultisample->alphaToCoverageEnable = raster.alphaToCoverageEnable;
            outMultisample->alphaToOneEnable = raster.alphaToOneEnable;
            outCreateInfo->pMultisampleState = outMultisample++;
        }
        {
            //outDepthStencil->flags no attachment order access yet
            outDepthStencil->depthTestEnable = raster.depthTestEnable();
            outDepthStencil->depthWriteEnable = raster.depthWriteEnable;
            outDepthStencil->depthCompareOp = static_cast<VkCompareOp>(raster.depthCompareOp);
            outDepthStencil->depthBoundsTestEnable = raster.depthBoundsTestEnable;
            outDepthStencil->stencilTestEnable = raster.stencilTestEnable();
            outDepthStencil->front = getVkStencilOpStateFrom(raster.frontStencilOps);
            outDepthStencil->back = getVkStencilOpStateFrom(raster.backStencilOps);
            outCreateInfo->pDepthStencilState = outDepthStencil++;
        }
        {
            const auto& blend = info.cached.blend;
            const auto& subpass = info.renderpass->getCreationParameters().subpasses[info.cached.subpassIx];
            //outColorBlend->flags no attachment order access yet
            outColorBlend->logicOpEnable = blend.logicOp!=asset::ELO_NO_OP;
            outColorBlend->logicOp = getVkLogicOpFromLogicOp(blend.logicOp);
            outColorBlend->pAttachments = outColorBlendAttachmentState;
            for (auto i=0; i<IGPURenderpass::SCreationParams::SSubpassDescription::MaxColorAttachments; i++)
            if (subpass.colorAttachments[i].render.used())
            {
                const auto& params = blend.blendParams[i];
                outColorBlendAttachmentState->blendEnable = params.blendEnabled();
                outColorBlendAttachmentState->srcColorBlendFactor = getVkBlendFactorFromBlendFactor(static_cast<asset::E_BLEND_FACTOR>(params.srcColorFactor));
                outColorBlendAttachmentState->dstColorBlendFactor = getVkBlendFactorFromBlendFactor(static_cast<asset::E_BLEND_FACTOR>(params.dstColorFactor));
                outColorBlendAttachmentState->colorBlendOp = getVkBlendOpFromBlendOp(static_cast<asset::E_BLEND_OP>(params.colorBlendOp));
                outColorBlendAttachmentState->srcAlphaBlendFactor = getVkBlendFactorFromBlendFactor(static_cast<asset::E_BLEND_FACTOR>(params.srcAlphaFactor));
                outColorBlendAttachmentState->dstAlphaBlendFactor = getVkBlendFactorFromBlendFactor(static_cast<asset::E_BLEND_FACTOR>(params.dstAlphaFactor));
                outColorBlendAttachmentState->alphaBlendOp = getVkBlendOpFromBlendOp(static_cast<asset::E_BLEND_OP>(params.alphaBlendOp));
                outColorBlendAttachmentState->colorWriteMask = getVkColorComponentFlagsFromColorWriteMask(params.colorWriteMask);
                outColorBlendAttachmentState++;
            }
            outColorBlend->attachmentCount = std::distance<const VkPipelineColorBlendAttachmentState*>(outColorBlend->pAttachments,outColorBlendAttachmentState);
            outCreateInfo->pColorBlendState = outColorBlend++;
        }
        outCreateInfo->pDynamicState = &vk_dynamicStateCreateInfo;
        outCreateInfo->renderPass = static_cast<const CVulkanRenderpass*>(info.renderpass)->getInternalObject();
        outCreateInfo->subpass = info.cached.subpassIx;
        outCreateInfo++;
    }
    
    auto vk_pipelines = reinterpret_cast<VkPipeline*>(output);
    if (m_devf.vk.vkCreateGraphicsPipelines(m_vkdev,vk_pipelineCache,vk_createInfos.size(),vk_createInfos.data(),nullptr,vk_pipelines)==VK_SUCCESS)
    {
        for (size_t i=0ull; i<createInfos.size(); ++i)
        {
            const VkPipeline vk_pipeline = vk_pipelines[i];
            // break the lifetime cause of the aliasing
            std::uninitialized_default_construct_n(output+i,1);
            output[i] = core::make_smart_refctd_ptr<CVulkanGraphicsPipeline>(createInfos[i],vk_pipeline);
        }
    }
    else
        std::fill_n(output,vk_createInfos.size(),nullptr);
}

core::smart_refctd_ptr<IQueryPool> CVulkanLogicalDevice::createQueryPool_impl(const IQueryPool::SCreationParams& params)
{
    VkQueryPoolCreateInfo info =  {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr};
    info.flags = 0; // "flags is reserved for future use."
    info.queryType = CVulkanQueryPool::getVkQueryTypeFrom(params.queryType);
    info.queryCount = params.queryCount;
    info.pipelineStatistics = CVulkanQueryPool::getVkPipelineStatisticsFlagsFrom(params.pipelineStatisticsFlags.value);

    VkQueryPool vk_queryPool = VK_NULL_HANDLE;
    if (m_devf.vk.vkCreateQueryPool(m_vkdev,&info,nullptr,&vk_queryPool)!=VK_SUCCESS)
        return core::make_smart_refctd_ptr<CVulkanQueryPool>(this,params,vk_queryPool);
    return nullptr;
}

bool CVulkanLogicalDevice::getQueryPoolResults_impl(const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount, void* const pData, const size_t stride, const core::bitflag<IQueryPool::RESULTS_FLAGS> flags)
{
    auto pseudoParams = queryPool->getCreationParameters();
    pseudoParams.queryCount = queryCount;
    const size_t dataSize = IQueryPool::calcQueryResultsSize(pseudoParams,stride,flags);
    const auto vk_queryResultsflags = CVulkanQueryPool::getVkQueryResultsFlagsFrom(flags.value);
    return m_devf.vk.vkGetQueryPoolResults(m_vkdev,static_cast<const CVulkanQueryPool*>(queryPool)->getInternalObject(),firstQuery,queryCount,dataSize,pData,stride,vk_queryResultsflags)==VK_SUCCESS;
}

core::smart_refctd_ptr<IGPUCommandPool> CVulkanLogicalDevice::createCommandPool_impl(const uint32_t familyIx, const core::bitflag<IGPUCommandPool::CREATE_FLAGS> flags)
{
    VkCommandPoolCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    vk_createInfo.pNext = nullptr; // pNext must be NULL
    vk_createInfo.flags = static_cast<VkCommandPoolCreateFlags>(flags.value);
    vk_createInfo.queueFamilyIndex = familyIx;

    VkCommandPool vk_commandPool = VK_NULL_HANDLE;
    if (m_devf.vk.vkCreateCommandPool(m_vkdev,&vk_createInfo,nullptr,&vk_commandPool)==VK_SUCCESS)
        return core::make_smart_refctd_ptr<CVulkanCommandPool>(core::smart_refctd_ptr<const CVulkanLogicalDevice>(this),flags,familyIx,vk_commandPool);
    return nullptr;
}