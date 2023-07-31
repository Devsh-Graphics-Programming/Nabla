#include "nbl/video/CVulkanSwapchain.h"

#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanConnection.h"

namespace nbl::video
{

core::smart_refctd_ptr<CVulkanSwapchain> CVulkanSwapchain::create(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params)
{ 
    if (!logicalDevice || logicalDevice->getAPIType()!=EAT_VULKAN || logicalDevice->getEnabledFeatures().swapchainMode==ESM_NONE)
        return nullptr;

    if (!params.surface)
        return nullptr;
    // now check if any queue family of the physical device supports this surface
    {
        auto physDev = logicalDevice->getPhysicalDevice();
        const uint32_t queueFamilyCount = physDev->getQueueFamilyProperties().size();
        uint32_t firstSupportIx = 0;
        while (firstSupportIx<queueFamilyCount && !params.surface->isSupportedForPhysicalDevice(physDev,firstSupportIx))
            firstSupportIx++;
        if (firstSupportIx==queueFamilyCount)
            return nullptr;
    }

    auto device = core::smart_refctd_ptr_static_cast<CVulkanLogicalDevice>(logicalDevice);
    const VkSurfaceKHR vk_surface = static_cast<const ISurfaceVulkan*>(params.surface.get())->getInternalObject();

    // get present mode
    VkPresentModeKHR vkPresentMode;
    if ((params.presentMode & ISurface::E_PRESENT_MODE::EPM_IMMEDIATE) == ISurface::E_PRESENT_MODE::EPM_IMMEDIATE)
        vkPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    else if ((params.presentMode & ISurface::E_PRESENT_MODE::EPM_MAILBOX) == ISurface::E_PRESENT_MODE::EPM_MAILBOX)
        vkPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    else if ((params.presentMode & ISurface::E_PRESENT_MODE::EPM_FIFO) == ISurface::E_PRESENT_MODE::EPM_FIFO)
        vkPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    else if ((params.presentMode & ISurface::E_PRESENT_MODE::EPM_FIFO_RELAXED) == ISurface::E_PRESENT_MODE::EPM_FIFO_RELAXED)
        vkPresentMode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
    
    //! fill format list for mutable formats
    VkImageFormatListCreateInfo vk_formatListStruct = { VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, nullptr };
    vk_formatListStruct.viewFormatCount = 0u;
    std::array<VkFormat,asset::E_FORMAT::EF_COUNT> vk_formatList;

    VkSwapchainCreateInfoKHR vk_createInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, &vk_formatListStruct };
    // if only there existed a nice iterator that would let me iterate over set bits 64 faster
    if (params.viewFormats.any())
    {
        // structure with a viewFormatCount greater than zero and pViewFormats must have an element equal to imageFormat
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSwapchainCreateInfoKHR.html#VUID-VkSwapchainCreateInfoKHR-flags-03168
        if (!params.viewFormats.test(params.surfaceFormat.format))
            return nullptr;

        for (auto fmt=0; fmt<vk_formatList.size(); fmt++)
        if (params.viewFormats.test(fmt))
        {
            const auto format = static_cast<asset::E_FORMAT>(fmt);
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSwapchainCreateInfoKHR.html#VUID-VkSwapchainCreateInfoKHR-pNext-04099
            if (asset::getFormatClass(format) != asset::getFormatClass(params.surfaceFormat.format))
                return nullptr;
            vk_formatList[vk_formatListStruct.viewFormatCount++] = getVkFormatFromFormat(format);
        }

        // just deduce the mutable flag, cause we're so constrained by spec, it dictates we must list all formats if we're doing mutable
        if (vk_formatListStruct.viewFormatCount > 1)
            vk_createInfo.flags |= VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR;
    }
    vk_formatListStruct.pViewFormats = vk_formatList.data();

    vk_createInfo.surface = vk_surface;
    vk_createInfo.minImageCount = params.minImageCount;
    vk_createInfo.imageFormat = getVkFormatFromFormat(params.surfaceFormat.format);
    vk_createInfo.imageColorSpace = getVkColorSpaceKHRFromColorSpace(params.surfaceFormat.colorSpace);
    vk_createInfo.imageExtent = {params.width,params.height};
    vk_createInfo.imageArrayLayers = params.arrayLayers;
    vk_createInfo.imageUsage = getVkImageUsageFlagsFromImageUsageFlags(params.imageUsage,asset::isDepthOrStencilFormat(params.surfaceFormat.format));
    vk_createInfo.imageSharingMode = params.isConcurrentSharing() ? VK_SHARING_MODE_CONCURRENT:VK_SHARING_MODE_EXCLUSIVE;
    vk_createInfo.queueFamilyIndexCount = params.queueFamilyIndexCount;
    vk_createInfo.pQueueFamilyIndices = params.queueFamilyIndices;
    vk_createInfo.preTransform = static_cast<VkSurfaceTransformFlagBitsKHR>(params.preTransform);
    vk_createInfo.compositeAlpha = static_cast<VkCompositeAlphaFlagBitsKHR>(params.compositeAlpha);
    vk_createInfo.presentMode = vkPresentMode;
    vk_createInfo.clipped = VK_FALSE;
    if (params.oldSwapchain && params.oldSwapchain->wasCreatedBy(device.get()))
        vk_createInfo.oldSwapchain = IBackendObject::device_compatibility_cast<CVulkanSwapchain*>(params.oldSwapchain.get(),device.get())->getInternalObject();
    else
        vk_createInfo.oldSwapchain = VK_NULL_HANDLE;

    auto& vk = device->getFunctionTable()->vk;

    VkSwapchainKHR vk_swapchain;
    if (vk.vkCreateSwapchainKHR(device->getInternalObject(),&vk_createInfo,nullptr,&vk_swapchain)!=VK_SUCCESS)
        return nullptr;

    uint32_t imageCount;
    VkResult retval = vk.vkGetSwapchainImagesKHR(device->getInternalObject(), vk_swapchain, &imageCount, nullptr);
    if (retval!=VK_SUCCESS)
        return nullptr;

    //
    VkSemaphore adaptorSemaphores[2*ISwapchain::MaxImages];
    {
        VkSemaphoreCreateInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,nullptr};
        info.flags = 0u;
        for (auto i=0u; i<2*imageCount; i++)
        if (vk.vkCreateSemaphore(device->getInternalObject(),&info,nullptr,adaptorSemaphores+i)!=VK_SUCCESS)
        {
            // handle successful allocs before failure
            for (auto j=0u; j<i; j++)
                vk.vkDestroySemaphore(device->getInternalObject(),adaptorSemaphores[j],nullptr);
            return nullptr;
        }
    }

    return core::make_smart_refctd_ptr<CVulkanSwapchain>(std::move(device),std::move(params),imageCount,vk_swapchain,adaptorSemaphores);
}

CVulkanSwapchain::CVulkanSwapchain(core::smart_refctd_ptr<const ILogicalDevice>&& logicalDevice, SCreationParams&& params, const uint32_t imageCount, const VkSwapchainKHR swapchain, const VkSemaphore* const _adaptorSemaphores)
    : ISwapchain(std::move(logicalDevice), std::move(params), imageCount), m_vkSwapchainKHR(swapchain)
{
    m_imgMemRequirements.size = 0ull;
    m_imgMemRequirements.memoryTypeBits = 0x0u;
    m_imgMemRequirements.alignmentLog2 = 63u;
    m_imgMemRequirements.prefersDedicatedAllocation = true;
    m_imgMemRequirements.requiresDedicatedAllocation = true;

    {
        const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
        uint32_t dummy = imageCount;
        auto retval = vulkanDevice->getFunctionTable()->vk.vkGetSwapchainImagesKHR(vulkanDevice->getInternalObject(),m_vkSwapchainKHR,&dummy,m_images);
        assert(retval==VK_SUCCESS && dummy==m_imageCount);
    }

    std::copy_n(_adaptorSemaphores,imageCount,m_adaptorSemaphores);
}

CVulkanSwapchain::~CVulkanSwapchain()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto& vk = vulkanDevice->getFunctionTable()->vk;

    for (auto i=0u; i<2*m_imageCount; i++)
        vk.vkDestroySemaphore(vulkanDevice->getInternalObject(),m_adaptorSemaphores[i],nullptr);

    vk.vkDestroySwapchainKHR(vulkanDevice->getInternalObject(), m_vkSwapchainKHR, nullptr);
}


auto CVulkanSwapchain::acquireNextImage_impl(const SAcquireInfo& info, uint32_t* const out_imgIx) -> ACQUIRE_IMAGE_RESULT
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    CVulkanQueue* vulkanQueue;
    if (info.signalSemaphoreCount)
    {
        vulkanQueue = IBackendObject::device_compatibility_cast<CVulkanQueue*>(info.queue,vulkanDevice);
        if (!vulkanQueue)
            return ACQUIRE_IMAGE_RESULT::_ERROR;
        for (auto i=0u; i<info.signalSemaphoreCount; i++)
        if (!IBackendObject::device_compatibility_cast<CVulkanSemaphore*>(info.signalSemaphores[i].semaphore,vulkanDevice))
            return ACQUIRE_IMAGE_RESULT::_ERROR;
    }

    auto& vk = vulkanDevice->getFunctionTable()->vk;
    const VkDevice vk_device = vulkanDevice->getInternalObject();
    const VkSemaphoreSubmitInfoKHR adaptorInfo = getAdaptorSemaphore(info.signalSemaphoreCount);
    {
        VkAcquireNextImageInfoKHR acquire = { VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR,nullptr };
        acquire.swapchain = m_vkSwapchainKHR;
        acquire.timeout = info.timeout;
        acquire.semaphore = adaptorInfo.semaphore;
        acquire.fence = VK_NULL_HANDLE;
        acquire.deviceMask = 0x1u<<adaptorInfo.deviceIndex;
        switch (vk.vkAcquireNextImage2KHR(vk_device,&acquire,out_imgIx))
        {
            case VK_SUCCESS:
                break;
            case VK_TIMEOUT:
                return ACQUIRE_IMAGE_RESULT::TIMEOUT;
            case VK_NOT_READY:
                return ACQUIRE_IMAGE_RESULT::NOT_READY;
            case VK_SUBOPTIMAL_KHR:
                return ACQUIRE_IMAGE_RESULT::SUBOPTIMAL;
            default:
                return ACQUIRE_IMAGE_RESULT::_ERROR;
        }
    }

    if (info.signalSemaphoreCount)
    {
        core::vector<VkSemaphoreSubmitInfoKHR> signalInfos(info.signalSemaphoreCount,{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,nullptr});
        for (auto i=0u; i<info.signalSemaphoreCount; i++)
        {
            signalInfos[i].semaphore = IBackendObject::device_compatibility_cast<CVulkanSemaphore*>(info.signalSemaphores[i].semaphore,vulkanDevice)->getInternalObject();
            signalInfos[i].stageMask = getVkPipelineStageFlagsFromPipelineStageFlags(info.signalSemaphores[i].stageMask);
            signalInfos[i].value = info.signalSemaphores[i].value;
            signalInfos[i].deviceIndex = 0u;
        }

        VkSubmitInfo2KHR submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,nullptr };
        submit.waitSemaphoreInfoCount = 1u;
        submit.pWaitSemaphoreInfos = &adaptorInfo;
        submit.commandBufferInfoCount = 0u;
        submit.signalSemaphoreInfoCount = info.signalSemaphoreCount;
        submit.pSignalSemaphoreInfos = signalInfos.data();
        const bool result = vk.vkQueueSubmit2KHR(vulkanQueue->getInternalObject(),1u,&submit,VK_NULL_HANDLE)==VK_SUCCESS;
        // if this goes wrong, we are fucked without KHR_swapchain_maintenance1 because there's no way to release acquired images without presenting them!
        assert(result);
    }
    return ACQUIRE_IMAGE_RESULT::SUCCESS;
}

auto CVulkanSwapchain::present_impl(const SPresentInfo& info) -> PRESENT_RESULT
{    
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto vulkanQueue = IBackendObject::device_compatibility_cast<CVulkanQueue*>(info.queue,vulkanDevice);
    if (!vulkanQueue)
        return PRESENT_RESULT::_ERROR;

    auto& vk = vulkanDevice->getFunctionTable()->vk;
    const VkSemaphoreSubmitInfoKHR adaptorInfo = getAdaptorSemaphore(info.waitSemaphoreCount);
    if (info.waitSemaphoreCount)
    {
        core::vector<VkSemaphoreSubmitInfoKHR> waitInfos(info.waitSemaphoreCount,{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,nullptr});
        for (auto i=0u; i<info.waitSemaphoreCount; i++)
        {
            auto sema = IBackendObject::device_compatibility_cast<CVulkanSemaphore*>(info.waitSemaphores[i].semaphore,vulkanDevice);
            if (!sema)
                return PRESENT_RESULT::_ERROR;
            waitInfos[i].semaphore = sema->getInternalObject();
            waitInfos[i].stageMask = getVkPipelineStageFlagsFromPipelineStageFlags(info.waitSemaphores[i].stageMask);
            waitInfos[i].value = info.waitSemaphores[i].value;
            waitInfos[i].deviceIndex = 0u;
        }

        VkSubmitInfo2KHR submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,nullptr };
        submit.waitSemaphoreInfoCount = info.waitSemaphoreCount;
        submit.pWaitSemaphoreInfos = waitInfos.data();
        submit.commandBufferInfoCount = 0u;
        submit.signalSemaphoreInfoCount = 1u;
        submit.pSignalSemaphoreInfos = &adaptorInfo;
        if (vk.vkQueueSubmit2KHR(vulkanQueue->getInternalObject(),1u,&submit,VK_NULL_HANDLE)!=VK_SUCCESS)
            return PRESENT_RESULT::_ERROR;
    }

    VkPresentInfoKHR presentInfo = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,nullptr };
    presentInfo.waitSemaphoreCount = info.waitSemaphoreCount ? 1u:0u;
    presentInfo.pWaitSemaphores = &adaptorInfo.semaphore;
    presentInfo.swapchainCount = 1u;
    presentInfo.pSwapchains = &m_vkSwapchainKHR;
    presentInfo.pImageIndices = &info.imgIndex;

    VkResult retval = vk.vkQueuePresentKHR(vulkanQueue->getInternalObject(),&presentInfo);
    switch (retval)
    {
        case VK_SUCCESS:
            break;
        case VK_SUBOPTIMAL_KHR:
            return PRESENT_RESULT::SUBOPTIMAL;
        default:
            return PRESENT_RESULT::_ERROR;
    }
    return PRESENT_RESULT::SUCCESS;
}

core::smart_refctd_ptr<IGPUImage> CVulkanSwapchain::createImage(const uint32_t imageIndex)
{
    if (!setImageExists(imageIndex))
        return nullptr;

    // TODO figure out image create path
    // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/4343
    // for (uint32_t i = 0; i < imageCount; i++)
    // {
    //     VkImageSwapchainCreateInfoKHR vk_imgSwapchainCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_SWAPCHAIN_CREATE_INFO_KHR };
    //     vk_imgSwapchainCreateInfo.pNext = nullptr;
    //     vk_imgSwapchainCreateInfo.swapchain = vk_swapchain;
    // 
    //     // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#swapchain-wsi-image-create-info
    //     VkImageCreateInfo vk_imgCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    //     vk_imgCreateInfo.pNext = &vk_imgSwapchainCreateInfo; // there are a lot of extensions
    //     vk_imgCreateInfo.flags = static_cast<VkImageCreateFlags>(0);
    //     vk_imgCreateInfo.imageType = static_cast<VkImageType>(VK_IMAGE_TYPE_2D);
    //     vk_imgCreateInfo.format = getVkFormatFromFormat(params.surfaceFormat.format);
    //     vk_imgCreateInfo.extent = { params.width, params.height, 1 };
    //     vk_imgCreateInfo.mipLevels = 1;
    //     vk_imgCreateInfo.arrayLayers = params.arrayLayers;
    //     vk_imgCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    //     vk_imgCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    //     vk_imgCreateInfo.usage = static_cast<VkImageUsageFlags>(params.imageUsage.value);
    //     vk_imgCreateInfo.sharingMode = static_cast<VkSharingMode>(params.imageSharingMode);
    //     vk_imgCreateInfo.queueFamilyIndexCount = params.queueFamilyIndexCount;
    //     vk_imgCreateInfo.pQueueFamilyIndices = params.queueFamilyIndices;
    //     vk_imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // 
    //     device->getFunctionTable()->vk.vkCreateImage(device->getInternalObject(), &vk_imgCreateInfo, nullptr, &vk_images[i]);
    // 
    //     VkBindImageMemorySwapchainInfoKHR vk_bindImgMemorySwapchainInfo = { VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHR };
    //     vk_bindImgMemorySwapchainInfo.pNext = nullptr;
    //     vk_bindImgMemorySwapchainInfo.imageIndex = i;
    //     vk_bindImgMemorySwapchainInfo.swapchain = vk_swapchain;
    // 
    //     VkBindImageMemoryInfo vk_bindImgMemoryInfo = { VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO };
    //     vk_bindImgMemoryInfo.pNext = &vk_bindImgMemorySwapchainInfo;
    //     vk_bindImgMemoryInfo.image = vk_images[i];
    //     vk_bindImgMemoryInfo.memory = nullptr;
    // 
    //     device->getFunctionTable()->vk.vkBindImageMemory2(device->getInternalObject(), 1, &vk_bindImgMemoryInfo);
    //     assert(vk_images[i]);
    // }

    auto device = core::smart_refctd_ptr<const CVulkanLogicalDevice>(static_cast<const CVulkanLogicalDevice*>(getOriginDevice()));

    IGPUImage::SCreationParams creationParams = {};
    static_cast<asset::IImage::SCreationParams&>(creationParams) = m_imgCreationParams;

    m_imgCreationParams.preDestroyCleanup = std::make_unique<CCleanupSwapchainReference>(core::smart_refctd_ptr<ISwapchain>(this),imageIndex);
    m_imgCreationParams.skipHandleDestroy = true;
    m_imgCreationParams.queueFamilyIndexCount = m_params.queueFamilyIndexCount;
    m_imgCreationParams.queueFamilyIndices = m_params.queueFamilyIndices;

    return core::make_smart_refctd_ptr<CVulkanImage>(core::smart_refctd_ptr<const ILogicalDevice>(getOriginDevice()),m_imgMemRequirements,std::move(creationParams),m_images[imageIndex]);
}

void CVulkanSwapchain::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

    if (!vkSetDebugUtilsObjectNameEXT)
        return;

    VkDebugUtilsObjectNameInfoEXT nameInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr };
    nameInfo.objectType = VK_OBJECT_TYPE_SWAPCHAIN_KHR;
    nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
    nameInfo.pObjectName = getObjectDebugName();
    vkSetDebugUtilsObjectNameEXT(static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getInternalObject(), &nameInfo);
}

}
