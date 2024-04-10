#include "nbl/video/ISwapchain.h"

#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanConnection.h"

namespace nbl::video
{
	
ISwapchain::ISwapchain(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params, const uint8_t imageCount, core::smart_refctd_ptr<ISwapchain>&& oldSwapchain) :
    IBackendObject(std::move(dev)), m_params(std::move(params)), m_imgCreationParams({
        .type = IGPUImage::ET_2D,
        .samples = IGPUImage::ESCF_1_BIT,
        .format = m_params.surfaceFormat.format,
        .extent = {m_params.sharedParams.width,m_params.sharedParams.height,1u},
        .mipLevels = 1u,
        .arrayLayers = m_params.sharedParams.arrayLayers,
        .flags = m_params.computeImageCreationFlags(getOriginDevice()->getPhysicalDevice()),
        .usage = m_params.sharedParams.imageUsage,
        // stencil usage remains none because swapchains don't have stencil formats!
        .viewFormats = m_params.sharedParams.viewFormats
    }), m_oldSwapchain(std::move(oldSwapchain)), m_imageCount(imageCount)
{
    assert(params.queueFamilyIndices.size()<=ILogicalDevice::MaxQueueFamilies);
    assert(imageCount<=ISwapchain::MaxImages);

    std::copy(m_params.queueFamilyIndices.begin(),m_params.queueFamilyIndices.end(),m_queueFamilies.data());
    m_params.queueFamilyIndices = {m_queueFamilies.data(),m_params.queueFamilyIndices.size()};

    for (auto i=0; i<m_imageCount; i++)
        m_frameResources[i] = std::make_unique<MultiTimelineEventHandlerST<DeferredFrameSemaphoreDrop>>(const_cast<ILogicalDevice*>(getOriginDevice()));
} 

}
