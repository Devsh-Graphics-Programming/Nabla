#include "nbl/video/ISwapchain.h"

#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanConnection.h"

namespace nbl::video
{
	
ISwapchain::ISwapchain(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params, const uint8_t imageCount)
    : IBackendObject(std::move(dev)), m_imageCount(imageCount)
{
    assert(params.queueFamilyIndexCount<=ILogicalDevice::SCreationParams::MaxQueueFamilies);
    std::copy_n(params.queueFamilyIndices,params.queueFamilyIndexCount,m_queueFamilies.data());
    params.queueFamilyIndices = m_queueFamilies.data();
    params.oldSwapchain = nullptr; // don't need to keep a reference to the old swapchain anymore

    assert(imageCount<=ISwapchain::MaxImages);
    m_imgCreationParams = {
        .type = IGPUImage::ET_2D,
        .samples = IGPUImage::ESCF_1_BIT,
        .format = m_params.surfaceFormat.format,
        .extent = { m_params.width, m_params.height, 1u },
        .mipLevels = 1u,
        .arrayLayers = m_params.arrayLayers,
        .flags = m_params.viewFormats.count()>1u ? IGPUImage::ECF_MUTABLE_FORMAT_BIT:IGPUImage::ECF_NONE,
        .usage = m_params.imageUsage,
        // stencil usage remains none because swapchains don't have stencil formats!
        .viewFormats = m_params.viewFormats
    };
    if (!(getOriginDevice()->getPhysicalDevice()->getImageFormatUsagesOptimalTiling()[m_imgCreationParams.format]<m_imgCreationParams.usage))
        m_imgCreationParams.flags |= IGPUImage::ECF_EXTENDED_USAGE_BIT;
}

}
