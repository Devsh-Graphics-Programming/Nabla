#include "nbl/video/ISwapchain.h"

#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanConnection.h"

namespace nbl::video
{
	
ISwapchain::ISwapchain(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params, const uint8_t imageCount)
    : IBackendObject(std::move(dev)), m_params(std::move(params)), m_imageCount(imageCount), m_imgCreationParams{}
{
    assert(imageCount <= ISwapchain::MaxImages);

    m_imgCreationParams.type = IGPUImage::ET_2D;
    m_imgCreationParams.samples = IGPUImage::ESCF_1_BIT;
    m_imgCreationParams.format = m_params.surfaceFormat.format;
    m_imgCreationParams.extent = { m_params.width, m_params.height, 1u };
    m_imgCreationParams.mipLevels = 1u;
    m_imgCreationParams.arrayLayers = m_params.arrayLayers;
    m_imgCreationParams.flags = m_params.viewFormats.count()>1u ? IGPUImage::ECF_MUTABLE_FORMAT_BIT:IGPUImage::ECF_NONE;
    m_imgCreationParams.usage = m_params.imageUsage;
    if (!(getOriginDevice()->getPhysicalDevice()->getImageFormatUsagesOptimalTiling()[m_imgCreationParams.format]<m_imgCreationParams.usage))
        m_imgCreationParams.flags |= IGPUImage::ECF_EXTENDED_USAGE_BIT;
    m_imgCreationParams.viewFormats = m_params.viewFormats;

    // don't need to keep a reference to the old swapchain anymore
    m_params.oldSwapchain = nullptr;
}

}
