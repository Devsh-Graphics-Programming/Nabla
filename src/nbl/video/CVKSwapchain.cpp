#include "nbl/video/CVKSwapchain.h"

#include "nbl/video/CVKLogicalDevice.h"
#include "nbl/video/surface/ISurfaceVK.h"
// #include "nbl/video/CVulkanImage.h"

namespace nbl
{
namespace video
{

CVKSwapchain::CVKSwapchain(SCreationParams&& params, CVKLogicalDevice* dev) : ISwapchain(dev, std::move(params)), m_device(dev)
{
    VkSwapchainCreateInfoKHR ci;
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.pNext = nullptr;
    ci.minImageCount = params.minImageCount;
    ci.clipped = VK_TRUE;
    ci.imageArrayLayers = params.arrayLayers;
    // TODO function mapping ISurface::SFormat -> VkColorSpaceKHR
    //ci.imageColorSpace = ...
    ci.presentMode = static_cast<VkPresentModeKHR>(params.presentMode);
    ci.imageExtent = { params.width, params.height };
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; // single queue famility at a time
    ci.queueFamilyIndexCount = params.queueFamilyIndices->size();
    ci.pQueueFamilyIndices = params.queueFamilyIndices->data();
    ci.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    ci.oldSwapchain = VK_NULL_HANDLE;
    //ci.imageUsage = ... // TODO (we dont have the enum yet)
    ci.flags = 0;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.imageFormat = static_cast<VkFormat>(params.surfaceFormat.format);
    ci.surface = static_cast<ISurfaceVK*>(params.surface.get())->getInternalObject();

    // auto* vk = m_device->getFunctionTable();
    // VkDevice vkdev = m_device->getInternalObject();

    VkSwapchainKHR sc;
    // vk->vk.vkCreateSwapchainKHR(vkdev, &ci, nullptr, &sc);

    m_swapchain = sc;

    uint32_t imgCount = 0u;
    // vkGetSwapchainImagesKHR(vkdev, m_swapchain, &imgCount, nullptr);
    m_images = core::make_refctd_dynamic_array<images_array_t>(imgCount); 

    VkImage vkimgs[100];
    assert(imgCount > 100);
    // vkGetSwapchainImagesKHR(vkdev, m_swapchain, &imgCount, vkimgs);

    // uint32_t i = 0u;
    // for (auto& img : (*m_images))
    // {
    //     CVulkanImage::SCreationParams params;
    //     params.arrayLayers = m_params.arrayLayers;
    //     params.extent = { m_params.width, m_params.height, 1u };
    //     params.flags = static_cast<CVulkanImage::E_CREATE_FLAGS>(0);
    //     params.format = m_params.surfaceFormat.format;
    //     params.mipLevels = 1u;
    //     params.samples = CVulkanImage::ESCF_1_BIT;
    //     params.type = CVulkanImage::ET_2D;
    //     
    //     // TODO might want to change this to dev->createImage()
    //     img = core::make_smart_refctd_ptr<CVulkanImage>(m_device, std::move(params), vkimgs[i++]);
    // }
}

CVKSwapchain::~CVKSwapchain()
{
    // auto* vk = m_device->getFunctionTable();
    // vk->vk.vkDestroySwapchainKHR(m_device->getInternalObject(), m_swapchain, nullptr);
}

auto CVKSwapchain::acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) -> E_ACQUIRE_IMAGE_RESULT
{
    // VkDevice dev = m_device->getInternalObject();
    // auto* vk = m_device->getFunctionTable();

    // TODO get semaphore and fence vk handles
    // VkResult result = vk->vk.vkAcquireNextImageKHR(dev, m_swapchain, timeout, 0, 0, out_imgIx);
    // switch (result)
    // {
    // case VK_SUCCESS:
    //     return EAIR_SUCCESS; break;
    // case VK_TIMEOUT:
    //     return EAIR_TIMEOUT; break;
    // case VK_NOT_READY:
    //     return EAIR_NOT_READY; break;
    // case VK_SUBOPTIMAL_KHR:
    //     return EAIR_SUBOPTIMAL; break;
    // default:
    //     return EAIR_ERROR; break;
    // }
    return EAIR_ERROR;
}

}
}