#include "nbl/video/CVulkanForeignImage.h"

namespace nbl::video
{

CVulkanForeignImage::~CVulkanForeignImage()
{
    // m_swapchain = nullptr;
    m_vkImage = VK_NULL_HANDLE;
}

}