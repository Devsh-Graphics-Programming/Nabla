#include "nbl/video/surface/ISurfaceVK.h"

#include "nbl/video/CVulkanPhysicalDevice.h"

namespace nbl::video
{

bool ISurfaceVK::isSupported(const IPhysicalDevice* dev, uint32_t _queueFamIx) const
{
    if (dev->getAPIType() != EAT_VULKAN)
    {
        // Todo(achal): Log error
        return false;
    }

    auto vkphd = static_cast<const CVulkanPhysicalDevice*>(dev)->getInternalObject();
    VkBool32 supported;
    vkGetPhysicalDeviceSurfaceSupportKHR(vkphd, _queueFamIx, m_surface, &supported);

    return static_cast<bool>(supported);
}

}