#include "nbl/video/surface/ISurfaceVK.h"

#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanConnection.h"

namespace nbl::video
{

ISurfaceVK::ISurfaceVK(core::smart_refctd_ptr<const CVulkanConnection>&& apiConnection)
    : m_apiConnection(std::move(apiConnection)) {}

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

ISurfaceVK::~ISurfaceVK()
{
    vkDestroySurfaceKHR(m_apiConnection->getInternalObject(), m_surface, nullptr);
}

}