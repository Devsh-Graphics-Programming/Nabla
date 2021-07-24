#ifndef __NBL_I_SURFACE_VK_H_INCLUDED__
#define __NBL_I_SURFACE_VK_H_INCLUDED__

#include <volk.h>
#include "nbl/video/surface/ISurface.h"

namespace nbl::video
{

class IPhysicalDevice;

class ISurfaceVK : public ISurface
{
public:
    inline VkSurfaceKHR getInternalObject() const { return m_surface; }

    bool isSupported(const IPhysicalDevice* dev, uint32_t _queueIx) const override;

// protected:
    ISurfaceVK(VkInstance instance)
        : m_instance(instance) {}

    virtual ~ISurfaceVK() { vkDestroySurfaceKHR(m_instance, m_surface, nullptr); }

    VkSurfaceKHR m_surface;
    VkInstance m_instance;
};

}

#endif