#ifndef __NBL_I_SURFACE_VK_H_INCLUDED__
#define __NBL_I_SURFACE_VK_H_INCLUDED__

#include <volk.h>
#include "nbl/video/surface/ISurface.h"

namespace nbl::video
{

class IPhysicalDevice;
class CVulkanConnection;

class ISurfaceVK : public ISurface
{
public:
    inline VkSurfaceKHR getInternalObject() const { return m_surface; }

    bool isSupported(const IPhysicalDevice* dev, uint32_t _queueIx) const override;

// protected:
    ISurfaceVK(core::smart_refctd_ptr<const CVulkanConnection>&& apiConnection);

    virtual ~ISurfaceVK();

    VkSurfaceKHR m_surface;
    core::smart_refctd_ptr<const CVulkanConnection> m_apiConnection;
};

}

#endif