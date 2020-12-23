#ifndef __NBL_I_SURFACE_VK_H_INCLUDED__
#define __NBL_I_SURFACE_VK_H_INCLUDED__

#include <volk.h>
#include "nbl/video/surface/ISurface.h"

namespace nbl {
namespace video
{

class IAPIConnection;

class ISurfaceVK : public ISurface
{
public:
    inline VkSurfaceKHR getInternalObject() const { return m_surface; }

protected:
    VkSurfaceKHR m_surface;
};

}
}

#endif