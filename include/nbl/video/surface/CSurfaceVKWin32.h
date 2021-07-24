#ifndef __NBL_C_SURFACE_VK_WIN32_H_INCLUDED__
#define __NBL_C_SURFACE_VK_WIN32_H_INCLUDED__

#include <volk.h>
#include "nbl/video/surface/ISurfaceWin32.h"
#include "nbl/video/surface/ISurfaceVK.h"

namespace nbl::video
{

class IAPIConnection;

class CSurfaceVKWin32 final : public ISurfaceWin32, public ISurfaceVK
{
public:
    static core::smart_refctd_ptr<CSurfaceVKWin32> create(const IAPIConnection* api, SCreationParams&& params);

// private:
    CSurfaceVKWin32(core::smart_refctd_ptr<const CVulkanConnection>&& connection, SCreationParams&& params);
};

}

#endif