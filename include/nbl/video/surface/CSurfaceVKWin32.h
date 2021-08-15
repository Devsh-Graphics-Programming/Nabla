#ifndef __NBL_C_SURFACE_VK_WIN32_H_INCLUDED__
#define __NBL_C_SURFACE_VK_WIN32_H_INCLUDED__

#include <volk.h>
#include "nbl/video/surface/ISurfaceVK.h"

namespace nbl::video
{

class IAPIConnection;

#if 0
class CSurfaceVKWin32 final : public ISurfaceVK
{
public:
    static core::smart_refctd_ptr<CSurfaceVKWin32> create(
        core::smart_refctd_ptr<CVulkanConnection>&& apiConnection,
        core::smart_refctd_ptr<Window>&& window);

private:
    CSurfaceVKWin32(core::smart_refctd_ptr<const CVulkanConnection>&& connection);
};
#endif

}

#endif