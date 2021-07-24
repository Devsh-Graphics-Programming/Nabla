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
    CSurfaceVKWin32(VkInstance instance, SCreationParams&& params)
        : ISurfaceWin32(std::move(params)), ISurfaceVK(instance)
    {
        VkWin32SurfaceCreateInfoKHR ci;
        ci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        ci.hinstance = m_params.hinstance;
        ci.hwnd = m_params.hwnd;
        ci.flags = 0;
        ci.pNext = nullptr;
        vkCreateWin32SurfaceKHR(m_instance, &ci, nullptr, &m_surface);
    }
};

}

#endif