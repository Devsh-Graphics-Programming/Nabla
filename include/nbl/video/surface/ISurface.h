#ifndef __NBL_I_SURFACE_H_INCLUDED__
#define __NBL_I_SURFACE_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"

#include "nbl/asset/format/EColorSpace.h"
#include "nbl/asset/format/EFormat.h"

#include "nbl/video/IAPIConnection.h"

namespace nbl::video
{

class IPhysicalDevice;

class ISurface : public core::IReferenceCounted
{
    protected:
        ISurface(core::smart_refctd_ptr<IAPIConnection>&& api) : m_api(std::move(api)) {}
        ~ISurface() = default;

        core::smart_refctd_ptr<IAPIConnection> m_api;

    public:
        struct SColorSpace
        {
            asset::E_COLOR_PRIMARIES primary;
            asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION eotf;
        };
        struct SFormat
        {
            asset::E_FORMAT format;
            SColorSpace colorSpace;
        };
        enum E_PRESENT_MODE
        {
            EPM_IMMEDIATE = 0,
            EPM_MAILBOX = 1,
            EPM_FIFO = 2,
            EPM_FIFO_RELAXED = 3
        };

        // TODO
        // IPhysicalDevice::getAvailableFormatsForSurface(const ISurface*, SFormat* out);
        // IPhysicalDevice::getAvailablePresentModesForSurface(const ISurface*, E_PRESENT_MODE* out);
        // IPhysicalDevice::getMinImageCountForSurface(const ISurface*)

        // vkGetPhysicalDeviceSurfaceSupportKHR on vulkan
        virtual bool isSupported(const IPhysicalDevice* dev, uint32_t _queueFamIx) const = 0;

        // used by some drivers
        virtual const void* getNativeWindowHandle() const = 0;
};

template<class Window>
class CSurface : public ISurface
{
    protected:
        CSurface(core::smart_refctd_ptr<IAPIConnection>&& api, core::smart_refctd_ptr<Window>&& window) : ISurface(std::move(api)), m_window(std::move(window)) {}
        ~CSurface() = default;

        core::smart_refctd_ptr<Window> m_window;
};

}


#endif