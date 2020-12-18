#ifndef __NBL_I_SURFACE_H_INCLUDED__
#define __NBL_I_SURFACE_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/asset/EColorSpace.h"
#include "nbl/asset/format/EFormat.h"

namespace nbl {
namespace video
{

class ISurface : public core::IReferenceCounted
{
protected:
    virtual ~ISurface() = default;

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
    // IPhysicalDevice::getAvailableSurfaceFormats(const ISurface*, SFormat* out);
    // IPhysicalDevice::getAvailableSurfacePresentModes(const ISurface*, E_PRESENT_MODE* out);

    // vkGetPhysicalDeviceSurfaceSupportKHR on vulkan
    virtual bool isSupported(const IPhysicalDevice* dev, uint32_t _queueIx) const = 0;
};

}
}


#endif