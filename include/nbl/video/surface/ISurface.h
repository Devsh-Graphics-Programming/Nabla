#ifndef __NBL_I_SURFACE_H_INCLUDED__
#define __NBL_I_SURFACE_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/asset/format/EColorSpace.h"
#include "nbl/asset/format/EFormat.h"

namespace nbl::video
{

class IPhysicalDevice;

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
    enum E_PRESENT_MODE : uint8_t
    {
        EPM_IMMEDIATE       = 1 << 0,
        EPM_MAILBOX         = 1 << 1,
        EPM_FIFO            = 1 << 2,
        EPM_FIFO_RELAXED    = 1 << 3
    };

    enum E_SURFACE_TRANSFORM_FLAGS : uint32_t
    {
        EST_IDENTITY_BIT = 0x00000001,
        EST_ROTATE_90_BIT = 0x00000002,
        EST_ROTATE_180_BIT = 0x00000004,
        EST_ROTATE_270_BIT = 0x00000008,
        EST_HORIZONTAL_MIRROR_BIT = 0x00000010,
        EST_HORIZONTAL_MIRROR_ROTATE_90_BIT = 0x00000020,
        EST_HORIZONTAL_MIRROR_ROTATE_180_BIT = 0x00000040,
        EST_HORIZONTAL_MIRROR_ROTATE_270_BIT = 0x00000080,
        EST_INHERIT_BIT = 0x00000100,
        EST_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
    };

    // vkGetPhysicalDeviceSurfaceSupportKHR on vulkan
    virtual bool isSupported(const IPhysicalDevice* dev, uint32_t _queueFamIx) const = 0;
};

}

#endif