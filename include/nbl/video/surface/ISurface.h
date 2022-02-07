#ifndef __NBL_I_SURFACE_H_INCLUDED__
#define __NBL_I_SURFACE_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"

#include "nbl/asset/format/EColorSpace.h"
#include "nbl/asset/format/EFormat.h"

#include "nbl/video/IAPIConnection.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

namespace nbl::video
{
class IPhysicalDevice;

class ISurface : public core::IReferenceCounted
{
protected:
    ISurface(core::smart_refctd_ptr<IAPIConnection>&& api)
        : m_api(std::move(api)) {}
    virtual ~ISurface() = default;

    //impl of getSurfaceCapabilitiesForPhysicalDevice() needs this
    virtual uint32_t getWidth() const = 0;
    virtual uint32_t getHeight() const = 0;

    core::smart_refctd_ptr<IAPIConnection> m_api;

public:
    struct SColorSpace
    {
        asset::E_COLOR_PRIMARIES primary;
        asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION eotf;
    };
    struct SFormat
    {
        SFormat()
            : format(asset::EF_UNKNOWN), colorSpace({asset::ECP_COUNT, asset::EOTF_UNKNOWN})
        {}

        SFormat(asset::E_FORMAT _format, asset::E_COLOR_PRIMARIES _primary, asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION _eotf)
            : format(_format), colorSpace({_primary, _eotf})
        {
        }
        asset::E_FORMAT format;
        SColorSpace colorSpace;
    };
    enum E_PRESENT_MODE
    {
        EPM_IMMEDIATE = 1 << 0,
        EPM_MAILBOX = 1 << 1,
        EPM_FIFO = 1 << 2,
        EPM_FIFO_RELAXED = 1 << 3,
        EPM_UNKNOWN = 0
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

    enum E_COMPOSITE_ALPHA : uint32_t
    {
        ECA_OPAQUE_BIT = 0x00000001,
        ECA_PRE_MULTIPLIED_BIT = 0x00000002,
        ECA_POST_MULTIPLIED_BIT = 0x00000004,
        ECA_INHERIT_BIT = 0x00000008,
        ECA_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
    };

    struct SCapabilities
    {
        uint32_t minImageCount;
        uint32_t maxImageCount;
        VkExtent2D currentExtent;
        VkExtent2D minImageExtent;
        VkExtent2D maxImageExtent;
        uint32_t maxImageArrayLayers;
        core::bitflag<E_SURFACE_TRANSFORM_FLAGS> supportedTransforms;
        E_SURFACE_TRANSFORM_FLAGS currentTransform;
        core::bitflag<E_COMPOSITE_ALPHA> supportedCompositeAlpha;
        core::bitflag<asset::IImage::E_USAGE_FLAGS> supportedUsageFlags;
    };

    inline E_API_TYPE getAPIType() const { return m_api->getAPIType(); }

    virtual bool isSupportedForPhysicalDevice(const IPhysicalDevice* dev, uint32_t _queueFamIx) const = 0;

    virtual void getAvailableFormatsForPhysicalDevice(const IPhysicalDevice* physicalDevice, uint32_t& formatCount, ISurface::SFormat* formats) const = 0;

    virtual ISurface::E_PRESENT_MODE getAvailablePresentModesForPhysicalDevice(const IPhysicalDevice* physicalDevice) const = 0;

    virtual bool getSurfaceCapabilitiesForPhysicalDevice(const IPhysicalDevice* physicalDevice, ISurface::SCapabilities& capabilities) const = 0;

    // used by some drivers
    virtual const void* getNativeWindowHandle() const = 0;
};

template<class Window>
class CSurface : public ISurface
{
public:
    inline const void* getNativeWindowHandle() const override final
    {
        return m_window->getNativeHandle();
    }

protected:
    CSurface(core::smart_refctd_ptr<IAPIConnection>&& api, core::smart_refctd_ptr<Window>&& window)
        : ISurface(std::move(api)), m_window(std::move(window)) {}
    virtual ~CSurface() = default;

    uint32_t getWidth() const override { return m_window->getWidth(); }
    uint32_t getHeight() const override { return m_window->getHeight(); }

    core::smart_refctd_ptr<Window> m_window;
};

template<class Window>
class CSurfaceNative : public ISurface
{
public:
    inline const void* getNativeWindowHandle() const override final
    {
        return m_handle;
    }

protected:
    CSurfaceNative(core::smart_refctd_ptr<IAPIConnection>&& api, typename Window::native_handle_t handle)
        : ISurface(std::move(api)), m_handle(handle) {}
    virtual ~CSurfaceNative() = default;

    typename Window::native_handle_t m_handle;
};

}

#endif