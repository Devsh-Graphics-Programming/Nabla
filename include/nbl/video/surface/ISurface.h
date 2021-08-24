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
            EPM_FIFO_RELAXED = 3,
            EPM_UNKNOWN = 4
        };

        struct SCapabilities
        {
            uint32_t minImageCount;
            uint32_t maxImageCount;
            VkExtent2D currentExtent;
            VkExtent2D minImageExtent;
            VkExtent2D maxImageExtent;
            uint32_t maxImageArrayLayers;
            // Todo(achal)
            // VkSurfaceTransformFlagsKHR       supportedTransforms;
            // VkSurfaceTransformFlagBitsKHR    currentTransform;
            // VkCompositeAlphaFlagsKHR         supportedCompositeAlpha;
            asset::IImage::E_USAGE_FLAGS supportedUsageFlags;
        };

        inline E_API_TYPE getAPIType() const { return m_api->getAPIType(); }

        virtual bool isSupported(const IPhysicalDevice* dev, uint32_t _queueFamIx) const = 0;

        // Todo(achal): Not making them pure virtual only because the OpenGL backends doesn't have any implementation
        // for them
        virtual void getAvailableFormatsForPhysicalDevice(const IPhysicalDevice* physicalDevice, uint32_t& formatCount, ISurface::SFormat* formats) const
        {
            assert(false); // Invalid code path
        }

        virtual ISurface::E_PRESENT_MODE getAvailablePresentModesForPhysicalDevice(const IPhysicalDevice* physicalDevice) const
        {
            assert(false); // Invalid code path
            return ISurface::EPM_UNKNOWN;
        }

        virtual bool getSurfaceCapabilitiesForPhysicalDevice(const IPhysicalDevice* physicalDevice, ISurface::SCapabilities& capabilities) const
        {
            assert(false); // Invalid code path
            return false;
        }

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