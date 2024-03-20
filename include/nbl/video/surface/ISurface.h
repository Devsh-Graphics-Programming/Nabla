#ifndef _NBL_VIDEO_I_SURFACE_H_INCLUDED_
#define _NBL_VIDEO_I_SURFACE_H_INCLUDED_

#include "nbl/core/IReferenceCounted.h"

#include "nbl/asset/format/EColorSpace.h"
#include "nbl/asset/format/EFormat.h"

#include "nbl/video/IAPIConnection.h"
#include "nbl/builtin/hlsl/surface_transform.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

namespace nbl::video
{

class IPhysicalDevice;

class ISurface : public core::IReferenceCounted
{
    protected:
        ISurface(core::smart_refctd_ptr<IAPIConnection>&& api) : m_api(std::move(api)) {}
        virtual ~ISurface() = default;

        core::smart_refctd_ptr<IAPIConnection> m_api;

    public:
        struct SColorSpace
        {
            inline bool operator==(const SColorSpace&) const = default;
            inline bool operator!=(const SColorSpace&) const = default;

            asset::E_COLOR_PRIMARIES primary = asset::ECP_COUNT;
            asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION eotf = asset::EOTF_UNKNOWN;
        };
        struct SFormat
        {
            inline SFormat() = default;
            inline SFormat(const asset::E_FORMAT _format, const asset::E_COLOR_PRIMARIES _primary, const asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION _eotf) : format(_format), colorSpace({_primary, _eotf}) {}

            inline bool operator==(const SFormat&) const = default;
            inline bool operator!=(const SFormat&) const = default;

            asset::E_FORMAT format = asset::EF_UNKNOWN;
            SColorSpace colorSpace = {};
        };

        // TODO: move these enums to HLSL header!
        enum E_PRESENT_MODE : uint8_t
        {
            EPM_NONE = 0x0,
            EPM_IMMEDIATE = 1<<0,
            EPM_MAILBOX = 1<<1,
            EPM_FIFO = 1<<2,
            EPM_FIFO_RELAXED = 1<<3,
            EPM_ALL_BITS = 0x7u,
            EPM_UNKNOWN = 0
        };

        enum E_COMPOSITE_ALPHA : uint8_t
        {
            ECA_NONE = 0x0,
            ECA_OPAQUE_BIT = 0x01,
            ECA_PRE_MULTIPLIED_BIT = 0x02,
            ECA_POST_MULTIPLIED_BIT = 0x04,
            ECA_INHERIT_BIT = 0x08,
            ECA_ALL_BITS = 0x0F
        };
        // TODO: end of move to HLSL block

        struct SCapabilities
        {
            core::bitflag<asset::IImage::E_USAGE_FLAGS> supportedUsageFlags = {};
            VkExtent2D currentExtent = {0u,0u};
            VkExtent2D minImageExtent = {~0u,~0u};
            VkExtent2D maxImageExtent = {0u,0u};
            uint8_t minImageCount = 0xffu;
            uint8_t maxImageCount = 0;
            uint8_t maxImageArrayLayers = 0;
            core::bitflag<E_COMPOSITE_ALPHA> supportedCompositeAlpha = ECA_NONE;
            core::bitflag<hlsl::SurfaceTransform::FLAG_BITS> supportedTransforms = hlsl::SurfaceTransform::FLAG_BITS::NONE;
            hlsl::SurfaceTransform::FLAG_BITS currentTransform = hlsl::SurfaceTransform::FLAG_BITS::NONE;
        };

        inline IAPIConnection* getAPIConnection() const { return m_api.get(); }

        inline E_API_TYPE getAPIType() const { return m_api->getAPIType(); }

        virtual bool isSupportedForPhysicalDevice(const IPhysicalDevice* dev, const uint32_t _queueFamIx) const = 0;

        virtual void getAvailableFormatsForPhysicalDevice(const IPhysicalDevice* physicalDevice, uint32_t& formatCount, ISurface::SFormat* formats) const = 0;

        virtual core::bitflag<ISurface::E_PRESENT_MODE> getAvailablePresentModesForPhysicalDevice(const IPhysicalDevice* physicalDevice) const = 0;

        virtual bool getSurfaceCapabilitiesForPhysicalDevice(const IPhysicalDevice* physicalDevice, ISurface::SCapabilities& capabilities) const = 0;

        // Can we Nuke this too and get rid of the extra `CSurface` and `CSurfaceNative` inheritance? 
        virtual const void* getNativeWindowHandle() const = 0;
};

// Base for use with Nabla's window wrappers, should maybe be called `CSurfaceWindow` instead, but oh well
template<class Window, class ImmediateBase> requires (std::is_base_of_v<ui::IWindow,Window> && std::is_base_of_v<ISurface,ImmediateBase>)
class CSurface : public ImmediateBase
{
        using this_t = CSurface<Window, ImmediateBase>;

    public:
        using window_t = Window;
        using immediate_base_t = ImmediateBase;

        inline window_t* getWindow()
        {
            return m_window.get();
        }
        inline const window_t* getWindow() const {return const_cast<window_t*>(const_cast<this_t*>(this)->getWindow());}

        inline const void* getNativeWindowHandle() const override final
        {
            return m_window->getNativeHandle();
        }

    protected:
        template<typename... Args>
        CSurface(core::smart_refctd_ptr<window_t>&& window, Args&&... args) : immediate_base_t(std::forward<Args>(args)...), m_window(std::move(window)) {}
        virtual ~CSurface() = default;

        core::smart_refctd_ptr<window_t> m_window;
};

// Base to make surfaces directly from Native OS window handles (TODO: while merging Erfan, template on the handle instead of the Window)
template<class Window, class ImmediateBase> requires std::is_base_of_v<ISurface,ImmediateBase>
class CSurfaceNative : public ImmediateBase
{
    public:
        inline const void* getNativeWindowHandle() const override final
        {
            return m_handle;
        }

    protected:
        template<typename... Args>
        CSurfaceNative(typename Window::native_handle_t handle, Args&&... args) : ImmediateBase(std::forward<Args>(args)...), m_handle(handle) {}
        virtual ~CSurfaceNative() = default;

        typename Window::native_handle_t m_handle;
};

}

#endif