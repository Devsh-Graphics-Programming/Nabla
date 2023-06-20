#ifndef _NBL_VIDEO_I_SURFACE_H_INCLUDED_
#define _NBL_VIDEO_I_SURFACE_H_INCLUDED_

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
                : format(asset::EF_UNKNOWN)
                , colorSpace({asset::ECP_COUNT, asset::EOTF_UNKNOWN})
            {}

            SFormat(asset::E_FORMAT _format, asset::E_COLOR_PRIMARIES _primary, asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION _eotf)
                : format(_format)
                , colorSpace({_primary, _eotf})
            {
            }
            asset::E_FORMAT format;
            SColorSpace colorSpace;
        };
        enum E_PRESENT_MODE
        {
            EPM_IMMEDIATE = 1<<0,
            EPM_MAILBOX = 1<<1,
            EPM_FIFO = 1<<2,
            EPM_FIFO_RELAXED = 1<<3,
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

        #include "nbl/builtin/glsl/utils/surface_transform_e.h"
        static_assert(int(EST_IDENTITY_BIT) == NBL_GLSL_SURFACE_TRANSFORM_E_IDENTITY);
        static_assert(int(EST_ROTATE_90_BIT) == NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_90);
        static_assert(int(EST_ROTATE_180_BIT) == NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_180);
        static_assert(int(EST_ROTATE_270_BIT) == NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_270);
        static_assert(int(EST_HORIZONTAL_MIRROR_BIT) == NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR);
        static_assert(int(EST_HORIZONTAL_MIRROR_ROTATE_90_BIT) == NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_90);
        static_assert(int(EST_HORIZONTAL_MIRROR_ROTATE_180_BIT) == NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_180);
        static_assert(int(EST_HORIZONTAL_MIRROR_ROTATE_270_BIT) == NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_270);
        #undef NBL_GLSL_SURFACE_TRANSFORM_E_IDENTITY
        #undef NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_90
        #undef NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_180
        #undef NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_270
        #undef NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR
        #undef NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_90
        #undef NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_180
        #undef NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_270

        // A matrix that can be pre-multiplied to the projection matrix in order to apply the
        // surface transform.
        static inline core::matrix4SIMD getSurfaceTransformationMatrix(const E_SURFACE_TRANSFORM_FLAGS transform)
        {
            const float sin90 = 1.0, cos90 = 0.0,
                sin180 = 0.0, cos180 = -1.0,
                sin270 = -1.0, cos270 = 0.0;

            switch (transform)
            {
            case ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_ROTATE_90_BIT:
                return core::matrix4SIMD(
                    cos90, -sin90, 0.0, 0.0,
                    sin90, cos90, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                );
            case ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_ROTATE_180_BIT:
                return core::matrix4SIMD(
                    cos180, -sin180, 0.0, 0.0,
                    sin180, cos180, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                );
            case ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_ROTATE_270_BIT:
                return core::matrix4SIMD(
                    cos270, -sin270, 0.0, 0.0,
                    sin270, cos270, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                );
            case ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_HORIZONTAL_MIRROR_BIT:
                return core::matrix4SIMD(
                    -1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                );
            // The same matricies as the rotation ones above, but with the horizontal mirror matrix
            // (directly above this) pre-multiplied
            case ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_HORIZONTAL_MIRROR_ROTATE_90_BIT:
                return core::matrix4SIMD(
                    -cos90, sin90, 0.0, 0.0,
                    sin90, cos90, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                );
            case ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_HORIZONTAL_MIRROR_ROTATE_180_BIT:
                return core::matrix4SIMD(
                    -cos180, sin180, 0.0, 0.0,
                    sin180, cos180, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                );
            case ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_HORIZONTAL_MIRROR_ROTATE_270_BIT:
                return core::matrix4SIMD(
                    -cos270, sin270, 0.0, 0.0,
                    sin270, cos270, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                );
            default:
                return core::matrix4SIMD();
            }
        }

        static inline float getTransformedAspectRatio(const E_SURFACE_TRANSFORM_FLAGS transform, uint32_t w, uint32_t h)
        {
            switch (transform)
            {
            case ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_ROTATE_90_BIT:
            case ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_ROTATE_270_BIT:
            case ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_HORIZONTAL_MIRROR_ROTATE_90_BIT:
            case ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_HORIZONTAL_MIRROR_ROTATE_270_BIT:
                return float(h) / w;
            case ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_ROTATE_180_BIT:
            case ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_HORIZONTAL_MIRROR_BIT:
            case ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_HORIZONTAL_MIRROR_ROTATE_180_BIT:
                return float(w) / h;
            default:
                return float(w) / h;
            }
        }

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

// Base for use with Nabla's window wrappers
template<class Window, class ImmediateBase>
class CSurface : public ImmediateBase
{
    public:
        inline const void* getNativeWindowHandle() const override final
        {
            return m_window->getNativeHandle();
        }

    protected:
        template<typename... Args>
        CSurface(core::smart_refctd_ptr<Window>&& window, Args&&... args) : ImmediateBase(std::forward<Args>(args)...), m_window(std::move(window)) {}
        virtual ~CSurface() = default;

        uint32_t getWidth() const override { return m_window->getWidth(); }
        uint32_t getHeight() const override { return m_window->getHeight(); }

        core::smart_refctd_ptr<Window> m_window;
};

// Base to make surfaces directly from Native OS window handles
template<class Window, class ImmediateBase>
class CSurfaceNative : public ImmediateBase
{
    public:
        inline const void* getNativeWindowHandle() const override final
        {
            return m_handle;
        }

    protected:
        CSurfaceNative(core::smart_refctd_ptr<IAPIConnection>&& api, typename Window::native_handle_t handle) : ImmediateBase(std::move(api)), m_handle(handle) {}
        virtual ~CSurfaceNative() = default;

        typename Window::native_handle_t m_handle;
};

}

#endif