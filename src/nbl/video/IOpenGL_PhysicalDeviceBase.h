#ifndef __NBL_I_OPENGL_PHYSICAL_DEVICE_BASE_H_INCLUDED__
#define __NBL_I_OPENGL_PHYSICAL_DEVICE_BASE_H_INCLUDED__

#include "nbl/video/CEGL.h"
#include "nbl/video/IPhysicalDevice.h"

namespace nbl { 
namespace video
{

template <typename LogicalDeviceType>
class IOpenGL_PhysicalDeviceBase : public IPhysicalDevice
{
public:
    explicit IOpenGL_PhysicalDeviceBase(const egl::CEGL* _egl, EGLConfig _config, EGLint _major, EGLint _minor) : 
        m_egl(_egl), config(_config), m_gl_major(_major), m_gl_minor(_minor)
    {
        // OpenGL backend emulates presence of just one queue with all capabilities (graphics, compute, transfer, ... what about sparse binding?)
        SQueueFamilyProperties qprops;
        qprops.queueFlags = EQF_GRAPHICS_BIT | EQF_COMPUTE_BIT | EQF_TRANSFER_BIT;
        qprops.queueCount = 1u;
        qprops.timestampValidBits = 64u; // ??? TODO
        qprops.minImageTransferGranularity = { 1u,1u,1u }; // ??? TODO

        m_qfamProperties = core::make_refctd_dynamic_array<qfam_props_array_t>(1u, qprops);

        // TODO fill m_properties and m_features (possibly should be done in derivative classes' ctors, not sure yet)
    }

protected:
    virtual ~IOpenGL_PhysicalDeviceBase() = default;

protected:
    const egl::CEGL* m_egl;
    EGLConfig m_config;
    EGLint m_gl_major, m_gl_minor;
};

}
}

#endif
