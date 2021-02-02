#ifndef __NBL_C_OPENGL_CONNECTION_H_INCLUDED__
#define __NBL_C_OPENGL_CONNECTION_H_INCLUDED__

#include "nbl/video/IAPIConnection.h"
#include "nbl/video/CEGL.h"
#include "nbl/video/COpenGLPhysicalDevice.h"

namespace nbl {
namespace video
{

class COpenGLConnection final : public IAPIConnection
{
public:
    COpenGLConnection()
    {
        m_egl.initialize();
        m_pdevice = COpenGLPhysicalDevice::create(&m_egl);
    }

    E_TYPE getAPIType() const override
    {
        return ET_OPENGL;
    }

    core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>> getPhysicalDevices() const override
    {          
        if (!m_pdevice)
            return core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>>{ nullptr, nullptr };

        return core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>>{ &m_pdevice, &m_pdevice + 1 };
    }

protected:
    ~COpenGLConnection()
    {
        m_egl.deinitialize();
    }

private:
    egl::CEGL m_egl;
    core::smart_refctd_ptr<IPhysicalDevice> m_pdevice;
};

}
}

#endif