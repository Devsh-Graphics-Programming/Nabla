#ifndef __NBL_C_OPENGL__CONNECTION_H_INCLUDED__
#define __NBL_C_OPENGL__CONNECTION_H_INCLUDED__

#include "nbl/video/IAPIConnection.h"
#include "nbl/video/debug/COpenGLDebugCallback.h"

namespace nbl::video
{

template<E_API_TYPE API_TYPE>
class COpenGL_Connection final : public IAPIConnection
{
    public:
        static core::smart_refctd_ptr<COpenGL_Connection<API_TYPE>> create(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName, COpenGLDebugCallback&& dbgCb);

        E_API_TYPE getAPIType() const override
        {
            return API_TYPE;
        }

        core::SRange<IPhysicalDevice *const> getPhysicalDevices() const override
        {      
            return core::SRange<IPhysicalDevice *const>(&m_pdevice, &m_pdevice + 1);
        }

        IDebugCallback* getDebugCallback() const override;

        ~COpenGL_Connection()
        {
            if (m_pdevice)
                delete m_pdevice;
        }

    private:
        COpenGL_Connection(IPhysicalDevice* _pdevice) : m_pdevice(_pdevice) {}

        // 1. Probably could make it a std::unique_ptr?
        // 2. Probably could put it into IAPIConnection itself with
        // a `count` value?
        IPhysicalDevice* m_pdevice = nullptr;
};

using COpenGLConnection = COpenGL_Connection<EAT_OPENGL>;
using COpenGLESConnection = COpenGL_Connection<EAT_OPENGL_ES>;

}

#endif
