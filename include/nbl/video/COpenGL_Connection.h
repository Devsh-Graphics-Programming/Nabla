#ifndef __NBL_C_OPENGL__CONNECTION_H_INCLUDED__
#define __NBL_C_OPENGL__CONNECTION_H_INCLUDED__

#include "nbl/system/ISystem.h"

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

        IDebugCallback* getDebugCallback() const override;

    private:
        COpenGL_Connection(std::vector<std::unique_ptr<IPhysicalDevice>>&& physicalDevices)
            : IAPIConnection(std::move(physicalDevices))
        {}
};

using COpenGLConnection = COpenGL_Connection<EAT_OPENGL>;
using COpenGLESConnection = COpenGL_Connection<EAT_OPENGL_ES>;

}

#endif
