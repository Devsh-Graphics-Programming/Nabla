#ifndef _NBL_C_OPENGL__CONNECTION_H_INCLUDED_
#define _NBL_C_OPENGL__CONNECTION_H_INCLUDED_

#include "nbl/system/ISystem.h"

#include "nbl/video/IAPIConnection.h"
#include "nbl/video/debug/COpenGLDebugCallback.h"
#include "nbl/video/CEGL.h"

namespace nbl::video
{

template<E_API_TYPE API_TYPE>
class COpenGL_Connection final : public IAPIConnection
{
    public:
        //
        struct SStuff
        {
            // to load function pointers, make EGL context current and use `egl->call.peglGetProcAddress("glFuncname")`
        };

        //
        static core::smart_refctd_ptr<COpenGL_Connection<API_TYPE>> create(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName, COpenGLDebugCallback&& dbgCb);

        E_API_TYPE getAPIType() const override
        {
            return API_TYPE;
        }

        IDebugCallback* getDebugCallback() const override;

        const egl::CEGL& getInternalObject() const;

    private:
        COpenGL_Connection() : IAPIConnection()
        {}
};

using COpenGLConnection = COpenGL_Connection<EAT_OPENGL>;
using COpenGLESConnection = COpenGL_Connection<EAT_OPENGL_ES>;

}

#endif
