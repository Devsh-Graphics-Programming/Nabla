#ifndef _NBL_C_OPENGL__CONNECTION_H_INCLUDED_
#define _NBL_C_OPENGL__CONNECTION_H_INCLUDED_

#include "nbl/system/ISystem.h"

#include "nbl/video/IAPIConnection.h"
#include "nbl/video/debug/COpenGLDebugCallback.h"
#include "nbl/video/CEGL.h"

namespace nbl::video
{

template<E_API_TYPE API_TYPE>
class NBL_API2 COpenGL_Connection final : public IAPIConnection
{
    public:
        //
        static core::smart_refctd_ptr<COpenGL_Connection<API_TYPE>> create(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName, COpenGLDebugCallback&& dbgCb);

        inline E_API_TYPE getAPIType() const override
        {
            return API_TYPE;
        }

        IDebugCallback* getDebugCallback() const override;

        const egl::CEGL& getInternalObject() const;

    private:
        inline COpenGL_Connection(const SFeatures& enabledFeatures, core::smart_refctd_ptr<asset::CGLSLCompiler>&& glslc) : IAPIConnection(enabledFeatures, std::move(glslc))
        {}
};

using COpenGLConnection = COpenGL_Connection<EAT_OPENGL>;
using COpenGLESConnection = COpenGL_Connection<EAT_OPENGL_ES>;

}

#endif
