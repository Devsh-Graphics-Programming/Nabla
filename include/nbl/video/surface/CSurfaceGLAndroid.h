#ifndef __NBL_C_SURFACE_GL_ANDROID_H_INCLUDED__
#define __NBL_C_SURFACE_GL_ANDROID_H_INCLUDED__

#include "nbl/core/compile_config.h"

#ifdef _NBL_PLATFORM_ANDROID_

#include "nbl/video/surface/ISurfaceAndroid.h"
#include "nbl/video/surface/ISurfaceGL.h"

namespace nbl {
namespace video
{

class CSurfaceGLAndroid final : public ISurfaceAndroid, public ISurfaceGL
{
public:
    explicit CSurfaceGLAndroid(SCreationParams&& params) : ISurfaceAndroid(std::move(params)), ISurfaceGL(params.anw)
    {

    }
};

}
}

#endif //_NBL_PLATFORM_ANDROID_

#endif //__NBL_C_SURFACE_GL_ANDROID_H_INCLUDED__