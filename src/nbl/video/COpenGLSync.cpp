#include "nbl/video/COpenGLSync.h"

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl::video
{
COpenGLSync::~COpenGLSync()
{
    if(sync)
        device->destroySync(sync);
}

}