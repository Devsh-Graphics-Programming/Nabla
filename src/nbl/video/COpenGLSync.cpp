#include "nbl/video/COpenGLSync.h"

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl {
namespace video
{

COpenGLSync::~COpenGLSync()
{
    if (sync)
        device->destroySync(sync);
}

}
}