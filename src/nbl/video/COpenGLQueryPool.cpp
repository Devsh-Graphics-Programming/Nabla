#include "nbl/video/COpenGLQueryPool.h"
#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl::video
{

COpenGLQueryPool::~COpenGLQueryPool()
{
    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->destroyQueryPool(this);
    _NBL_ALIGNED_FREE(lastQueueToUseArray);
}

}