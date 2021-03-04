#ifndef __NBL_C_OPENGL_COMMAND_POOL_H_INCLUDED__
#define __NBL_C_OPENGL_COMMAND_POOL_H_INCLUDED__

#include "nbl/video/IGPUCommandPool.h"

namespace nbl {
namespace video
{

class COpenGLCommandPool : public IGPUCommandPool
{
public:
    using IGPUCommandPool::IGPUCommandPool;
};

}
}

#endif
