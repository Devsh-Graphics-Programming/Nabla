#ifndef __NBL_C_OPENGL_PRIMARY_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_C_OPENGL_PRIMARY_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/video/COpenGLCommandBuffer.h"
#include "nbl/video/IGPUPrimaryCommandBuffer.h"

namespace nbl {
namespace video
{

class COpenGLPrimaryCommandBuffer final : public COpenGLCommandBuffer, public IGPUPrimaryCommandBuffer
{
    explicit COpenGLPrimaryCommandBuffer(const IGPUCommandPool* _cmdpool) : IGPUCommandBuffer(_cmdpool)
    {

    }

protected:
    ~COpenGLPrimaryCommandBuffer() = default;
};

}
}


#endif
