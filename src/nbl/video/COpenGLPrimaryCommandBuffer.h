#ifndef __NBL_C_OPENGL_PRIMARY_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_C_OPENGL_PRIMARY_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/video/COpenGLCommandBuffer.h"
#include "nbl/video/COpenGLFunctionTable.h"

namespace nbl {
namespace video
{

class COpenGLPrimaryCommandBuffer final : public COpenGLCommandBuffer
{
public:
    explicit COpenGLPrimaryCommandBuffer(const IGPUCommandPool* _cmdpool) : IGPUCommandBuffer(_cmdpool)
    {

    }

    void executeAll(const COpenGLFunctionTable* gl) const
    {
        for (const SCommand& cmd : m_commands)
        {
            switch (cmd.type)
            {
            case ECT_BEGIN_QUERY:
                // TODO call GL using given function table
                break;
            // ....
            // TODO impl all commands
            }
        }
    }

protected:
    ~COpenGLPrimaryCommandBuffer() = default;
};

}
}


#endif
