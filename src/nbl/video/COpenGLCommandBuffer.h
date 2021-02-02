#ifndef __NBL_C_OPENGL_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_C_OPENGL_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/core/Types.h"

namespace nbl {
namespace video
{

class COpenGLCommandBuffer : public virtual IGPUCommandBuffer
{
public:
    // this init of IGPUCommandBuffer will be always ignored by compiler since COpenGLCommandBuffer will never be most derived class
    COpenGLCommandBuffer() : IGPUCommandBuffer(nullptr) {}

protected:
    virtual ~COpenGLCommandBuffer() = default;

    enum E_COMMAND_TYPE
    {
        ECT_BEGIN_QUERY,
        //.... TODO
    };
    struct SCommand
    {
        E_COMMAND_TYPE type;
        union 
        {
            // ... TODO (structs for all possible commands)
        };
    };
    
    core::vector<SCommand> m_commands;
};

}
}

#endif
