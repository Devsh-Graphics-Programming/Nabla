#ifndef __IRR_I_GPU_PRIMARY_COMMAND_BUFFER_H_INCLUDED__
#define __IRR_I_GPU_PRIMARY_COMMAND_BUFFER_H_INCLUDED__

#include <nbl/video/IGPUCommandBuffer.h>

namespace nbl
{
namespace video
{
class IGPUPrimaryCommandBuffer : public IGPUCommandBuffer
{
    using base_t = IGPUCommandBuffer;

public:
    using base_t::base_t;

    E_LEVEL getLevel() const override { return EL_PRIMARY; }

    //passthrough, more specific impl in backend-specific classes
    void begin(uint32_t _flags) override
    {
        base_t::begin(_flags);
    }

    //passthrough, more specific impl in backend-specific classes
    void reset(uint32_t _flags) override
    {
        base_t::reset(_flags);
    }

    //passthrough, more specific impl in backend-specific classes
    void end() override
    {
        base_t::end();
    }

    //there's no IGPUSecondaryCommandBuffer yet
    //virtual void executeCommands(uint32_t commandBufferCount, IGPUSecondaryCommandBuffer* pCommandBuffer) = 0;
};

}
}

#endif