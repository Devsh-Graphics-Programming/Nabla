#ifndef __NBL_I_GPU_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_I_GPU_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/asset/ICommandBuffer.h"

#include "nbl/video/IGPUImage.h"
#include "nbl/video/IGPURenderpass.h"
#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/IGPURenderpassIndependentPipeline.h"
#include "nbl/video/IGPUComputePipeline.h"
#include "nbl/video/IGPUEvent.h"
#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/video/IGPUPipelineLayout.h"
#include "nbl/video/IGPUCommandPool.h"

namespace nbl {
namespace video
{

class IGPUCommandBuffer :
    public core::IReferenceCounted,
    public asset::ICommandBuffer<
        IGPUBuffer,
        IGPUImage,
        IGPURenderpass,
        IGPUFramebuffer,
        IGPURenderpassIndependentPipeline, // TODO change to IGPUGraphicsPipeline
        IGPUComputePipeline,
        IGPUDescriptorSet,
        IGPUPipelineLayout,
        IGPUEvent
    >
{
public:
    // TODO impl commands

    uint32_t getQueueFamilyIndex() const { return m_cmdpool->getQueueFamilyIndex(); }

protected:
    explicit IGPUCommandBuffer(const IGPUCommandPool* _cmdpool) : m_cmdpool(_cmdpool)
    {

    }
    virtual ~IGPUCommandBuffer() = default;

    const IGPUCommandPool* m_cmdpool; // not owning
};

}
}

#endif
