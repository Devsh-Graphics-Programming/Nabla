#ifndef __NBL_I_GPU_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_I_GPU_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/asset/ICommandBuffer.h"

#include "nbl/video/IGPUImage.h"
#include "nbl/video/IGPUImageView.h"
#include "nbl/video/IGPURenderpass.h"
#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/IGPUGraphicsPipeline.h"
#include "nbl/video/IGPUComputePipeline.h"
#include "nbl/video/IGPUEvent.h"
#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/video/IGPUPipelineLayout.h"
#include "nbl/video/IGPUCommandPool.h"

namespace nbl::video
{

class IGPUCommandBuffer :
    public core::IReferenceCounted,
    public asset::ICommandBuffer<
        IGPUBuffer,
        IGPUImage,
        IGPUImageView,
        IGPURenderpass,
        IGPUFramebuffer,
        IGPUGraphicsPipeline,
        IGPUComputePipeline,
        IGPUDescriptorSet,
        IGPUPipelineLayout,
        IGPUEvent,
        IGPUCommandBuffer
    >,
    public IBackendObject
{
    using base_t = asset::ICommandBuffer<
        IGPUBuffer,
        IGPUImage,
        IGPUImageView,
        IGPURenderpass,
        IGPUFramebuffer,
        IGPUGraphicsPipeline,
        IGPUComputePipeline,
        IGPUDescriptorSet,
        IGPUPipelineLayout,
        IGPUEvent,
        IGPUCommandBuffer
    >;

public:
    virtual void begin(uint32_t _flags) override
    {
        asset::ICommandBuffer<
            IGPUBuffer,
            IGPUImage,
            IGPUImageView,
            IGPURenderpass,
            IGPUFramebuffer,
            IGPUGraphicsPipeline,
            IGPUComputePipeline,
            IGPUDescriptorSet,
            IGPUPipelineLayout,
            IGPUEvent,
            IGPUCommandBuffer
        >::begin(_flags);
        if ((m_cmdpool->getCreationFlags()&IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT)==0u)
        {
            assert(m_state != ES_INITIAL);
        }
    }

    uint32_t getQueueFamilyIndex() const { return m_cmdpool->getQueueFamilyIndex(); }

    IGPUCommandPool* getPool() const { return m_cmdpool.get(); }

protected:
    IGPUCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, E_LEVEL lvl, IGPUCommandPool* _cmdpool) : base_t(lvl), IBackendObject(std::move(dev)), m_cmdpool(_cmdpool)
    {
    }
    virtual ~IGPUCommandBuffer() = default;

    core::smart_refctd_ptr<IGPUCommandPool> m_cmdpool;


    static void bindDescriptorSets_generic(const IGPUPipelineLayout* _newLayout, uint32_t _first, uint32_t _count, const IGPUDescriptorSet* const* _descSets, const IGPUPipelineLayout** const _destPplnLayouts)
    {
        int32_t compatibilityLimits[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT]{};
        for (uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
        {
            const int32_t lim = _destPplnLayouts[i] ? //if no descriptor set bound at this index
                _destPplnLayouts[i]->isCompatibleUpToSet(IGPUPipelineLayout::DESCRIPTOR_SET_COUNT - 1u, _newLayout) : -1;

            compatibilityLimits[i] = lim;
        }

        /*
        https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#descriptorsets-compatibility
        When binding a descriptor set (see Descriptor Set Binding) to set number N, if the previously bound descriptor sets for sets zero through N-1 were all bound using compatible pipeline layouts, then performing this binding does not disturb any of the lower numbered sets.
        */
        for (int32_t i = 0; i < static_cast<int32_t>(_first); ++i)
            if (compatibilityLimits[i] < i)
                _destPplnLayouts[i] = nullptr;
        /*
        If, additionally, the previous bound descriptor set for set N was bound using a pipeline layout compatible for set N, then the bindings in sets numbered greater than N are also not disturbed.
        */
        if (compatibilityLimits[_first] < static_cast<int32_t>(_first))
            for (uint32_t i = _first + 1u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
                _destPplnLayouts[i] = nullptr;
    }
};

}

#endif
