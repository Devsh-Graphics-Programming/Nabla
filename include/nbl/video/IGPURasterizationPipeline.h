#ifndef _NBL_I_GPU_RASTERIZATION_PIPELINE_H_INCLUDED_
#define _NBL_I_GPU_RASTERIZATION_PIPELINE_H_INCLUDED_

#include "nbl/asset/IRasterizationPipeline.h"

#include "nbl/video/IGPUPipelineLayout.h"
#include "nbl/video/IGPURenderpass.h"
#include "nbl/video/IGPUPipeline.h"

namespace nbl::video
{

//this is going to be made pure virtual (basically just a name)
//IGPUGraphicsPipeline needs to use IGPUPipeline<IGraphicsPipeline>, but IGPURaster would need IGPUPipeline<IRasterizationPipeline>
//skipping IGraphics would NOT be good, some data would be lost
//and templating ICPURaster for IGraphics would defeat the purpose of polymorphism
//the alternative would be to rework IGPUPipeline or to rework 

//pure virtual
class IGPURasterizationPipeline // : public IGPUPipeline<asset::IRasterizationPipeline<const IGPUPipelineLayout, const IGPURenderpass>>
{
    /*
    using pipeline_t = asset::IRasterizationPipeline<const IGPUPipelineLayout, const IGPURenderpass>;

    public:
        struct SCreationParams : public SPipelineCreationParams<const IGPURasterizationPipeline>
        {
            public:
            #define base_flag(F) static_cast<uint64_t>(pipeline_t::FLAGS::F)
            enum class FLAGS : uint64_t
            {
                NONE = base_flag(NONE),
                DISABLE_OPTIMIZATIONS = base_flag(DISABLE_OPTIMIZATIONS),
                ALLOW_DERIVATIVES = base_flag(ALLOW_DERIVATIVES),
                VIEW_INDEX_FROM_DEVICE_INDEX = 1<<3,
                FAIL_ON_PIPELINE_COMPILE_REQUIRED = base_flag(FAIL_ON_PIPELINE_COMPILE_REQUIRED),
                EARLY_RETURN_ON_FAILURE = base_flag(EARLY_RETURN_ON_FAILURE),
            };
            #undef base_flag

            inline core::bitflag<FLAGS>& getFlags() { return flags; }

            inline core::bitflag<FLAGS> getFlags() const { return flags; }
            
            const IGPUPipelineLayout* layout = nullptr;
            renderpass_t* renderpass = nullptr;

            // TODO: Could guess the required flags from SPIR-V introspection of declared caps
            core::bitflag<FLAGS> flags = FLAGS::NONE;
        };

        inline core::bitflag<SCreationParams::FLAGS> getCreationFlags() const {return m_flags;}

    protected:
        IGPURasterizationPipeline(const IRasterizationPipelineBase::SCreationParams& params, IRasterizationPipelineBase::SCachedCreationParams const& cached) :
            IGPUPipeline(core::smart_refctd_ptr<const ILogicalDevice>(params.layout->getOriginDevice()), params.layout, cached, params.renderpass), m_flags(params.flags)
        {}
        virtual ~IGPURasterizationPipeline() override = default;
        
        const core::bitflag<SCreationParams::FLAGS> m_flags;
    */
};


}


#endif