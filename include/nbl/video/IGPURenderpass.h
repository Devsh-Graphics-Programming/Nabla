#ifndef _NBL_VIDEO_I_GPU_RENDERPASS_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_RENDERPASS_H_INCLUDED_


#include "nbl/asset/IRenderpass.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IGPURenderpass : public core::IReferenceCounted, public asset::IRenderpass, public IBackendObject
{
        using base_t = asset::IRenderpass;

    public:
        inline IGPURenderpass(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const SCreationParams& params, const SCreationParamValidationResult& counts) : base_t(params,counts), IBackendObject(std::move(dev)) {}
};

}

#endif