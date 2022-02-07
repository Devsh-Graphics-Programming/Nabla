#ifndef __NBL_I_GPU_RENDERPASS_H_INCLUDED__
#define __NBL_I_GPU_RENDERPASS_H_INCLUDED__

#include "nbl/asset/IRenderpass.h"

#include "nbl/video/decl/IBackendObject.h"

namespace nbl::video
{
class IGPURenderpass : public asset::IRenderpass, public core::IReferenceCounted, public IBackendObject
{
    using base_t = asset::IRenderpass;

public:
    IGPURenderpass(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const SCreationParams& params)
        : base_t(params), IBackendObject(std::move(dev)) {}
};

}

#endif