#ifndef __NBL_I_GPU_FRAMEBUFFER_H_INCLUDED__
#define __NBL_I_GPU_FRAMEBUFFER_H_INCLUDED__


#include "nbl/core/IReferenceCounted.h"

#include "nbl/asset/IFramebuffer.h"

#include "nbl/video/IGPURenderpass.h"
#include "nbl/video/IGPUImageView.h"


namespace nbl::video
{

class NBL_API IGPUFramebuffer : public asset::IFramebuffer<IGPURenderpass, IGPUImageView>, public core::IReferenceCounted, public IBackendObject
{
        using base_t = asset::IFramebuffer<IGPURenderpass, IGPUImageView>;

    public:
        IGPUFramebuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params) : base_t(std::move(params)), IBackendObject(std::move(dev)) {}

    protected:
        virtual ~IGPUFramebuffer() = default;
};

}

#endif
