#ifndef _NBL_VIDEO_I_GPU_FRAMEBUFFER_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_FRAMEBUFFER_H_INCLUDED_


#include "nbl/asset/IFramebuffer.h"

#include "nbl/video/IGPURenderpass.h"
#include "nbl/video/IGPUImageView.h"


namespace nbl::video
{

class IGPUFramebuffer : public IBackendObject, public asset::IFramebuffer<IGPURenderpass,IGPUImageView>
{
        using base_t = asset::IFramebuffer<IGPURenderpass,IGPUImageView>;

    public:
        IGPUFramebuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params) : IBackendObject(std::move(dev)), base_t(std::move(params)) {}

    protected:
        virtual ~IGPUFramebuffer() = default;
};

}

#endif
