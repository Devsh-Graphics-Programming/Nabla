#ifndef __NBL_I_GPU_FRAMEBUFFER_H_INCLUDED__
#define __NBL_I_GPU_FRAMEBUFFER_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/asset/IFramebuffer.h"
#include "nbl/video/IGPURenderpass.h"
#include "nbl/video/IGPUImageView.h"

namespace nbl {
namespace video
{

class IGPUFramebuffer : public asset::IFramebuffer<IGPURenderpass, IGPUImageView>, public core::IReferenceCounted
{
    using base_t = asset::IFramebuffer<IGPURenderpass, IGPUImageView>;

public:
    using base_t::base_t;

protected:
    virtual ~IGPUFramebuffer() = default;
};

}
}

#endif
