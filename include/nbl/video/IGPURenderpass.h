#ifndef __NBL_I_GPU_RENDERPASS_H_INCLUDED__
#define __NBL_I_GPU_RENDERPASS_H_INCLUDED__

#include "nbl/asset/IRenderpass.h"
#include "nbl/video/IBackendObject.h"

namespace nbl {
namespace video
{

class IGPURenderpass : public asset::IRenderpass, public core::IReferenceCounted, public IBackendObject
{
    using base_t = asset::IRenderpass;

public:
    IGPURenderpass(ILogicalDevice* dev, const SCreationParams& params) : base_t(params), IBackendObject(dev) {}
};

}
}

#endif