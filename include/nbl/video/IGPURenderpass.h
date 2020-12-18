#ifndef __NBL_I_GPU_RENDERPASS_H_INCLUDED__
#define __NBL_I_GPU_RENDERPASS_H_INCLUDED__

#include "nbl/asset/IRenderpass.h"

namespace nbl {
namespace video
{

class IGPURenderpass : public asset::IRenderpass, public core::IReferenceCounted
{
    using base_t = asset::IRenderpass;

public:
    using base_t::base_t;
};

}
}

#endif