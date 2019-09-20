#ifndef __IRR_I_GPU_TEXTURE_VIEW_H_INCLUDED__
#define __IRR_I_GPU_TEXTURE_VIEW_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/IDescriptor.h"

namespace irr {
namespace video
{

class IGPUTextureView : public core::IReferenceCounted, public asset::IDescriptor
{
protected:
    virtual ~IGPUTextureView() = default;
};

}}

#endif