#ifndef __IRR_I_CPU_TEXTURE_VIEW_H_INCLUDED__
#define __IRR_I_CPU_TEXTURE_VIEW_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/asset/IDescriptor.h"

namespace irr {
namespace asset
{

class ICPUTextureView : public IAsset, public IDescriptor
{
protected:
    virtual ~ICPUTextureView() = default;
};

}}

#endif