#ifndef __IRR_I_GPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__
#define __IRR_I_GPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__

#include "irr/asset/IDescriptorSetLayout.h"
#include "irr/core/IReferenceCounted.h"

namespace irr { namespace video
{

class IGPUDescriptorSetLayout : public asset::IDescriptorSetLayout, public core::IReferenceCounted
{
public:
    using IDescriptorSetLayout::IDescriptorSetLayout;

protected:
    virtual ~IGPUDescriptorSetLayout() = default;

    bool m_isPushDescLayout = false;
    bool m_canUpdateAfterBind = false;
};

}}

#endif