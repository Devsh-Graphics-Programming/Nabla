#ifndef __IRR_I_GPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__
#define __IRR_I_GPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__

#include "irr/asset/IDescriptorSetLayout.h"

#include "irr/video/IGPUSampler.h"

namespace irr
{
namespace video
{

//! GPU Version of Descriptor Set Layout
/*
    @see IDescriptorSetLayout
*/

class IGPUDescriptorSetLayout : public asset::IDescriptorSetLayout<IGPUSampler>
{
public:
    using IDescriptorSetLayout<IGPUSampler>::IDescriptorSetLayout;

protected:
    virtual ~IGPUDescriptorSetLayout() = default;

    bool m_isPushDescLayout = false;
    bool m_canUpdateAfterBind = false;
};

}
}

#endif