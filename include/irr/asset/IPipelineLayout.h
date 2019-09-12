#ifndef __IRR_I_PIPELINE_LAYOUT_H_INCLUDED__
#define __IRR_I_PIPELINE_LAYOUT_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"

namespace irr {
namespace asset
{

template<typename DescLayoutType>
class IPipelineLayout
{
protected:
    virtual ~IPipelineLayout() = default;

public:
    IPipelineLayout(
        core::smart_refctd_ptr<DescLayoutType> _layout0, core::smart_refctd_ptr<DescLayoutType> _layout1 = nullptr,
        core::smart_refctd_ptr<DescLayoutType> _layout2 = nullptr, core::smart_refctd_ptr<DescLayoutType> _layout3 = nullptr
    ) : m_descSetLayouts{_layout0, _layout1, _layout2, _layout3}
    {}

protected:
    core::smart_refctd_ptr<DescLayoutType> m_descSetLayouts[4];
};

}
}

#endif