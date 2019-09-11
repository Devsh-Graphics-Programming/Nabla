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
    void setDescSetLayout(core::smart_refctd_ptr<DescLayoutType> _layout, uint32_t _num)
    {
        assert(_num < 4u);
        m_descSetLayouts[_num] = _layout;
    }

protected:
    core::smart_refctd_ptr<DescLayoutType> m_descSetLayouts[4];
};

}
}

#endif