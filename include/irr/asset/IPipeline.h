#ifndef __IRR_I_PIPELINE_H_INCLUDED__
#define __IRR_I_PIPELINE_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"//for smart_refctd_ptr
#include <utility>

namespace irr {
namespace asset
{

template<typename LayoutType>
class IPipeline
{
protected:
    IPipeline(core::smart_refctd_ptr<LayoutType>&& _layout) : m_layout(std::move(_layout)) {};
    virtual ~IPipeline() = default;

    core::smart_refctd_ptr<LayoutType> m_layout;
    bool m_disableOptimizations = false;
    bool m_allowDerivatives = false;
};

}}

#endif