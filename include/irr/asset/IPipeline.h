#ifndef __IRR_I_PIPELINE_H_INCLUDED__
#define __IRR_I_PIPELINE_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include <utility>

namespace irr {
namespace asset
{

template<typename LayoutType>
class IPipeline
{
public:
    inline const LayoutType* getLayout() const { return m_layout.get(); }

protected:
    IPipeline(core::smart_refctd_ptr<core::IReferenceCounted>&& _parent, core::smart_refctd_ptr<LayoutType>&& _layout) :
        /*m_parent(std::move(_parent)), */m_layout(std::move(_layout))
    {

    }
    virtual ~IPipeline() = default;

    //TODO find some way to make this work!! template param to smart_refctd_ptr cannot be type inheriting from IPipeline because it is undefined while IPipeline is being defined!
    //core::smart_refctd_ptr<core::IReferenceCounted> m_parent;//must be IReferenceCounted because no CRTP will work at this point because undefined class is not allowed as arg to is_base_of
    core::smart_refctd_ptr<LayoutType> m_layout;
    bool m_disableOptimizations = false;
    bool m_allowDerivatives = false;
};

}}

#endif