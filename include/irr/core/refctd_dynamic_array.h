#ifndef __IRR_REFCTD_DYNAMIC_ARRAY_H_INCLUDED__
#define __IRR_REFCTD_DYNAMIC_ARRAY_H_INCLUDED__

#include "irr/core/dynamic_array.h"
#include "irr/core/IReferenceCounted.h"

namespace irr { namespace core
{

template<typename T, class allocator = core::allocator<T>>
class refctd_dynamic_array : public dynamic_array<T, allocator>, public IReferenceCounted
{
    using base_t = dynamic_array<T, allocator>;
public:
    refctd_dynamic_array(size_t _length, const allocator& _alctr = allocator()) : base_t(_length, _alctr) {}
    refctd_dynamic_array(size_t _length, const T& _val, const allocator& _alctr = allocator()) : base_t(_length, _val, _alctr) {}
    refctd_dynamic_array(std::initializer_list<T> _contents, const allocator& _alctr = allocator()) : base_t(_contents, _alctr) {}

protected:
    virtual ~refctd_dynamic_array() = default;
};

}}

#endif