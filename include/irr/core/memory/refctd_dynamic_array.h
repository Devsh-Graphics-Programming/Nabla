#ifndef __IRR_REFCTD_DYNAMIC_ARRAY_H_INCLUDED__
#define __IRR_REFCTD_DYNAMIC_ARRAY_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/core/alloc/AlignedBase.h"
#include "irr/core/memory/dynamic_array.h"

namespace irr
{
namespace core
{

template<typename T, class allocator = core::allocator<T>>
class IRR_FORCE_EBO refctd_dynamic_array : public IReferenceCounted, public dynamic_array<T,allocator,refctd_dynamic_array<T,allocator> >
{
		using base_t = dynamic_array<T, allocator, refctd_dynamic_array<T, allocator> >;
		friend class base_t;
		static_assert(sizeof(base_t) == sizeof(impl::dynamic_array_base<T, allocator>), "memory has been added to dynamic_array");
		static_assert(sizeof(base_t) == sizeof(dynamic_array<T, allocator>),"non-CRTP and CRTP base class definitions differ in size");

		class IRR_FORCE_EBO fake_size_class : public IReferenceCounted, public dynamic_array<T, allocator> {};
	public:
		_IRR_STATIC_INLINE_CONSTEXPR size_t dummy_item_count = sizeof(fake_size_class)/sizeof(T);

		_IRR_RESOLVE_NEW_DELETE_AMBIGUITY(base_t) // only want new and delete operators from `dynamic_array`
	protected:
		friend class base_t;

		refctd_dynamic_array(size_t _length, const allocator& _alctr = allocator()) : base_t(_length, _alctr) {}
		refctd_dynamic_array(size_t _length, const T& _val, const allocator& _alctr = allocator()) : base_t(_length, _val, _alctr) {}
		refctd_dynamic_array(std::initializer_list<T> _contents, const allocator& _alctr = allocator()) : base_t(std::move(_contents), _alctr) {}

		virtual ~refctd_dynamic_array() = default;
};

}}

#endif