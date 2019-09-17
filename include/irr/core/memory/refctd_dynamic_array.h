#ifndef __IRR_REFCTD_DYNAMIC_ARRAY_H_INCLUDED__
#define __IRR_REFCTD_DYNAMIC_ARRAY_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/core/alloc/AlignedBase.h"
#include "irr/core/memory/dynamic_array.h"

namespace irr
{
namespace core
{

template<typename T, class allocator = allocator<T>>
class IRR_FORCE_EBO refctd_dynamic_array : public IReferenceCounted, public dynamic_array<T,allocator,refctd_dynamic_array<T,allocator> >
{
		friend class dynamic_array<T, allocator, refctd_dynamic_array<T, allocator> >;
		using base_t = dynamic_array<T, allocator, refctd_dynamic_array<T, allocator> >;

		static_assert(sizeof(base_t) == sizeof(impl::dynamic_array_base<T, allocator>), "memory has been added to dynamic_array");
		static_assert(sizeof(base_t) == sizeof(dynamic_array<T, allocator>),"non-CRTP and CRTP base class definitions differ in size");

		class IRR_FORCE_EBO fake_size_class : public IReferenceCounted, public dynamic_array<T, allocator> {};
	public:
		_IRR_STATIC_INLINE_CONSTEXPR size_t dummy_item_count = sizeof(fake_size_class)/sizeof(T);

		_IRR_RESOLVE_NEW_DELETE_AMBIGUITY(base_t) // only want new and delete operators from `dynamic_array`
	protected:

		inline refctd_dynamic_array(size_t _length, const allocator& _alctr = allocator()) : base_t(_length, _alctr) {}
		inline refctd_dynamic_array(size_t _length, const T& _val, const allocator& _alctr = allocator()) : base_t(_length, _val, _alctr) {}
		template<typename container_t, typename iterator_t = typename container_t::iterator>
		inline refctd_dynamic_array(const container_t& _containter, const allocator& _alctr = allocator()) : base_t(_containter, _alctr) {}
		template<typename container_t, typename iterator_t = typename container_t::iterator>
		inline refctd_dynamic_array(container_t&& _containter, const allocator& _alctr = allocator()) : base_t(std::move(_containter),_alctr) {}

		virtual ~refctd_dynamic_array() = default;
};


template<typename T, class allocator = allocator<T> >
using smart_refctd_dynamic_array = smart_refctd_ptr<refctd_dynamic_array<T, allocator> >;

template<class smart_refctd_dynamic_array_type, typename... Args>
inline smart_refctd_dynamic_array_type make_refctd_dynamic_array(Args&&... args)
{
	return smart_refctd_dynamic_array_type(smart_refctd_dynamic_array_type::pointee::create_dynamic_array(std::forward<Args>(args)...),dont_grab);
}


}
}

#endif