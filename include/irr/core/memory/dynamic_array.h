#ifndef __IRR_DYNAMIC_ARRAY_H_INCLUDED__
#define __IRR_DYNAMIC_ARRAY_H_INCLUDED__

#include "irr/macros.h"
#include "irr/core/Types.h"//for core::allocator

namespace irr
{
namespace core
{

template<typename T, class allocator = core::allocator<T> >
class IRR_FORCE_EBO dynamic_array : public Uncopyable, public Unmovable
{
	public:
		using allocator_type = allocator;
		using value_type = T;
		using pointer = typename std::allocator_traits<allocator_type>::pointer;
		using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;
		using iterator = T*;
		using const_iterator = const T*;

	protected:
		struct alignas(alignof(T)) BaseMembers
		{
			allocator alctr;
			size_t item_count;
		} base;

		_IRR_STATIC_INLINE_CONSTEXPR size_t dummy_item_count = sizeof(BaseMembers)/sizeof(T);


		virtual ~dynamic_array()
		{
			for (size_t i = 0ull; i < base.item_count; ++i)
				std::allocator_traits<allocator>::destroy(base.alctr, data()+i);
		}

	public:
		dynamic_array(size_t _length, const allocator& _alctr = allocator()) : base({_alctr,_length})
		{
			for (size_t i = 0ull; i < base.item_count; ++i)
				std::allocator_traits<allocator>::construct(base.alctr, data() + i);
		}
		dynamic_array(size_t _length, const T& _val, const allocator& _alctr = allocator()) : base({_alctr,_length})
		{
			for (size_t i = 0ull; i < base.item_count; ++i)
				std::allocator_traits<allocator>::construct(base.alctr, data() + i, _val);
		}
		dynamic_array(std::initializer_list<T> _contents, const allocator& _alctr = allocator()) : base({_alctr,_contents.size()})
		{
			for (size_t i = 0ull; i < base.item_count; ++i)
				std::allocator_traits<allocator>::construct(base.alctr, data() + i, *(_contents.begin() + i));
		}

        inline static void* operator new(size_t size) noexcept
        {
			return allocator().allocate(dummy_item_count + size/sizeof(T));
        }
        static inline void* operator new[](size_t size) noexcept
        {
			return allocator().allocate(dummy_item_count + size/sizeof(T));
        }

        static inline void operator delete(void* ptr) noexcept
        {
			return allocator().deallocate(reinterpret_cast<pointer>(ptr));
        }
        static inline void operator delete[](void* ptr) noexcept
        {
			return allocator().deallocate(reinterpret_cast<pointer>(ptr));
        }

		inline bool operator!=(const dynamic_array<T, allocator>& _other) const
		{
			if (size() != _other.size())
				return true;
			for (size_t i = 0u; i < size(); ++i)
				if ((*this)[i] != _other[i])
					return true;
			return false;
		}
		inline bool operator==(const dynamic_array<T, allocator>& _other) const
		{
			return !((*this) != _other);
		}

		inline iterator begin() noexcept { return data(); }
		inline const_iterator begin() const noexcept { return data(); }
		inline iterator end() noexcept { return data()+size(); }
		inline const_iterator end() const noexcept { return data()+size(); }
		inline const_iterator cend() const noexcept { return data()+size(); }
		inline const_iterator cbegin() const noexcept { return data(); }

		inline size_t size() const noexcept { return base.item_count; }
		inline bool empty() const noexcept { return !size(); }

		inline const T& operator[](size_t ix) const noexcept { return data()[ix]; }
		inline T& operator[](size_t ix) noexcept { return data()[ix]; }

		inline T& front() noexcept { return *begin(); }
		inline const T& front() const noexcept { return *begin(); }
		inline T& back() noexcept { return *(end()-1); }
		inline const T& back() const noexcept { return *(end()-1); }
		inline pointer data() noexcept { return reinterpret_cast<T*>(this)+dummy_item_count; }
		inline const_pointer data() const noexcept { return reinterpret_cast<const T*>(this)+dummy_item_count; }
};

}
}

#endif