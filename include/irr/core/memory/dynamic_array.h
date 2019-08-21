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
		struct alignas(value_type) BaseMembers
		{
			allocator alctr;
			size_t item_count;
		};
		BaseMembers base;

		_IRR_STATIC_INLINE_CONSTEXPR size_t dummy_item_count = sizeof(BaseMembers)/sizeof(T);


		dynamic_array(size_t _length, const allocator& _alctr = allocator()) : base({ _alctr,_length })
		{
			for (size_t i = 0ull; i < base.item_count; ++i)
				std::allocator_traits<allocator>::construct(base.alctr, data() + i);
		}
		dynamic_array(size_t _length, const T& _val, const allocator& _alctr = allocator()) : base({ _alctr,_length })
		{
			for (size_t i = 0ull; i < base.item_count; ++i)
				std::allocator_traits<allocator>::construct(base.alctr, data() + i, _val);
		}
		dynamic_array(std::initializer_list<T>&& _contents, const allocator& _alctr = allocator()) : base({ _alctr,_contents.size() })
		{
			for (size_t i = 0ull; i < base.item_count; ++i)
				std::allocator_traits<allocator>::construct(base.alctr, data() + i, *(_contents.begin() + i));
		}
	public:
		virtual ~dynamic_array()
		{
			for (size_t i = 0ull; i < base.item_count; ++i)
				std::allocator_traits<allocator>::destroy(base.alctr, data() + i);
		}

		static inline void* allocate_dynamic_array(size_t length, const allocator& _alctr = allocator())
		{
			return allocator().allocate(dummy_item_count+length);
		}
		// factory method to use instead of `new`
		template<class U>
		static inline U* create_dynamic_array(size_t length, const allocator& _alctr = allocator())
		{
			void* ptr = allocate_dynamic_array(length,_alctr);
			return new(ptr) U(length,_alctr);
		}
		template<class U>
		static inline U* create_dynamic_array(size_t _length, const T& _val, const allocator& _alctr = allocator())
		{
			void* ptr = allocate_dynamic_array(length,_alctr);
			return new(ptr) U(length,_val,_alctr);
		}
		template<class U>
		static inline U* create_dynamic_array(std::initializer_list<T> _contents, const allocator& _alctr = allocator())
		{
			void* ptr = allocate_dynamic_array(_contents.size(), _alctr);
			return new(ptr) U(std::move(_contents), _alctr);
		}

		// the usual new allocation operator won't let us analyze the constructor arguments to decide the size of the object
		static void* operator new(size_t size) = delete;
		// we cannot new or delete arrays of `dynamic_array` (non uniform sizes)
		static void* operator new[](size_t size) = delete;

        static inline void operator delete(void* ptr) noexcept
        {
			return allocator().deallocate(reinterpret_cast<pointer>(ptr));
        }
		// size hint is ill-formed
		static void operator delete(void* ptr, std::size_t sz) = delete;
		// no arrays
		static void operator delete[](void* ptr) = delete;
		static void operator delete[](void* ptr, std::size_t sz) = delete;
#if __cplusplus >= 201703L
		static void* operator new(size_t size, std::align_val_t al) = delete;
		static void* operator new[](size_t size, std::align_val_t al) = delete;

		// TODO: Support for C++17 align allocator
		static void operator delete(void* ptr, std::align_val_t al) = delete;
		static void operator delete(void* ptr, std::size_t sz, std::align_val_t al) = delete;
		// no arrays
		static void operator delete[](void* ptr, std::align_val_t al) = delete;
		static void operator delete[](void* ptr, std::size_t sz, std::align_val_t al) = delete;
#if __cplusplus >= 201704L // change later when c++20 is standardised
		static void operator delete(dynamic_array<T, allocator>* ptr, std::destroying_delete_t) = delete;
		static void operator delete(dynamic_array<T, allocator>* ptr, std::destroying_delete_t, std::align_val_t al) = delete;
		static void operator delete(dynamic_array<T, allocator>* ptr, std::destroying_delete_t, std::size_t sz) = delete;
		static void operator delete(dynamic_array<T, allocator>* ptr, std::destroying_delete_t, std::size_t sz, std::align_val_t al) = delete;
#endif
#endif

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