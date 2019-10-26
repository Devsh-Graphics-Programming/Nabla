#ifndef __IRR_DYNAMIC_ARRAY_H_INCLUDED__
#define __IRR_DYNAMIC_ARRAY_H_INCLUDED__

#include "irr/macros.h"
#include "irr/core/Types.h"//for core::allocator

namespace irr
{
namespace core
{

namespace impl
{
	template<typename T, class allocator>
	class alignas(T) IRR_FORCE_EBO dynamic_array_base // : public Uncopyable, public Unmovable // cannot due to diamon inheritance
	{
		protected:
			allocator alctr;
			size_t item_count;

			dynamic_array_base(const allocator& _alctr, size_t _item_count) : alctr(_alctr), item_count(_item_count) {}

			virtual ~dynamic_array_base() {} // do not remove, need for class size computation!
		public:
			_IRR_NO_COPY_FINAL(dynamic_array_base);
			_IRR_NO_MOVE_FINAL(dynamic_array_base);
	};
}

template<typename T, class allocator = core::allocator<T>, class CRTP=void>
class IRR_FORCE_EBO dynamic_array : public impl::dynamic_array_base<T,allocator>
{
		using base_t = impl::dynamic_array_base<T,allocator>;

	public:
		using allocator_type = allocator;
		using value_type = T;
		using pointer = typename std::allocator_traits<allocator_type>::pointer;
		using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;
		using iterator = T*;
		using const_iterator = const T*;

		using this_real_type = typename std::conditional<std::is_void<CRTP>::value, dynamic_array<T, allocator>, CRTP>::type;

	protected:
		inline dynamic_array(size_t _length, const allocator& _alctr = allocator()) : base_t( _alctr,_length )
		{
			for (size_t i = 0ull; i < base_t::item_count; ++i)
				std::allocator_traits<allocator>::construct(base_t::alctr, data() + i);
		}
		inline dynamic_array(size_t _length, const T& _val, const allocator& _alctr = allocator()) : base_t( _alctr,_length )
		{
			for (size_t i = 0ull; i < base_t::item_count; ++i)
				std::allocator_traits<allocator>::construct(base_t::alctr, data() + i, _val);
		}
		template<typename container_t, typename iterator_t = typename container_t::iterator>
		inline dynamic_array(const container_t& _containter, const allocator& _alctr = allocator()) : base_t( _alctr,_containter.size())
		{
			auto it = _containter.begin();
			for (size_t i = 0ull; i < base_t::item_count; ++i)
				std::allocator_traits<allocator>::construct(base_t::alctr, data() + i, *(it++));
		}
		template<typename container_t, typename iterator_t = typename container_t::iterator>
		inline dynamic_array(container_t&& _containter, const allocator& _alctr = allocator()) : base_t( _alctr,_containter.size())
		{
			auto it = _containter.begin();
			for (size_t i = 0ull; i < base_t::item_count; ++i)
				std::allocator_traits<allocator>::construct(base_t::alctr, data() + i, std::move(*(it++)));
		}

	public:
		_IRR_STATIC_INLINE_CONSTEXPR size_t dummy_item_count = sizeof(base_t)/sizeof(T);

		virtual ~dynamic_array()
		{
			for (size_t i = 0ull; i < base_t::item_count; ++i)
				std::allocator_traits<allocator>::destroy(base_t::alctr, data() + i);
		}

		static inline size_t size_of(size_t length)
		{
			return (this_real_type::dummy_item_count + length) * sizeof(T);
		}
		template<typename container_t, typename iterator_t = typename container_t::iterator>
		static inline size_t size_of(const container_t& _containter)
		{
			return (this_real_type::dummy_item_count + _containter.size()) * sizeof(T);
		}

		static inline void* allocate_dynamic_array(size_t length)
		{
            auto gccHappyVar = allocator();
			return std::allocator_traits<allocator>::allocate(gccHappyVar, this_real_type::size_of(length) / sizeof(T));
		}
		template<typename container_t, typename iterator_t = typename container_t::iterator>
		static inline void* allocate_dynamic_array(const container_t& _containter)
		{
    		auto gccHappyVar = allocator();
				return std::allocator_traits<allocator>::allocate(gccHappyVar, this_real_type::size_of(_containter) / sizeof(T));
		}
		static inline void* allocate_dynamic_array(size_t length, allocator& _alctr)
		{
			return std::allocator_traits<allocator>::allocate(_alctr,this_real_type::size_of(length)/sizeof(T));
		}
		template<typename container_t, typename iterator_t = typename container_t::iterator>
		static inline void* allocate_dynamic_array(const container_t& _containter, allocator& _alctr)
		{
			return std::allocator_traits<allocator>::allocate(_alctr, this_real_type::size_of(_containter)/sizeof(T));
		}
		// factory method to use instead of `new`
		template<typename... Args>
		static inline this_real_type* create_dynamic_array(Args&&... args)
		{
			void* ptr = this_real_type::allocate_dynamic_array(args...);
			return new(ptr) this_real_type(std::forward<Args>(args)...);
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

		inline size_t size() const noexcept { return base_t::item_count; }
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