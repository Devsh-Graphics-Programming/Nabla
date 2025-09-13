// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifndef _NBL_CORE_DECL_SMART_REFCTD_PTR_H_INCLUDED_
#define _NBL_CORE_DECL_SMART_REFCTD_PTR_H_INCLUDED_

#include "nbl/core/IReferenceCounted.h"

namespace nbl::core
{

// Parameter types for special overloaded constructors
struct dont_grab_t {};
constexpr dont_grab_t dont_grab{};
struct dont_drop_t {};
constexpr dont_drop_t dont_drop{};

// A RAII-like class to help you safeguard against memory leaks.
// Will automagically drop reference counts when it goes out of scope
template<class I_REFERENCE_COUNTED>
class smart_refctd_ptr
{			
		mutable I_REFERENCE_COUNTED* ptr; // since IReferenceCounted declares the refcount mutable atomic

		template<class U> friend class smart_refctd_ptr;
		template<class U, class T> friend smart_refctd_ptr<U> smart_refctd_ptr_static_cast(smart_refctd_ptr<T>&&);
		template<class U, class T> friend smart_refctd_ptr<U> smart_refctd_ptr_dynamic_cast(smart_refctd_ptr<T>&&);

        template<class U>
		void copy(const smart_refctd_ptr<U>& other) noexcept;
        template<class U>
        void move(smart_refctd_ptr<U>&& other) noexcept
        {
            ptr = other.ptr;
            other.ptr = nullptr;
        }
	public:
		using pointee = I_REFERENCE_COUNTED;
		using value_type = I_REFERENCE_COUNTED*;

		struct hash
		{
			size_t operator() (const core::smart_refctd_ptr<I_REFERENCE_COUNTED>& ptr) const;
		};

		constexpr smart_refctd_ptr() noexcept : ptr(nullptr) {}
		constexpr smart_refctd_ptr(std::nullptr_t) noexcept : ptr(nullptr) {}
		template<class U>
		explicit smart_refctd_ptr(U* _pointer) noexcept;
		template<class U>
		explicit smart_refctd_ptr(U* _pointer, dont_grab_t t) noexcept : ptr(_pointer) {}
			
        template<class U> requires (!std::is_same_v<U,I_REFERENCE_COUNTED>)
        inline smart_refctd_ptr(const smart_refctd_ptr<U>& other) noexcept
        {
            this->copy(other);
        }
		inline smart_refctd_ptr(const smart_refctd_ptr<I_REFERENCE_COUNTED>& other) noexcept
        {
            this->copy(other);
        }

		template<class U> requires (!std::is_same_v<U,I_REFERENCE_COUNTED>)
		inline smart_refctd_ptr(smart_refctd_ptr<U>&& other) noexcept
		{
            this->move(std::move(other));
		}
        inline smart_refctd_ptr(smart_refctd_ptr<I_REFERENCE_COUNTED>&& other) noexcept
        {
            this->move(std::move(other));
        }

		~smart_refctd_ptr() noexcept;

		inline smart_refctd_ptr& operator=(const smart_refctd_ptr<I_REFERENCE_COUNTED>& other) noexcept;
        template<class U> requires (!std::is_same_v<U,I_REFERENCE_COUNTED>)
		inline smart_refctd_ptr& operator=(const smart_refctd_ptr<U>& other) noexcept;

		inline smart_refctd_ptr& operator=(smart_refctd_ptr<I_REFERENCE_COUNTED>&& other) noexcept;
        //those std::enable_if_t's most likely not needed, but just to be sure (i put them to trigger SFINAE to be sure call to non-templated ctor is always generated in case of same type)
		template<class U> requires (!std::is_same_v<U,I_REFERENCE_COUNTED>)
		inline smart_refctd_ptr& operator=(smart_refctd_ptr<U>&& other) noexcept;

		// so that you don't mix refcounting methods
		void grab() = delete;
		void grab() const = delete;
		bool drop() = delete;
		bool drop() const = delete;

		// TODO: const correctness on the access operators (std::enable_if)
		inline I_REFERENCE_COUNTED* const& get() const { return ptr; }

		inline I_REFERENCE_COUNTED* operator->() const { return ptr; }

		inline I_REFERENCE_COUNTED& operator*() const { return *ptr; }

		inline I_REFERENCE_COUNTED& operator[](size_t idx) { return ptr[idx]; }
		inline const I_REFERENCE_COUNTED& operator[](size_t idx) const { return ptr[idx]; }

		// conversions
		inline explicit operator bool() const { return ptr; }
		inline bool operator!() const { return !ptr; }

// TODO: later, need to figure out some stuff about fwd declarations of concepts
//		template<class U> requires (!std::is_same_v<U,I_REFERENCE_COUNTED> && std::is_assignable_v<U,I_REFERENCE_COUNTED>)
//		inline operator smart_refctd_ptr<U>&&() && {return *reinterpret_cast<smart_refctd_ptr<U>>(this);}

		template<class U>
		inline bool operator==(const smart_refctd_ptr<U> &other) const { return ptr == other.ptr; }
		template<class U>
		inline bool operator!=(const smart_refctd_ptr<U> &other) const { return ptr != other.ptr; }

		template<class U>
		inline bool operator<(const smart_refctd_ptr<U> &other) const { return ptr < other.ptr; }
		template<class U>
		inline bool operator>(const smart_refctd_ptr<U>& other) const { return ptr > other.ptr; }
};
static_assert(sizeof(smart_refctd_ptr<IReferenceCounted>) == sizeof(IReferenceCounted*), "smart_refctd_ptr has a memory overhead!");


template< class T, class... Args >
smart_refctd_ptr<T> make_smart_refctd_ptr(Args&& ... args);


template< class U, class T >
smart_refctd_ptr<U> smart_refctd_ptr_static_cast(const smart_refctd_ptr<T>& smart_ptr);
template< class U, class T >
smart_refctd_ptr<U> smart_refctd_ptr_static_cast(smart_refctd_ptr<T>&& smart_ptr);

template< class U, class T >
smart_refctd_ptr<U> move_and_static_cast(smart_refctd_ptr<T>& smart_ptr);
template< class U, class T >
smart_refctd_ptr<U> move_and_static_cast(smart_refctd_ptr<T>&& smart_ptr) {return move_and_static_cast<U,T>(smart_ptr);}


template< class U, class T >
smart_refctd_ptr<U> smart_refctd_ptr_dynamic_cast(const smart_refctd_ptr<T>& smart_ptr);
template< class U, class T >
smart_refctd_ptr<U> smart_refctd_ptr_dynamic_cast(smart_refctd_ptr<T>&& smart_ptr);

template< class U, class T >
smart_refctd_ptr<U> move_and_dynamic_cast(smart_refctd_ptr<T>& smart_ptr);
template< class U, class T >
smart_refctd_ptr<U> move_and_dynamic_cast(smart_refctd_ptr<T>&& smart_ptr) {return move_and_dynamic_cast<U,T>(smart_ptr);}

} // end namespace nbl::core

/*
namespace std
{

    template <typename T>
    struct hash<nbl::core::smart_refctd_ptr<T>>
    {
		std::size_t operator()(const nbl::core::smart_refctd_ptr<T>& k) const;
    };

}
*/

#endif

