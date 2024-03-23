// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef _NBL_CORE_DEF_SMART_REFCTD_PTR_H_INCLUDED_
#define _NBL_CORE_DEF_SMART_REFCTD_PTR_H_INCLUDED_

#include "nbl/core/decl/smart_refctd_ptr.h"

namespace nbl::core
{

template<class I_REFERENCE_COUNTED>
template<class U>
inline void smart_refctd_ptr<I_REFERENCE_COUNTED>::copy(const smart_refctd_ptr<U>& other) noexcept
{
    if (other.ptr)
        other.ptr->grab();
    ptr = other.ptr;
}

template<class I_REFERENCE_COUNTED>
inline size_t smart_refctd_ptr<I_REFERENCE_COUNTED>::hash::operator() (const core::smart_refctd_ptr<I_REFERENCE_COUNTED>& ptr) const
{
	return std::hash<void*>{}(ptr.get());
}

template<class I_REFERENCE_COUNTED>
template<class U>
smart_refctd_ptr<I_REFERENCE_COUNTED>::smart_refctd_ptr(U* _pointer) noexcept : ptr(_pointer)
{
	if (_pointer)
		_pointer->grab();
}

template<class I_REFERENCE_COUNTED>
inline smart_refctd_ptr<I_REFERENCE_COUNTED>::~smart_refctd_ptr() noexcept
{
	if (ptr)
		ptr->drop();
}

template<class I_REFERENCE_COUNTED>
inline smart_refctd_ptr<I_REFERENCE_COUNTED>& smart_refctd_ptr<I_REFERENCE_COUNTED>::operator=(const smart_refctd_ptr<I_REFERENCE_COUNTED>& other) noexcept
{
	if (other.ptr)
		other.ptr->grab();
	if (ptr)
		ptr->drop();
	ptr = other.ptr;
	return *this;
}
template<class I_REFERENCE_COUNTED>
template<class U> requires (!std::is_same_v<U,I_REFERENCE_COUNTED>)
inline smart_refctd_ptr<I_REFERENCE_COUNTED>& smart_refctd_ptr<I_REFERENCE_COUNTED>::operator=(const smart_refctd_ptr<U>& other) noexcept
{
	if (other.ptr)
		other.ptr->grab();
	if (ptr)
		ptr->drop();
	ptr = other.ptr;
	return *this;
}

template<class I_REFERENCE_COUNTED>
inline smart_refctd_ptr<I_REFERENCE_COUNTED>& smart_refctd_ptr<I_REFERENCE_COUNTED>::operator=(smart_refctd_ptr<I_REFERENCE_COUNTED>&& other) noexcept
{
    if (ptr) // should only happen if constexpr (is convertible)
        ptr->drop();
    ptr = other.ptr;
    other.ptr = nullptr; // should only happen if constexpr (is convertible)
    return *this;
}
template<class I_REFERENCE_COUNTED>
template<class U> requires (!std::is_same_v<U,I_REFERENCE_COUNTED>)
inline smart_refctd_ptr<I_REFERENCE_COUNTED>& smart_refctd_ptr<I_REFERENCE_COUNTED>::operator=(smart_refctd_ptr<U>&& other) noexcept
{
	if (ptr) // should only happen if constexpr (is convertible)
		ptr->drop();
	ptr = other.ptr;
	other.ptr = nullptr; // should only happen if constexpr (is convertible)
	return *this;
}



template< class T, class... Args >
inline smart_refctd_ptr<T> make_smart_refctd_ptr(Args&& ... args)
{
    T* obj = new T(std::forward<Args>(args)...);
    smart_refctd_ptr<T> smart(obj, dont_grab);
    return smart;
}


template< class U, class T >
inline smart_refctd_ptr<U> smart_refctd_ptr_static_cast(const smart_refctd_ptr<T>& smart_ptr)
{
	return smart_refctd_ptr<U>(static_cast<U*>(smart_ptr.get()));
}
template< class U, class T >
inline smart_refctd_ptr<U> smart_refctd_ptr_static_cast(smart_refctd_ptr<T>&& smart_ptr)
{
	T* ptr = nullptr;
	std::swap(ptr, smart_ptr.ptr);
	return smart_refctd_ptr<U>(static_cast<U*>(ptr), dont_grab);
}

template< class U, class T >
inline smart_refctd_ptr<U> move_and_static_cast(smart_refctd_ptr<T>& smart_ptr)
{
	return smart_refctd_ptr_static_cast<U,T>(std::move(smart_ptr));
}


template< class U, class T >
inline smart_refctd_ptr<U> smart_refctd_ptr_dynamic_cast(const smart_refctd_ptr<T>& smart_ptr)
{
	return smart_refctd_ptr<U>(dynamic_cast<U*>(smart_ptr.get()));
}
template< class U, class T >
inline smart_refctd_ptr<U> smart_refctd_ptr_dynamic_cast(smart_refctd_ptr<T>&& smart_ptr)
{
	T* ptr = nullptr;
	std::swap(ptr, smart_ptr.ptr);
	return smart_refctd_ptr<U>(dynamic_cast<U*>(ptr), dont_grab);
}

template< class U, class T >
inline smart_refctd_ptr<U> move_and_dynamic_cast(smart_refctd_ptr<T>& smart_ptr)
{
	return smart_refctd_ptr_dynamic_cast<U,T>(std::move(smart_ptr));
}

} // end namespace nbl::core

namespace std
{

    template <typename T>
    struct hash<nbl::core::smart_refctd_ptr<T>>
    {
        std::size_t operator()(const nbl::core::smart_refctd_ptr<T>& k) const
        {
            return reinterpret_cast<std::size_t>(k.get());
        }
    };

}

#endif

