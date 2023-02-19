// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_NEW_DELETE_H_INCLUDED__
#define __NBL_CORE_NEW_DELETE_H_INCLUDED__

#include <memory>

#include "nbl/macros.h"
#include "nbl/core/alloc/aligned_allocator.h"



//! Allocator MUST provide a function with signature `pointer allocate(size_type n, size_type alignment, const_void_pointer hint=nullptr) noexcept`
#define _NBL_DEFAULT_ALLOCATOR_METATYPE                                 nbl::core::aligned_allocator

namespace nbl
{
namespace core
{
namespace impl
{

template<typename T, class Alloc=_NBL_DEFAULT_ALLOCATOR_METATYPE<T> >
struct NBL_API AlignedWithAllocator
{
    template<typename... Args>
    static inline T*    new_(size_t align, Alloc& alloc, Args&&... args)
    {
        T* retval = alloc.allocate(1u,align);
        if (!retval)
            return nullptr;
        std::allocator_traits<Alloc>::construct(alloc,retval,std::forward<Args>(args)...);
        return retval;
    }
    template<typename... Args>
    static inline T*    new_(size_t align, Alloc&& alloc, Args&&... args)
    {
        return new_(align,static_cast<Alloc&>(alloc),std::forward<Args>(args)...);
    }
    static inline void  delete_(T* p, Alloc& alloc)
    {
        if (!p)
            return;
        std::allocator_traits<Alloc>::destroy(alloc,p);
        alloc.deallocate(p,1u);
    }
    static inline void  delete_(T* p, Alloc&& alloc=Alloc())
    {
        delete_(p,static_cast<Alloc&>(alloc));
    }


    static inline T*    new_array(size_t n, size_t align, Alloc& alloc)
    {
        T* retval = alloc.allocate(n,align);
        if (!retval)
            return nullptr;
        for (size_t dit=0; dit<n; dit++)
            std::allocator_traits<Alloc>::construct(alloc,retval+dit);
        return retval;
    }
    static inline T*    new_array(size_t n, size_t align, Alloc&& alloc=Alloc())
    {
        return new_array(n,align,static_cast<Alloc&>(alloc));
    }
    static inline void  delete_array(T* p, size_t n, Alloc& alloc)
    {
        if (!p)
            return;
        for (size_t dit=0; dit<n; dit++)
            std::allocator_traits<Alloc>::destroy(alloc,p+dit);
        alloc.deallocate(p,n);
    }
    static inline void  delete_array(T* p, size_t n, Alloc&& alloc=Alloc())
    {
        delete_array(p,n,static_cast<Alloc&>(alloc));
    }

    struct NBL_API VA_ARGS_comma_workaround
    {
        VA_ARGS_comma_workaround(size_t align, Alloc _alloc = Alloc()) : m_align(align), m_alloc(_alloc) {}

        template<typename... Args>
        inline T*       new_(Args&&... args)
        {
            return AlignedWithAllocator::new_(m_align,m_alloc,std::forward<Args>(args)...);
        }


        size_t m_align;
        Alloc m_alloc;
    };
};

}
}
}

//use these by default instead of new and delete, single object (non-array) new takes constructor parameters as va_args
#define _NBL_NEW(_obj_type, ... )                               nbl::core::impl::AlignedWithAllocator<_obj_type,_NBL_DEFAULT_ALLOCATOR_METATYPE<_obj_type> >::VA_ARGS_comma_workaround(_NBL_DEFAULT_ALIGNMENT(_obj_type)).new_(__VA_ARGS__)
#define _NBL_DELETE(_obj)                                       nbl::core::impl::AlignedWithAllocator<typename std::remove_reference<typename std::remove_pointer<decltype(_obj)>::type>::type,_NBL_DEFAULT_ALLOCATOR_METATYPE<typename std::remove_reference<typename std::remove_pointer<decltype(_obj)>::type>::type> >::delete_(_obj)

#define _NBL_NEW_ARRAY(_obj_type,count)                         nbl::core::impl::AlignedWithAllocator<_obj_type,_NBL_DEFAULT_ALLOCATOR_METATYPE<_obj_type> >::new_array(count,_NBL_DEFAULT_ALIGNMENT(_obj_type))
#define _NBL_DELETE_ARRAY(_obj,count)                           nbl::core::impl::AlignedWithAllocator<typename std::remove_reference_t<typename std::remove_pointer_t<decltype(_obj)>>,_NBL_DEFAULT_ALLOCATOR_METATYPE<typename std::remove_reference_t<typename std::remove_pointer_t<decltype(_obj)>>> >::delete_array(_obj,count)

//! Extra Utility Macros for when you don't want to always have to deduce the alignment but want to use a specific allocator
//#define _NBL_ASSERT_ALLOCATOR_VALUE_TYPE(_obj_type,_allocator_type)     static_assert(std::is_same<_obj_type,_allocator_type::value_type>::value,"Wrong allocator value_type!")
#define _NBL_ASSERT_ALLOCATOR_VALUE_TYPE(_obj_type,_allocator_type)     static_assert(std::is_same<_obj_type,std::allocator_traits<_allocator_type >::value_type>::value,"Wrong allocator value_type!")

#define _NBL_NEW_W_ALLOCATOR(_obj_type,_allocator, ... )        nbl::core::impl::AlignedWithAllocator<_obj_type,_NBL_DEFAULT_ALLOCATOR_METATYPE<_obj_type> >::VA_ARGS_comma_workaround(_NBL_DEFAULT_ALIGNMENT(_obj_type),_allocator).new_(__VA_ARGS__); \
                                                                    _NBL_ASSERT_ALLOCATOR_VALUE_TYPE(_obj_type,decltype(_allocator))
#define _NBL_DELETE_W_ALLOCATOR(_obj,_allocator)                nbl::core::impl::AlignedWithAllocator<typename std::remove_reference<typename std::remove_pointer<decltype(_obj)>::type>::type,_NBL_DEFAULT_ALLOCATOR_METATYPE<typename std::remove_reference<typename std::remove_pointer<decltype(_obj)>::type>::type> >::delete_(_obj,_allocator); \
                                                                    _NBL_ASSERT_ALLOCATOR_VALUE_TYPE(std::remove_reference<typename std::remove_pointer<decltype(_obj)>::type>::type,decltype(_allocator))

#define _NBL_NEW_ARRAY_W_ALLOCATOR(_obj_type,count,_allocator)  nbl::core::impl::AlignedWithAllocator<_obj_type,_NBL_DEFAULT_ALLOCATOR_METATYPE<_obj_type> >::new_array(count,_NBL_DEFAULT_ALIGNMENT(_obj_type),_allocator); \
                                                                    _NBL_ASSERT_ALLOCATOR_VALUE_TYPE(_obj_type,decltype(_allocator))
#define _NBL_DELETE_ARRAY_W_ALLOCATOR(_obj,count,_allocator)    nbl::core::impl::AlignedWithAllocator<typename std::remove_reference<typename std::remove_pointer<decltype(_obj)>::type>::type,_NBL_DEFAULT_ALLOCATOR_METATYPE<typename std::remove_reference<typename std::remove_pointer<decltype(_obj)>::type>::type> >::delete_array(_obj,count,_allocator); \
                                                                    _NBL_ASSERT_ALLOCATOR_VALUE_TYPE(std::remove_reference<typename std::remove_pointer<decltype(_obj)>::type>::type,decltype(_allocator))


namespace nbl
{
namespace core
{
/**
//Maybe: Create a nbl::AllocatedByDynamicAllocation class with a static function new[] like operator that takes an DynamicAllocator* parameter

template<class CRTP, class Alloc=aligned_allocator<CRTP> >
class NBL_API NBL_FORCE_EBO AllocatedWithStatelessAllocator
{
    public:
};
*/

//! Special Class For providing deletion for things like C++11 smart-pointers
struct NBL_API alligned_delete
{
    template<class T>
    void operator()(T* ptr) const noexcept(noexcept(ptr->~T()))
    {
        if (ptr)
        {
            ptr->~T();
            _NBL_ALIGNED_FREE(ptr);
        }
    }
};

}
}

#endif

