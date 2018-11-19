// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h


#ifndef __IRR_ALIGNMENT_H_INCLUDED__
#define __IRR_ALIGNMENT_H_INCLUDED__


#include "irr/core/memory/new_delete.h"
#include "irr/core/memory/memory.h"


namespace irr
{
namespace core
{

//Maybe: Create a irr::AllocatedByDynamicAllocation class with a static function new[] like operator that takes an DynamicAllocator* parameter
// but would need to keep a pointer to allocator and allocated object size (+16 bytes!)


//! Inherit from this class if you want to make sure all the derivations are aligned.
/** Any class derived from this, even indirectly will be declared as aligned to
`object_alignment`, unfortunately this is only enforced for objects on the stack.
To make sure your object is aligned on heaps as well you need to inherit from
`AlignedAllocOverrideBase` or one of its aliases instead. **/
template<size_t object_alignment=_IRR_SIMD_ALIGNMENT>
class IRR_FORCE_EBO alignas(object_alignment) AlignedBase
{
    static_assert(is_alignment(object_alignment),"Alignments must be PoT and positive!");
    static_assert(object_alignment<=128,"Pending migration to GCC 7+ highest alignment on c++ class is 128 bytes");
};

//! Here I VIOLATE THE C++ SPEC, ALL NEW OPERATORS ARE NOTHROW BY DEFAULT!
namespace impl
{
    //! Variadic template class for metaprogrammatically resolving max alignment resulting from multiple inheritance of AlignedBase. PLEASE DO NOT USE ANYWHERE ELSE.
    template <class... Ts>
    class IRR_FORCE_EBO ResolveAlignment
    {
    };

    //! Specialization of ResolveAlignment for recursively resolving the alignment of many types.
    template <class T, class... Ts>
    class IRR_FORCE_EBO ResolveAlignment<T, Ts...> :  public T, public Ts...
    {
        private:
            struct DummyForConditional {typedef uint8_t most_aligned_type;};
            static_assert(std::alignment_of<typename DummyForConditional::most_aligned_type>::value==1ull,"Why is uint8_t not aligned to 1 in your compiler?");

            //! In this section we get the type with maximum alignment for the N-1 recursion.
            constexpr static bool isNextRecursionEmpty = std::is_same<ResolveAlignment<Ts...>,ResolveAlignment<> >::value;
            typedef typename std::conditional<isNextRecursionEmpty,DummyForConditional,ResolveAlignment<Ts...> >::type::most_aligned_type otherType;

            //! Bool telling is whether our current type, `T`, is larger than the max type of the recursion.
            constexpr static bool isTLargestType = std::alignment_of<T>::value>std::alignment_of<otherType>::value;
        public:
            //! The maximally aligned type for this recursion of N template parameters
            typedef typename std::conditional<isTLargestType,T,otherType>::type most_aligned_type;
        private:
            //! Fallback to use in-case most_aligned_type does not declare its own new and delete (fallback is a specialization of this class for single element).
            using DefaultAlignedAllocationOverriden = ResolveAlignment<AlignedBase<std::alignment_of<most_aligned_type>::value> >;

            //! Some meta-functions to allow us for static checking of metaprogrammatically derived most_aligned_type
            template<class U> using operator_new_t                  = decltype(U::operator new(0ull));
            template<class U> using operator_new_array_t            = decltype(U::operator new[](0ull));
            template<class U> using operator_placement_new_t        = decltype(U::operator new(0ull,nullptr));
            template<class U> using operator_placement_new_array_t  = decltype(U::operator new[](0ull,nullptr));
            template<class U> using operator_delete_t               = decltype(U::operator delete(nullptr));
            template<class U> using operator_delete_array_t         = decltype(U::operator delete[](nullptr));
            template<class U> using operator_delete_w_size_t        = decltype(U::operator delete(nullptr,0ull));
            template<class U> using operator_delete_array_w_size_t  = decltype(U::operator delete[](nullptr,0ull));

            template<class,class=void> struct has_new_operator                  : std::false_type {};
            template<class,class=void> struct has_new_array_operator            : std::false_type {};
            template<class,class=void> struct has_placement_new_operator        : std::false_type {};
            template<class,class=void> struct has_placement_new_array_operator  : std::false_type {};
            template<class,class=void> struct has_delete_operator               : std::false_type {};
            template<class,class=void> struct has_delete_array_operator         : std::false_type {};
            template<class,class=void> struct has_delete_operator_w_size        : std::false_type {};
            template<class,class=void> struct has_delete_array_operator_w_size  : std::false_type {};
            template<class U> struct has_new_operator<U,void_t<operator_new_t<U> > >                                    : std::is_same<operator_new_t<U>,void*> {};
            template<class U> struct has_new_array_operator<U,void_t<operator_new_array_t<U> > >                        : std::is_same<operator_new_array_t<U>,void*> {};
            template<class U> struct has_placement_new_operator<U,void_t<operator_placement_new_t<U> > >                : std::is_same<operator_placement_new_t<U>,void*> {};
            template<class U> struct has_placement_new_array_operator<U,void_t<operator_placement_new_array_t<U> > >    : std::is_same<operator_placement_new_array_t<U>,void*> {};
            template<class U> struct has_delete_operator<U,void_t<operator_delete_t<U> > >                              : std::is_same<operator_delete_t<U>,void> {};
            template<class U> struct has_delete_array_operator<U,void_t<operator_delete_array_t<U> > >                  : std::is_same<operator_delete_array_t<U>,void> {};
            template<class U> struct has_delete_operator_w_size<U,void_t<operator_delete_w_size_t<U> > >                : std::is_same<operator_delete_w_size_t<U>,void> {};
            template<class U> struct has_delete_array_operator_w_size<U,void_t<operator_delete_array_w_size_t<U> > >    : std::is_same<operator_delete_array_w_size_t<U>,void> {};
        public:
            /** Now we could override the new and delete operators always with the same thing, and allocate aligned to `std::alignment_of<most_aligned_type>::value`,
            however we want to call the most aligned class' new and delete operators (if such exist) so its overrides actually matter.
            **/
            inline static void* operator new(size_t size) noexcept
            {
                return std::conditional<has_new_operator<most_aligned_type>::value,most_aligned_type,DefaultAlignedAllocationOverriden>::type::operator new(size);
            }
            static inline void* operator new[](size_t size) noexcept
            {
                return std::conditional<has_new_array_operator<most_aligned_type>::value,most_aligned_type,DefaultAlignedAllocationOverriden>::type::operator new[](size);
            }
            inline static void* operator new(size_t size, void* where) noexcept
            {
                return std::conditional<has_placement_new_operator<most_aligned_type>::value,most_aligned_type,DefaultAlignedAllocationOverriden>::type::operator new(size,where);
            }
            static inline void* operator new[](size_t size, void* where) noexcept
            {
                return std::conditional<has_placement_new_array_operator<most_aligned_type>::value,most_aligned_type,DefaultAlignedAllocationOverriden>::type::operator new[](size,where);
            }

            static inline void operator delete(void* ptr) noexcept
            {
                std::conditional<has_delete_operator<most_aligned_type>::value,most_aligned_type,DefaultAlignedAllocationOverriden>::type::operator delete(ptr);
            }
            static inline void operator delete[](void* ptr) noexcept
            {
                std::conditional<has_delete_array_operator<most_aligned_type>::value,most_aligned_type,DefaultAlignedAllocationOverriden>::type::operator delete[](ptr);
            }
            static inline void operator delete(void* ptr, size_t size) noexcept
            {
                std::conditional<has_delete_operator_w_size<most_aligned_type>::value,most_aligned_type,DefaultAlignedAllocationOverriden>::type::operator delete(ptr,size);
            }
            static inline void operator delete[](void* ptr, size_t size) noexcept
            {
                std::conditional<has_delete_array_operator_w_size<most_aligned_type>::value,most_aligned_type,DefaultAlignedAllocationOverriden>::type::operator delete[](ptr,size);
            }

    };

    //! Specialization for the base case of a single parameter whose type is AlignedBase (needed to specify default new and delete)
    /** Note regarding C++17, we don't overload the alignment on the versions with std::align_val_t.
    Why? Because we want to respect and not f-up the explicitly requested alignment. **/
    template <size_t object_alignment>
    class IRR_FORCE_EBO ResolveAlignment<AlignedBase<object_alignment> > :  public AlignedBase<object_alignment>
    {
        public:
            //! The maximally aligned type for this recursion of N template parameters
            typedef AlignedBase<object_alignment> most_aligned_type;
        private:
            //
        public:
            inline static void* operator new(size_t size) noexcept
            {
                //std::cout << "Alloc aligned to " << object_alignment << std::endl;
                return _IRR_ALIGNED_MALLOC(size,object_alignment);
            }
            static inline void* operator new[](size_t size) noexcept
            {
                //std::cout << "Alloc aligned to " << object_alignment << std::endl;
                return _IRR_ALIGNED_MALLOC(size,object_alignment);
            }
            inline static void* operator new(size_t size, void* where) noexcept
            {
                return where;
            }
            static inline void* operator new[](size_t size, void* where) noexcept
            {
                return where;
            }

            static inline void operator delete(void* ptr) noexcept
            {
                //std::cout << "Delete aligned to " << object_alignment << std::endl;
                _IRR_ALIGNED_FREE(ptr);
            }
            static inline void  operator delete[](void* ptr) noexcept
            {
                //std::cout << "Delete aligned to " << object_alignment << std::endl;
                _IRR_ALIGNED_FREE(ptr);
            }
            static inline void operator delete(void* ptr, size_t size) noexcept {operator delete(ptr);} //roll back to own operator with no size
            static inline void operator delete[](void* ptr, size_t size) noexcept {operator delete[](ptr);} //roll back to own operator with no size

            static inline bool isPtrAlignedForThisType(const void* ptr) noexcept
            {
                return is_aligned_to(ptr,object_alignment);
            }
    };
}

//! This is a base class for overriding default memory management.
template <size_t _in_alignment>
using AllocationOverrideBase = impl::ResolveAlignment<AlignedBase<_in_alignment> >;

//#define IRR_TEST_NO_NEW_DELETE_OVERRIDE

#ifndef IRR_TEST_NO_NEW_DELETE_OVERRIDE
using AllocationOverrideDefault = AllocationOverrideBase<_IRR_SIMD_ALIGNMENT>;


//! Put in class if the compiler is complaining about ambiguous references to new and delete operators. Needs to be placed in the public section of methods
#define _IRR_RESOLVE_NEW_DELETE_AMBIGUITY(...) \
            static inline void* operator new(size_t size)                noexcept {return (irr::core::impl::ResolveAlignment<__VA_ARGS__>::operator new(size));} \
            static inline void* operator new[](size_t size)              noexcept {return irr::core::impl::ResolveAlignment<__VA_ARGS__>::operator new[](size);} \
            static inline void* operator new(size_t size, void* where)   noexcept {return (irr::core::impl::ResolveAlignment<__VA_ARGS__>::operator new(size,where));} \
            static inline void* operator new[](size_t size, void* where) noexcept {return irr::core::impl::ResolveAlignment<__VA_ARGS__>::operator new[](size,where);} \
            static inline void operator delete(void* ptr)                noexcept {irr::core::impl::ResolveAlignment<__VA_ARGS__>::operator delete(ptr);} \
            static inline void operator delete[](void* ptr)              noexcept {irr::core::impl::ResolveAlignment<__VA_ARGS__>::operator delete[](ptr);} \
            static inline void operator delete(void* ptr, size_t size)   noexcept {irr::core::impl::ResolveAlignment<__VA_ARGS__>::operator delete(ptr,size);} \
            static inline void operator delete[](void* ptr, size_t size) noexcept {irr::core::impl::ResolveAlignment<__VA_ARGS__>::operator delete[](ptr,size);}
#else
struct IRR_FORCE_EBO AllocationOverrideDefault {};

#define _IRR_RESOLVE_NEW_DELETE_AMBIGUITY(...)
#endif // IRR_TEST_NO_NEW_DELETE_OVERRIDE
}
}

#endif // __IRR_ALIGNMENT_H_INCLUDED__


