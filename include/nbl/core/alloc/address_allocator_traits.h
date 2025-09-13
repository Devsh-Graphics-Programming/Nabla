// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_ADDRESS_ALLOCATOR_TYPE_TRAITS_H_INCLUDED__
#define __NBL_CORE_ADDRESS_ALLOCATOR_TYPE_TRAITS_H_INCLUDED__

#include <algorithm>
#include "stdint.h"
#include "nbl/macros.h"
#include "nbl/type_traits.h"

namespace nbl::core
{

    template<typename AddressType>
    struct address_type_traits;

    template<>
    struct address_type_traits<uint32_t>
    {
        address_type_traits() = delete;
        _NBL_STATIC_INLINE_CONSTEXPR uint32_t   invalid_address = 0xdeadbeefu;
    };

    template<>
    struct address_type_traits<uint64_t>
    {
        address_type_traits() = delete;
        _NBL_STATIC_INLINE_CONSTEXPR uint64_t   invalid_address = 0xdeadbeefBADC0FFEull;
    };


    namespace impl
    {
        template<class AddressAlloc, bool hasAttribute>
        struct address_allocator_traits_base;
        //provide default traits
        template<class AddressAlloc>
        struct address_allocator_traits_base<AddressAlloc,false>
        {
            typedef typename AddressAlloc::size_type size_type;

            static inline void         multi_alloc_addr(AddressAlloc& alloc, uint32_t count, size_type* outAddresses, const size_type* bytes,
                                                        const size_type* alignment, const size_type* hint=nullptr) noexcept
            {
                for (uint32_t i=0; i<count; i++)
                {
                    if (outAddresses[i]!=AddressAlloc::invalid_address)
                        continue;

                    outAddresses[i] = alloc.alloc_addr(bytes[i],alignment[i],hint ? hint[i]:0ull);
                }
            }

            static inline void         multi_alloc_addr(AddressAlloc& alloc, uint32_t count, size_type* outAddresses, const size_type* bytes,
                                                        const size_type alignment, const size_type* hint=nullptr) noexcept
            {
                for (uint32_t i=0; i<count; i++)
                {
                    if (outAddresses[i]!=AddressAlloc::invalid_address)
                        continue;

                    outAddresses[i] = alloc.alloc_addr(bytes[i],alignment,hint ? hint[i]:0ull);
                }
            }

            static inline void         multi_free_addr(AddressAlloc& alloc, uint32_t count, const size_type* addr, const size_type* bytes) noexcept
            {
                for (uint32_t i=0; i<count; i++)
                {
                    if (addr[i]==AddressAlloc::invalid_address)
                        continue;

                    alloc.free_addr(addr[i],bytes[i]);
                }
            }


            static inline size_type    get_real_addr(const AddressAlloc& alloc, size_type allocated_addr)
            {
                return allocated_addr;
            }
        };
        //forward existing traits
        template<class AddressAlloc>
        struct address_allocator_traits_base<AddressAlloc,true>
        {
            typedef typename AddressAlloc::size_type size_type;

            template<typename... Args>
            static inline void         multi_alloc_addr(AddressAlloc& alloc, Args&&... args) noexcept
            {
                alloc.multi_alloc_addr(std::forward<Args>(args)...);
            }

            template<typename... Args>
            static inline void         multi_free_addr(AddressAlloc& alloc, Args&&... args) noexcept
            {
                alloc.multi_free_addr(std::forward<Args>(args)...);
            }


            static inline size_type    get_real_addr(const AddressAlloc& alloc, size_type allocated_addr)
            {
                return alloc.get_real_addr(allocated_addr);
            }
        };
    }


    //! TODO: https://en.cppreference.com/w/cpp/experimental/is_detected
    template<class AddressAlloc>
    class address_allocator_traits : protected AddressAlloc //maybe private?
    {
        public:
            typedef AddressAlloc                        allocator_type;
            typedef typename AddressAlloc::size_type    size_type;
        private:
            class ConstGetter : public AddressAlloc
            {
                    ConstGetter() = delete;
                    virtual ~ConstGetter() = default;
                public:
                    inline const void*  getReservedSpacePtr() const noexcept{return AddressAlloc::getReservedSpacePtr();}
                    inline size_type    max_size() const noexcept           {return AddressAlloc::max_size();}
                    inline size_type    min_size() const noexcept           {return AddressAlloc::min_size();}
                    inline size_type    max_alignment() const noexcept      {return AddressAlloc::max_alignment();}
                    inline size_type    get_align_offset() const noexcept   {return AddressAlloc::get_align_offset();}
                    inline size_type    get_combined_offset() const noexcept{return AddressAlloc::get_combined_offset();}
                    inline size_type    get_free_size() const noexcept      {return AddressAlloc::get_free_size();}
                    inline size_type    get_allocated_size() const noexcept {return AddressAlloc::get_allocated_size();}
                    inline size_type    get_total_size() const noexcept     {return AddressAlloc::get_total_size();}
            };
            virtual ~address_allocator_traits() = default;

            template<class U> using cstexpr_supportsArbitraryOrderFrees = decltype(std::declval<U&>().supportsArbitraryOrderFrees);
            template<class U> using cstexpr_maxMultiOps                 = decltype(std::declval<U&>().maxMultiOps);

            template<class U> using func_multi_alloc_addr               = decltype(std::declval<U&>().multi_alloc_addr(0u,nullptr,nullptr,nullptr,nullptr));
            template<class U> using func_multi_free_addr                = decltype(std::declval<U&>().multi_free_addr(0u,nullptr,nullptr));

            template<class U> using func_get_real_addr                  = decltype(std::declval<U&>().get_real_addr(0u));
        /// C++17 protected:
        public:
            template<class,class=void> struct resolve_supportsArbitraryOrderFrees  : std::true_type {};
            template<class,class=void> struct resolve_maxMultiOps                           : std::integral_constant<uint32_t,256u> {};

            template<class,class=void> struct has_func_multi_alloc_addr                : std::false_type {};
            template<class,class=void> struct has_func_multi_free_addr                 : std::false_type {};

            template<class,class=void> struct has_func_get_real_addr                     : std::false_type {};


            template<class U> struct resolve_supportsArbitraryOrderFrees<U,std::void_t<cstexpr_supportsArbitraryOrderFrees<U> > >
                                                                            :  std::conditional<std::true_type/*std::is_same<cstexpr_supportsArbitraryOrderFrees<U>,bool>*/::value,nbl::bool_constant<U::supportsArbitraryOrderFrees>,resolve_supportsArbitraryOrderFrees<void,void> >::type {};
            template<class U> struct resolve_maxMultiOps<U,std::void_t<cstexpr_maxMultiOps<U> > >
                                                                            : std::conditional<std::true_type/*std::is_integral<cstexpr_maxMultiOps<U> >*/::value,std::integral_constant<uint32_t,U::maxMultiOps>, resolve_maxMultiOps<void, void> >::type {};

            template<class U> struct has_func_multi_alloc_addr<U,std::void_t<func_multi_alloc_addr<U> > >
                                                                            : std::is_same<func_multi_alloc_addr<U>,void> {};
            template<class U> struct has_func_multi_free_addr<U,std::void_t<func_multi_free_addr<U> > >
                                                                            : std::is_same<func_multi_free_addr<U>,void> {};

            template<class U> struct has_func_get_real_addr<U,std::void_t<func_get_real_addr<U> > >  : std::is_same<func_get_real_addr<U>,size_type> {};

            _NBL_STATIC_INLINE_CONSTEXPR bool         supportsArbitraryOrderFrees = resolve_supportsArbitraryOrderFrees<AddressAlloc>::value;
            _NBL_STATIC_INLINE_CONSTEXPR uint32_t     maxMultiOps                 = resolve_maxMultiOps<AddressAlloc>::value;

            // TODO: make the printer customizable without making a `core`->`system` circular dep
            static inline void          printDebugInfo()
            {
                printf("has_func_multi_alloc_addr : %s\n",                  has_func_multi_alloc_addr<AddressAlloc>::value ? "true":"false");
                printf("has_func_multi_free_addr : %s\n",                   has_func_multi_free_addr<AddressAlloc>::value ? "true":"false");
                printf("has_func_get_real_addr : %s\n",                        has_func_get_real_addr<AddressAlloc>::value ? "true":"false");

                printf("supportsArbitraryOrderFrees == %d\n", supportsArbitraryOrderFrees);
                printf("maxMultiOps == %d\n",                           maxMultiOps);
            }


            using AddressAlloc::AddressAlloc;


            static inline size_type        get_real_addr(const AddressAlloc& alloc, size_type allocated_addr) noexcept
            {
                return impl::address_allocator_traits_base<AddressAlloc,has_func_get_real_addr<AddressAlloc>::value>::get_real_addr(allocated_addr);
            }

            //!
            /** Warning outAddresses needs to be primed with `invalid_address` values,
            otherwise no allocation happens for elements not equal to `invalid_address`. */
            static inline void              multi_alloc_addr(AddressAlloc& alloc, uint32_t count, size_type* outAddresses,
                                                             const size_type* bytes, const size_type* alignment, const size_type* hint=nullptr) noexcept
            {
                for (uint32_t i=0; i<count; i+=maxMultiOps)
                    impl::address_allocator_traits_base<AddressAlloc,has_func_multi_alloc_addr<AddressAlloc>::value>::multi_alloc_addr(
                                                                alloc,std::min(count-i,maxMultiOps),outAddresses+i,bytes+i,alignment+i,hint ? (hint+i):nullptr);
            }

            static inline void              multi_alloc_addr(AddressAlloc& alloc, uint32_t count, size_type* outAddresses,
                                                             const size_type* bytes, const size_type alignment, const size_type* hint=nullptr) noexcept
            {
                for (uint32_t i=0; i<count; i+=maxMultiOps)
                    impl::address_allocator_traits_base<AddressAlloc,has_func_multi_alloc_addr<AddressAlloc>::value>::multi_alloc_addr(
                                                                alloc,std::min(count-i,maxMultiOps),outAddresses+i,bytes+i,alignment,hint ? (hint+i):nullptr);
            }

            static inline void             multi_free_addr(AddressAlloc& alloc, uint32_t count, const size_type* addr, const size_type* bytes) noexcept
            {
                for (uint32_t i=0; i<count; i+=maxMultiOps)
                    impl::address_allocator_traits_base<AddressAlloc,has_func_multi_free_addr<AddressAlloc>::value>::multi_free_addr(
                                                                alloc,std::min(count-i,maxMultiOps),addr+i,bytes+i);
            }

            static inline const void*       getReservedSpacePtr(const AddressAlloc& alloc) noexcept
            {
                return static_cast<const ConstGetter&>(alloc).getReservedSpacePtr();
            }


            static inline size_type        max_size(const AddressAlloc& alloc) noexcept
            {
                return static_cast<const ConstGetter&>(alloc).max_size();
            }
            static inline size_type        min_size(const AddressAlloc& alloc) noexcept
            {
                return static_cast<const ConstGetter&>(alloc).min_size();
            }
            static inline size_type        max_alignment(const AddressAlloc& alloc) noexcept
            {
                return static_cast<const ConstGetter&>(alloc).max_alignment();
            }
            static inline size_type        get_align_offset(const AddressAlloc& alloc) noexcept
            {
                return static_cast<const ConstGetter&>(alloc).get_align_offset();
            }
            static inline size_type        get_combined_offset(const AddressAlloc& alloc) noexcept
            {
                return static_cast<const ConstGetter&>(alloc).get_combined_offset();
            }


            static inline size_type        get_free_size(const AddressAlloc& alloc) noexcept
            {
                return static_cast<const ConstGetter&>(alloc).get_free_size();
            }
            static inline size_type        get_allocated_size(const AddressAlloc& alloc) noexcept
            {
                return static_cast<const ConstGetter&>(alloc).get_allocated_size();
            }
            static inline size_type        get_total_size(const AddressAlloc& alloc) noexcept
            {
                return static_cast<const ConstGetter&>(alloc).get_total_size();
            }

            // underlying allocator statics
            template<typename... Args>
            static inline size_type reserved_size(const Args&... args) noexcept
            {
                return AddressAlloc::reserved_size(args...);
            }
    };

}

#endif