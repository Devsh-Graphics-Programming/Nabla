// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_ADDRESS_ALLOCATOR_TYPE_TRAITS_H_INCLUDED__
#define __IRR_ADDRESS_ALLOCATOR_TYPE_TRAITS_H_INCLUDED__

namespace irr
{
namespace core
{

    template<typename AddressType>
    struct address_type_traits;

    template<>
    struct address_type_traits<uint32_t>
    {
        address_type_traits() = delete;
        static constexpr uint32_t   invalid_address = 0xdeadbeefu;
    };

    template<>
    struct address_type_traits<uint64_t>
    {
        address_type_traits() = delete;
        static constexpr uint64_t   invalid_address = 0xdeadbeefBADC0FFEull;
    };


    namespace impl
    {
        template<class AddressAlloc, bool hasAttribute>
        struct address_allocator_traits_base;
        //provide default traits
        template<class AddressAlloc>
        struct address_allocator_traits_base<AddressAlloc,false>
        {
            /// C++17 inline static constexpr bool supportsNullBuffer= false;
            static constexpr uint32_t maxMultiOps   = 256u;

            typedef typename AddressAlloc::size_type size_type;

            static inline void         multi_alloc_addr(AddressAlloc& alloc, uint32_t count, size_type* outAddresses, const size_type* bytes,
                                                        const size_type* alignment, const size_type* hint=nullptr) noexcept
            {
                for (uint32_t i=0; i<count; i++)
                    outAddresses[i] = alloc.alloc_addr(bytes[i],alignment[i],hint ? hint[i]:0ull);
            }

            static inline void         multi_free_addr(AddressAlloc& alloc, uint32_t count, const size_type* addr, const size_type* bytes) noexcept
            {
                for (uint32_t i=0; i<count; i++)
                    alloc.free_addr(addr[i],bytes[i]);
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
            /// C++17 inline static constexpr bool supportsNullBuffer= AddressAlloc::supportsNullBuffer;
            static constexpr uint32_t maxMultiOps   = AddressAlloc::maxMultiOps;

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

    template<class AddressAlloc>
    class address_allocator_traits : protected AddressAlloc //maybe private?
    {
        private:
            template<class U> using cstexpr_maxMultiOps             = decltype(U::maxMultiOps);
            template<class U> using cstexpr_supportsNullBuffer      = decltype(U::supportsNullBuffer);

            template<class U> using func_multi_alloc_addr           = decltype(U::multi_alloc_addr(0u,nullptr,nullptr,nullptr,nullptr));
            template<class U> using func_multi_free_addr            = decltype(U::multi_free_addr(0u,nullptr,nullptr));

            template<class U> using func_get_real_addr              = decltype(U::get_real_addr(0u));
        protected:
            template<class,class=void> struct has_maxMultiOps                                       : std::false_type {};
            template<class,class=void> struct has_supportsNullBuffer                                : std::false_type {};

            template<class,class=void> struct has_func_multi_alloc_addr                             : std::false_type {};
            template<class,class=void> struct has_func_multi_free_addr                              : std::false_type {};

            template<class,class=void> struct has_func_get_real_addr                                : std::false_type {};


            template<class U> struct has_maxMultiOps<U,void_t<cstexpr_maxMultiOps<U> > >            : std::is_same<cstexpr_maxMultiOps<U>,void> {};
            template<class U> struct has_supportsNullBuffer<U,void_t<cstexpr_supportsNullBuffer<U> > > :
                                                                                                std::is_same<cstexpr_supportsNullBuffer<U>,void> {};

            template<class U> struct has_func_multi_alloc_addr<U,void_t<func_multi_alloc_addr<U> > >: std::is_same<func_multi_alloc_addr<U>,void> {};
            template<class U> struct has_func_multi_free_addr<U,void_t<func_multi_free_addr<U> > >  : std::is_same<func_multi_free_addr<U>,void> {};

            template<class U> struct has_func_get_real_addr<U,void_t<func_get_real_addr<U> > >      : std::is_same<func_get_real_addr<U>,void> {};
        public:
            /// C++17 inline static constexpr bool supportsNullBuffer= impl::address_allocator_traits_base<AddressAlloc,has_supportsNullBuffer<AddressAlloc>::value>::supportsNullBuffer;
            /// C++17 inline static constexpr uint32_t maxMultiOps = impl::address_allocator_traits_base<AddressAlloc,has_maxMultiOps<AddressAlloc>::value>::maxMultiOps;

            typedef AddressAlloc                        allocator_type;
            typedef typename AddressAlloc::size_type    size_type;


            using AddressAlloc::AddressAlloc;
/*
            template<typename... Args>
            address_allocator_traits(void* reservedSpc, size_t alignOff, typename AddressAlloc::size_type bufSz, Args&&... args) noexcept :
                    AddressAlloc(reservedSpc, nullptr, alignOff, bufSz, std::forward<Args>(args)...)
            {
            }
*/

            static inline size_type        get_real_addr(const AddressAlloc& alloc, size_type allocated_addr) noexcept
            {
                return impl::address_allocator_traits_base<AddressAlloc,has_func_get_real_addr<AddressAlloc>::value>::get_real_addr(allocated_addr);
            }


            static inline void              multi_alloc_addr(AddressAlloc& alloc, uint32_t count, size_type* outAddresses,
                                                             const size_type* bytes, const size_type* alignment, const size_type* hint=nullptr) noexcept
            {
                constexpr uint32_t maxMultiOps = impl::address_allocator_traits_base<AddressAlloc,has_maxMultiOps<AddressAlloc>::value>::maxMultiOps;
                for (uint32_t i=0; i<count; i+=maxMultiOps)
                    impl::address_allocator_traits_base<AddressAlloc,has_func_multi_alloc_addr<AddressAlloc>::value>::multi_alloc_addr(
                                                                alloc,std::min(count-i,maxMultiOps),outAddresses+i,bytes+i,alignment+i,hint ? (hint+i):nullptr);
            }

            static inline void             multi_free_addr(AddressAlloc& alloc, uint32_t count, const size_type* addr, const size_type* bytes) noexcept
            {
                constexpr uint32_t maxMultiOps = impl::address_allocator_traits_base<AddressAlloc,has_maxMultiOps<AddressAlloc>::value>::maxMultiOps;
                for (uint32_t i=0; i<count; i+=maxMultiOps)
                    impl::address_allocator_traits_base<AddressAlloc,has_func_multi_free_addr<AddressAlloc>::value>::multi_free_addr(
                                                                alloc,std::min(count-i,maxMultiOps),addr+i,bytes+i);
            }

            static inline size_type        get_free_size(const AddressAlloc& alloc) noexcept
            {
                return alloc.get_free_size();
            }
            static inline size_type        get_allocated_size(const AddressAlloc& alloc) noexcept
            {
                return alloc.get_allocated_size();
            }
            static inline size_type        get_total_size(const AddressAlloc& alloc) noexcept
            {
                return alloc.get_total_size();
            }
    };

}
}

#endif // __IRR_ADDRESS_ALLOCATOR_TYPE_TRAITS_H_INCLUDED__
