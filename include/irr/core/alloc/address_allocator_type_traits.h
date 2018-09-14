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


    template<class AddressAlloc>
    class address_allocator_traits
    {
        private:
            template<class U> using func_resize_address_range       = decltype(U::resize_address_range(nullptr,0ull,0u));
        protected:
            template<class,class=void> struct has_func_resize_address_range                                 : std::false_type {};
            template<class U> struct has_func_resize_address_range<U,void_t<func_resize_address_range<U> > >: std::is_same<func_resize_address_range<U>,void> {};
    };

}
}

#endif // __IRR_ADDRESS_ALLOCATOR_TYPE_TRAITS_H_INCLUDED__
