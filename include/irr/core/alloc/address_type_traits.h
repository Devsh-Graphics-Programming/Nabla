// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_ADDRESS_TYPE_TRAITS_H_INCLUDED__
#define __IRR_ADDRESS_TYPE_TRAITS_H_INCLUDED__

namespace irr
{
namespace core
{

    template<typename AddressType>
    struct address_type_traits;

    template<>
    struct address_type_traits<uint32_t>
    {
        static constexpr uint32_t   invalid_address = 0xdeadbeefu;
    };

    template<>
    struct address_type_traits<uint64_t>
    {
        static constexpr uint64_t   invalid_address = 0xdeadbeefBADC0FFEull;
    };

}
}

#endif
