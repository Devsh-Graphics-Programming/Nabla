// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_CORE_C_TO_UNDERLYING_H_INCLUDED_
#define _NBL_CORE_C_TO_UNDERLYING_H_INCLUDED_

namespace nbl::core
{
    #if defined(__cpp_lib_to_underlying)
        using to_underlying = std::to_underlying;
    #else
        template<typename E>
        constexpr auto to_underlying(E e) -> typename std::underlying_type<E>::type 
        {
            return static_cast<typename std::underlying_type<E>::type>(e);
        }
    #endif
}

#endif