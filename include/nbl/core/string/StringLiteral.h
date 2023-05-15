// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_STRING_LITERAL_H_INCLUDED_
#define _NBL_CORE_STRING_LITERAL_H_INCLUDED_

#include <algorithm>

namespace nbl::core
{

template<size_t N>
struct StringLiteral
{
    constexpr StringLiteral(const char (&str)[N])
    {
        std::copy_n(str, N, value);
    }

    char value[N];
};

}

// for compatibility's sake
#define NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(STRING_LITERAL) nbl::core::StringLiteral(STRING_LITERAL)

#endif // _NBL_CORE_STRING_LITERAL_H_INCLUDED_
