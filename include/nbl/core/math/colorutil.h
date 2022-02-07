// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_COLOR_UTIL_H_INCLUDED__
#define __NBL_CORE_COLOR_UTIL_H_INCLUDED__

#include <cstdlib>

namespace nbl
{
namespace core
{
inline double lin2srgb(double _lin)
{
    if(_lin <= 0.0031308)
        return _lin * 12.92;
    return 1.055 * pow(_lin, 1. / 2.4) - 0.055;
}

inline double srgb2lin(double _s)
{
    if(_s <= 0.04045)
        return _s / 12.92;
    return pow((_s + 0.055) / 1.055, 2.4);
}

}
}

#endif