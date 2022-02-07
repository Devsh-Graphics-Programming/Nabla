// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_CORE_ATOMIC_H_INCLUDED_
#define _NBL_CORE_ATOMIC_H_INCLUDED_

#include "nbl/core/Types.h"

#include <atomic>

namespace nbl::core
{
template<typename T>
using atomic = std::atomic<T>;

template<typename T>
typename atomic<T>::value_type atomic_fetch_max(atomic<T>* obj, typename atomic<T>::value_type value) noexcept
{
    auto prev_value = std::atomic_load(obj);
    while(value > prev_value && !std::atomic_compare_exchange_weak(obj, &prev_value, value))
    {
    }
    return prev_value;
}

template<typename T>
typename atomic<T>::value_type atomic_fetch_min(atomic<T>* obj, typename atomic<T>::value_type value) noexcept
{
    auto prev_value = std::atomic_load(obj);
    while(value < prev_value && !std::atomic_compare_exchange_weak(obj, &prev_value, value))
    {
    }
    return prev_value;
}

}

#endif
