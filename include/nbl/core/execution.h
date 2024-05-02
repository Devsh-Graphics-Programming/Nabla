// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_CORE_EXECUTION_H_INCLUDED_
#define _NBL_CORE_EXECUTION_H_INCLUDED_

#if __has_include (<execution>)
#include <execution>
#include <algorithm>
#else
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include "oneapi/dpl/pstl/execution_defs.h"
#include "oneapi/dpl/pstl/glue_algorithm_defs.h"
#include "oneapi/dpl/pstl/glue_algorithm_ranges_defs.h"
#endif

#define ALIAS_TEMPLATE_FUNCTION(highLevelF, lowLevelF) \
template<typename... Args> \
inline auto highLevelF(Args&&... args) -> decltype(lowLevelF(std::forward<Args>(args)...)) \
{ \
    return lowLevelF(std::forward<Args>(args)...); \
}

namespace nbl::core
{
#if __has_include(<execution>)
namespace execution = std::execution;

ALIAS_TEMPLATE_FUNCTION(for_each_n, std::for_each_n)
ALIAS_TEMPLATE_FUNCTION(for_each, std::for_each)
ALIAS_TEMPLATE_FUNCTION(swap_ranges, std::swap_ranges)
ALIAS_TEMPLATE_FUNCTION(nth_element, std::nth_element)
//template <class _ExPo, class _FwdIt, class _Diff, class _Fn>
//const auto for_each_n = std::for_each_n<_ExPo, _FwdIt, _Diff, _Fn>;
//
//template <class _ExPo, class _FwdIt, class _Fn>
//const auto for_each = std::for_each<_ExPo, _FwdIt, _Fn>;
//
//template <class _ExPo, class _FwdIt1, class _FwdIt2>
//const auto swap_ranges = std::swap_ranges<_ExPo, _FwdIt1, _FwdIt2>;
#else
namespace execution = oneapi::dpl::execution;

ALIAS_TEMPLATE_FUNCTION(for_each_n, oneapi::dpl::for_each_n)
ALIAS_TEMPLATE_FUNCTION(for_each, oneapi::dpl::for_each)
ALIAS_TEMPLATE_FUNCTION(swap_ranges, oneapi::dpl::swap_ranges)
ALIAS_TEMPLATE_FUNCTION(nth_element, oneapi::dpl::nth_element)
//template <class _ExPo, class _FwdIt, class _Diff, class _Fn>
//const auto for_each_n = oneapi::dpl::for_each_n<_ExPo, _FwdIt, _Diff, _Fn>;
//
//template <class _ExPo, class _FwdIt, class _Fn>
//const auto for_each = oneapi::dpl::for_each<_ExPo, _FwdIt, _Fn>;
//
//template <class _ExPo, class _FwdIt1, class _FwdIt2>
//const auto swap_ranges = oneapi::dpl::swap_ranges<_ExPo, _FwdIt1, _FwdIt2>;
#endif
}

#undef ALIAS_TEMPLATE_FUNCTION

#endif

