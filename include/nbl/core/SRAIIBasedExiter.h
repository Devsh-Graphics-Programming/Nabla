// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_SRAIIBASEDEXITER_H_INCLUDED__
#define __NBL_CORE_SRAIIBASEDEXITER_H_INCLUDED__

#include "stddef.h"
#include "nbl/core/Types.h"

/*! \file SRAIIBasedExiter.h
	\brief File containing SRAIIBasedExiter utility struct for invoking functions upon leaving a scope.
*/

namespace nbl
{
namespace core
{
template<typename F>
class SRAIIBasedExiter
{
    F onDestr;

public:
    SRAIIBasedExiter(F&& _exitFn)
        : onDestr{std::move(_exitFn)} {}
    SRAIIBasedExiter(const F& _exitFn)
        : onDestr{_exitFn} {}

    SRAIIBasedExiter& operator=(F&& _exitFn)
    {
        onDestr = std::move(_exitFn);
        return *this;
    }
    SRAIIBasedExiter& operator=(const F& _exitFn)
    {
        onDestr = _exitFn;
        return *this;
    }

    ~SRAIIBasedExiter() { onDestr(); }
};

template<typename F>
SRAIIBasedExiter<std::decay_t<F>> makeRAIIExiter(F&& _exitFn)
{
    return SRAIIBasedExiter<std::decay_t<F>>(std::forward<F>(_exitFn));
}

}
}

#endif