// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_I_BUFFER_H_INCLUDED__
#define __NBL_CORE_I_BUFFER_H_INCLUDED__

#include "nbl/core/decl/Types.h"
#include "nbl/core/IReferenceCounted.h"
#include "nbl/asset/ECommonEnums.h"

namespace nbl::core
{

struct adopt_memory_t {};
constexpr adopt_memory_t adopt_memory{};

class IBuffer : public virtual IReferenceCounted
{
    public:
        //! size in BYTES
        virtual const uint64_t& getSize() const = 0;

    protected:
        _NBL_INTERFACE_CHILD(IBuffer) {}
};

} // end namespace nbl::video

#endif

