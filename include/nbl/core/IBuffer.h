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

        enum E_USAGE_FLAGS : uint32_t
        {
            EUF_TRANSFER_SRC_BIT = 0x00000001,
            EUF_TRANSFER_DST_BIT = 0x00000002,
            EUF_UNIFORM_TEXEL_BUFFER_BIT = 0x00000004,
            EUF_STORAGE_TEXEL_BUFFER_BIT = 0x00000008,
            EUF_UNIFORM_BUFFER_BIT = 0x00000010,
            EUF_STORAGE_BUFFER_BIT = 0x00000020,
            EUF_INDEX_BUFFER_BIT = 0x00000040,
            EUF_VERTEX_BUFFER_BIT = 0x00000080,
            EUF_INDIRECT_BUFFER_BIT = 0x00000100
        };

        struct SCreationParams
        {
            uint64_t size;
            E_USAGE_FLAGS usage;
            asset::E_SHARING_MODE sharingMode;
            uint32_t queueFamilyIndexCount;
            const uint32_t* queuueFamilyIndices;
        };

    protected:
        _NBL_INTERFACE_CHILD(IBuffer) {}
};

} // end namespace nbl::video

#endif

