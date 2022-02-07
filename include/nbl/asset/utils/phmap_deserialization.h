// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_PHMAP_DESERIALIZATION_H_INCLUDED__
#define __NBL_ASSET_PHMAP_DESERIALIZATION_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/asset/ICPUBuffer.h"

namespace nbl
{
namespace asset
{
class CBufferPhmapInputArchive
{
public:
    CBufferPhmapInputArchive(const SBufferRange<const ICPUBuffer>& _buffer)
    {
        buffPtr = reinterpret_cast<const uint8_t*>(_buffer.buffer.get()->getPointer()) + _buffer.offset;
    }

    // TODO: protect against reading out of the buffer range
    bool load(char* p, size_t sz)
    {
        memcpy(p, buffPtr, sz);
        buffPtr += sz;

        return true;
    }

    template<typename V>
    typename std::enable_if<phmap::type_traits_internal::IsTriviallyCopyable<V>::value, bool>::type load(V* v)
    {
        memcpy(reinterpret_cast<uint8_t*>(v), buffPtr, sizeof(V));
        buffPtr += sizeof(V);

        return true;
    }

private:
    const uint8_t* buffPtr;
};

}
}

#endif