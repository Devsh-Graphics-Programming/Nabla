// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_PHMAP_SERIALIZATION_H_INCLUDED__
#define __NBL_ASSET_PHMAP_SERIALIZATION_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/asset/ICPUBuffer.h"

namespace nbl
{
namespace asset
{
class CBufferPhmapOutputArchive
{
public:
    CBufferPhmapOutputArchive(const SBufferRange<ICPUBuffer>& _buffer)
    {
        bufferPtr = static_cast<uint8_t*>(_buffer.buffer.get()->getPointer()) + _buffer.offset;
    }

    // TODO: protect against writing out of bounds as defined by SBufferRange
    bool dump(const char* p, size_t sz)
    {
        memcpy(bufferPtr, p, sz);
        bufferPtr += sz;

        return true;
    }

    template<typename V>
    typename std::enable_if<phmap::type_traits_internal::IsTriviallyCopyable<V>::value, bool>::type dump(const V& v)
    {
        memcpy(bufferPtr, reinterpret_cast<const uint8_t*>(&v), sizeof(V));
        bufferPtr += sizeof(V);

        return true;
    }

private:
    uint8_t* bufferPtr;
};

}
}

#endif