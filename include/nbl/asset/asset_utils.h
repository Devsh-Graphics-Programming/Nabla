// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_UTILS_H_INCLUDED__
#define __NBL_ASSET_UTILS_H_INCLUDED__

// TODO: this whole header should die

#include "nbl/asset/ICPUBuffer.h"
#include <algorithm>

namespace nbl
{
namespace asset
{
inline void fillBufferWithDWORD(ICPUBuffer* _buf, uint32_t _val)
{
    const size_t dwCnt = _buf->getSize() / 4ull;
    const size_t rem = _buf->getSize() - (dwCnt * 4ull);

    uint32_t* dwptr = reinterpret_cast<uint32_t*>(_buf->getPointer());
    std::fill(dwptr, dwptr + dwCnt, _val);
    memcpy(dwptr + dwCnt, &_val, rem);
}

inline void fillBufferWithDeadBeef(ICPUBuffer* _buf)
{
    fillBufferWithDWORD(_buf, 0xdeadbeefu);
}

#include "nbl/nblpack.h"
//! Designed for use with interface blocks declared with `layout (row_major, std140)`
// TODO: change members to core::matrix3x4SIMD and core::matrix4SIMD
struct SBasicViewParameters
{
    float MVP[4 * 4];
    //! Might be used for Model matrix just as well
    float MV[3 * 4];
    //! 3x3 but each row is padded to 4 floats (16 bytes), so that actually we have 3x4
    //! so we have 3 floats free for use (last element of each row) which can be used for storing some vec3 (most likely camera position)
    //! This vec3 is accessible in GLSL by accessing 4th column with [] operator
    float NormalMat[3 * 3 + 3];
} PACK_STRUCT;
#include "nbl/nblunpack.h"

}
}

#endif