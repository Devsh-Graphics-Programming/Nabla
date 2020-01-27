#ifndef __IRR_ASSET_UTILS_H_INCLUDED__
#define __IRR_ASSET_UTILS_H_INCLUDED__

#include "irr/asset/ICPUBuffer.h"
#include <algorithm>

namespace irr {
namespace asset
{

inline void fillBufferWithDWORD(ICPUBuffer* _buf, uint32_t _val)
{
    const size_t dwCnt = _buf->getSize()/4ull;
    const size_t rem = _buf->getSize()-(dwCnt*4ull);

    uint32_t* dwptr = reinterpret_cast<uint32_t*>(_buf->getPointer());
    std::fill(dwptr, dwptr+dwCnt, _val);
    memcpy(dwptr+dwCnt, &_val, rem);
}

inline void fillBufferWithDeadBeef(ICPUBuffer* _buf)
{
    fillBufferWithDWORD(_buf, 0xdeadbeefu);
}

#include "irr/irrpack.h"
//! Designed for use with interface blocks declared with `layout (row_major, std140)`
struct SBasicViewParameters
{
    float MVP[4*4];
    //! Might be used for Model matrix just as well
    float MV[3*4];
    //! 3x3 but each row is padded to 4 floats (16 bytes), so that actually we have 3x4
    //! so we have 3 floats free for use (last element of each row) which can be used for storing some vec3 (most likely camera position)
    //! This vec3 is accessible in GLSL by accessing 4th column with [] operator
    float NormalMat[3*3+3];
} PACK_STRUCT;
#include "irr/irrunpack.h"

}}

#endif//__IRR_ASSET_UTILS_H_INCLUDED__