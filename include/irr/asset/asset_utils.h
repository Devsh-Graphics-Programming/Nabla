#ifndef __IRR_ASSET_UTILS_H_INCLUDED__
#define __IRR_ASSET_UTILS_H_INCLUDED__

#include "irr/asset/ICPUBuffer.h"
#include <algorithm>

namespace irr {
namespace asset
{

void fillBufferWithDword(ICPUBuffer* _buf, uint32_t _val)
{
    const size_t dwCnt = _buf->getSize()/4ull;
    const size_t rem = _buf->getSize()-(dwCnt*4ull);

    uint32_t* dwptr = reinterpret_cast<uint32_t*>(_buf->getPointer());
    std::fill(dwptr, dwptr+dwCnt, _val);
    memcpy(dwptr+dwCnt, &_val, rem);
}

void fillBufferWithDeadBeef(ICPUBuffer* _buf)
{
    fillBufferWithDword(_buf, 0xdeadbeefu);
}

}}

#endif//__IRR_ASSET_UTILS_H_INCLUDED__