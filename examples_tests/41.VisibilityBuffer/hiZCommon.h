
#ifndef _HI_Z_COMMON_H_INCLUDED_
#define _HI_Z_COMMON_H_INCLUDED_

#ifdef __cplusplus
#define mat4 core::matrix4SIMD
#define uvec2 core::vector2d<uint32_t>
#define uint uint32_t
#endif

struct SHiZPushConstants
{
    mat4 vp;
    uvec2 lvl0MipExtent;
    uint maxBatchCnt;
};

#ifdef __cplusplus
#undef mat4
#undef uvec2
#undef uint
#endif

#endif