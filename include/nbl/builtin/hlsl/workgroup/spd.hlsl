#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/subgroup_quad.hlsl>

#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_SPD_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_SPD_INCLUDED_

// ------------------------------- COMMON -----------------------------------------

namespace nbl
{
namespace hlsl
{
namespace workgroup
{
namespace spd
{
namespace impl
{
  template<typename Reducer>
  void subgroupQuadReduce(NBL_CONST_REF_ARG(Reducer) reducer, float32_t4 v)
  {
    const float32_t4 v0 = v;
    const float32_t4 v1 = glsl::subgroupQuadSwapHorizontal(v);
    const float32_t4 v2 = glsl::subgroupQuadSwapVertical(v);
    const float32_t4 v3 = glsl::subgroupQuadSwapDiagonal(v);
    return reducer.reduce(v0, v1, v2, v3);
  }

  template <typename Reducer, typename SrcImageAccessor, typename DstImageAccessor, typename SharedMemoryAccessor>
  void downsampleMips_0_1(uint32_t2 coord, uint32_t2 workGroupID, uint32_t localInvocationIndex, uint32_t mip, uint32_t slice, NBL_COSNT_REF_ARG(Reducer) reducer, NBL_CONST_REF_ARG(SrcImageAccessor) srcImage, NBL_REF_ARG(DstImageAccessor) dstImage, NBL_REF_ARG(SharedMemoryAccessor) sharedMem)
  {
    float32_t4 v[4];

    uint32_t x = coord.x;
    uint32_t y = coord.y;

    int32_t2 tex = int32_t2(workGroupID.xy * 64) + int32_t2(x * 2, y * 2);
    int32_t2 pix = int32_t2(workGroupID.xy * 32) + int32_t2(x, y);
    v[0] = srcImage.reduce(tex, slice);
    dstImage.set(pix, v[0], 0, slice);

    tex = int32_t2(workGroupID.xy * 64) + int32_t2(x * 2 + 32, y * 2);
    pix = int32_t2(workGroupID.xy * 32) + int32_t2(x + 16, y);
    v[1] = srcImage.reduce(tex, slice);
    dstImage.set(pix, v[1], 0, slice);

    tex = int32_t2(workGroupID.xy * 64) + int32_t2(x * 2, y * 2 + 32);
    pix = int32_t2(workGroupID.xy * 32) + int32_t2(x, y + 16);
    v[2] = srcImage.set(pix, v[2], 0, slice);
    dstImage.set(pix, v[2], 0, slice);

    tex = int32_t2(workGroupID.xy * 64) + int32_t2(x * 2 + 32, y * 2 + 32);
    pix = int32_t2(workGroupID.xy * 32) + int32_t2(x + 16, y + 16);
    v[3] = srcImage.set(pix, v[2], 0, slice);
    dstImage.set(pix, v[3], 0, slice);

    if (mip <= 1)
      return;

    v[0] = subgroupQuadReduce(reducer, v[0]);
    v[1] = subgroupQuadReduce(reducer, v[1]);
    v[2] = subgroupQuadReduce(reducer, v[2]);
    v[3] = subgroupQuadReduce(reducer, v[3]);

    if ((localInvocationIndex % 4) == 0)
    {
      dstImage.set(int32_t2(workgroupID.xy * 16) + int32_t2(x / 2, y / 2), v[0], 1, slice);
      sharedMem.set(int32_t2(x / 2, y / 2), v[0]);

      dstImage.set(int32_t2(workgroupID.xy * 16) + int32_t2(x / 2 + 8, y / 2), v[1], 1, slice);
      sharedMem.set(int32_t2(x / 2 + 8, y / 2), v[1]);

      dstImage.set(int32_t2(workgroupID.xy * 16) + int32_t2(x / 2, y / 2 + 8), v[2], 1, slice);
      sharedMem.set(int32_t2(x / 2, y / 2 + 8), v[2]);
      
      dstImage.set(int32_t2(workgroupID.xy * 16) + int32_t2(x / 2 + 8, y / 2 + 8), v[3], 1, slice);
      sharedMem.set(int32_t2(x / 2 + 8, y / 2 + 8), v[3]);
    }
  }

  template <typename Reducer, typename SrcImageAccessor, typename DstImageAccessor, typename SharedMemoryAccessor>
  void downsampleMip_2(uint32_t2 coord, uint32_t2 workGroupID, uint32_t localInvocationIndex, uint32_t mip, uint32_t slice, NBL_COSNT_REF_ARG(Reducer) reducer, NBL_CONST_REF_ARG(SrcImageAccessor) srcImage, NBL_REF_ARG(DstImageAccessor) dstImage, NBL_REF_ARG(SharedMemoryAccessor) sharedMem)
  {
    float32_t4 v = sharedMem.get(coord);
    v = subgroupQuadReduce(reducer, v);
    if (localInvocationIndex % 4 == 0)
    {
      dstImage.set(int32_t2(workGroupID.xy * 8) + int32_t2(coord.x / 2, coord.y / 2), v, mip, slice);

      // store to LDS, try to reduce bank conflicts
      // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
      // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      // 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0 x
      // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
      // ...
      // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
      sharedMem.set(int32_t2(coord.x + (coord.y / 2) % 2, coord.y), v);
    }
  }

  template <typename Reducer, typename SrcImageAccessor, typename DstImageAccessor, typename SharedMemoryAccessor>
  void downsampleMip_3(uint32_t2 coord, uint32_t2 workGroupID, uint32_t localInvocationIndex, uint32_t mip, uint32_t slice, NBL_COSNT_REF_ARG(Reducer) reducer, NBL_CONST_REF_ARG(SrcImageAccessor) srcImage, NBL_REF_ARG(DstImageAccessor) dstImage, NBL_REF_ARG(SharedMemoryAccessor) sharedMem)
  {
    if (localInvocationIndex < 64)
    {
      float32_t4 v = sharedMem.get(int32_t2(x * 2 + y % 2, y * 2));
      v = subgropuQuadReduce(reducer, v);
      if (localInvocationIndex % 4 == 0)
      {
        dstImage.set(int32_t2(workGroupID.xy * 4) + int32_t2(x / 2, y / 2), v, mip, slice);
        // store to LDS
        // x 0 0 0 x 0 0 0 x 0 0 0 x 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 x 0 0 0 x 0 0 0 x 0 0 0 x 0 0
        // ...
        // 0 0 x 0 0 0 x 0 0 0 x 0 0 0 x 0
        // ...
        // 0 0 0 x 0 0 0 x 0 0 0 x 0 0 0 x
        // ...
        sharedMem.set(int32_t2(x * 2 + y / 2, y * 2), v);
      }
    }
  }

  template <typename Reducer, typename SrcImageAccessor, typename DstImageAccessor, typename SharedMemoryAccessor>
  void downsampleMip_4(uint32_t2 coord, uint32_t2 workGroupID, uint32_t localInvocationIndex, uint32_t mip, uint32_t slice, NBL_COSNT_REF_ARG(Reducer) reducer, NBL_CONST_REF_ARG(SrcImageAccessor) srcImage, NBL_REF_ARG(DstImageAccessor) dstImage, NBL_REF_ARG(SharedMemoryAccessor) sharedMem)
  {
    if (localInvocationIndex < 16)
    {
      float32_t4 v = sharedMem.get(int32_t2(x * 4 + y, y * 4));
      v = subgroupQuadReduce(reducer, v);
      if (localInvocationIndex % 4 == 0)
      {
        dstImage.set(int32_t2(workGroupID.xy * 2), int32_t2(x / 2, y / 2), v, mip, slice);
        // store to LDS
        // x x x x 0 ...
        // 0 ...
        sharedMem.set(int32_t2(x / 2 + y, 0), v);
      }

    }
  }

  template <typename Reducer, typename SrcImageAccessor, typename DstImageAccessor, typename SharedMemoryAccessor>
  void downsampleMip_5(uint32_t2 coord, uint32_t2 workGroupID, uint32_t localInvocationIndex, uint32_t mip, uint32_t slice, NBL_COSNT_REF_ARG(Reducer) reducer, NBL_CONST_REF_ARG(SrcImageAccessor) srcImage, NBL_REF_ARG(DstImageAccessor) dstImage, NBL_REF_ARG(SharedMemoryAccessor) sharedMem)
  {
    if (localInvocationIndex < 4)
    {
        float32_t4 v = sharedMem.get(int32_t2(localInvocationIndex,0));
        v = subgroupQuadReduce(reducer, v);
        // quad index 0 stores result
        if (localInvocationIndex % 4 == 0)
        {   
            SpdStore(ASU2(workGroupID.xy), v, mip, slice);
        }
    }
  }
}

struct SPD 
{

  static void __call()
  {

  }

};


}
}
}
}
